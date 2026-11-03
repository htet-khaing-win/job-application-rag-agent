"""Per-node model/key assignment (the "profile") and the runtime that resolves it.

The profile stores only *which provider/model/key alias* each node uses - never secrets.
Secrets are resolved from the encrypted vault at call time.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from cost import BudgetExceeded, SpendLedger, estimate_cost
from providers import PROVIDERS, ProviderError, build_chat_model, build_embeddings, build_pinecone, build_tavily
from resilience import ChaosModel, ResilientLLM, record_usage
from vault import APP_DIR, KeyVault

# role -> (kind, human label, description)
ROLES: Dict[str, Dict[str, str]] = {
    "jd_analyst":   {"kind": "chat",      "label": "JD Analyst",    "desc": "Validates and distils the job description"},
    "writer":       {"kind": "chat",      "label": "Copywriter",    "desc": "Drafts the cover letter"},
    "critic":       {"kind": "chat",      "label": "Hiring Manager", "desc": "Critiques the draft"},
    "editor":       {"kind": "chat",      "label": "Editor",        "desc": "Applies edits and fixes flow"},
    "verifier":     {"kind": "chat",      "label": "Fact Checker",  "desc": "Repairs letters that fail grounding checks"},
    "judge":        {"kind": "chat",      "label": "Judge",
                     "desc": "Pairwise letter judge (use a different model than the writer)"},
    "embeddings":   {"kind": "embedding", "label": "Embeddings",    "desc": "Resume/JD vectors (must match index dimension)"},
    "researcher":   {"kind": "search",    "label": "Researcher",    "desc": "Live company research"},
    "vector_store": {"kind": "vector",    "label": "Vector Store",  "desc": "Resume index"},
}

_ENV_FALLBACK = {  # only honoured when JOBAGENT_ALLOW_ENV_KEYS=1 (CI / headless)
    "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "tavily": "TAVILY_API_KEY", "pinecone": "PINECONE_API_KEY",
}

DEFAULT_SETTINGS: Dict[str, Any] = {"alpha": 0.4, "max_resumes": 10, "top_k": 10}


@dataclass
class Binding:
    provider: str
    model: str
    key_alias: str = ""        # "" = first vault key stored for this provider
    temperature: float = 0.0

    def validate(self, kind: str) -> None:
        spec = PROVIDERS.get(self.provider)
        if spec is None or kind not in spec.kinds:
            raise ValueError(f"Provider '{self.provider}' cannot serve '{kind}'.")
        if not self.model or len(self.model) > 120:
            raise ValueError("Invalid model name.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be within 0..2.")


def _default_roles() -> Dict[str, List[Binding]]:
    return {
        "jd_analyst":   [Binding("openai", "gpt-4o-mini"), Binding("gemini", "gemini-2.5-flash")],
        "writer":       [Binding("gemini", "gemini-2.5-flash"), Binding("openai", "gpt-4o-mini")],
        "critic":       [Binding("openai", "gpt-4o-mini"), Binding("gemini", "gemini-2.5-flash")],
        "editor":       [Binding("gemini", "gemini-2.5-flash"), Binding("openai", "gpt-4o-mini")],
        "verifier":     [Binding("openai", "gpt-4o-mini"), Binding("gemini", "gemini-2.5-flash")],
        "judge":        [Binding("anthropic", "claude-opus-5-5"), Binding("openai", "gpt-4o")],
        "embeddings":   [Binding("openai", "text-embedding-3-small")],
        "researcher":   [Binding("tavily", "advanced")],
        "vector_store": [Binding("pinecone", "resume-index")],
    }


def validate_settings(settings: Dict[str, Any]) -> None:
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    alpha = settings.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a number within 0..1.")
    for key in ("max_resumes", "top_k"):
        v = settings.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 50:
            raise ValueError(f"{key} must be an integer within 1..50.")


@dataclass
class Profile:
    roles: Dict[str, List[Binding]] = field(default_factory=_default_roles)
    settings: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SETTINGS))

    def validate(self) -> None:
        for role, bindings in self.roles.items():
            if role not in ROLES:
                raise ValueError(f"Unknown role '{role}'.")
            if not bindings:
                raise ValueError(f"Role '{role}' needs at least one model.")
            for b in bindings:
                b.validate(ROLES[role]["kind"])
        validate_settings(self.settings)

    def to_json(self) -> str:
        return json.dumps({"roles": {r: [asdict(b) for b in bs] for r, bs in self.roles.items()},
                           "settings": self.settings}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Profile":
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("Profile JSON must be an object.")
        if isinstance(raw.get("roles"), dict):          # current shape
            raw_roles, raw_settings = raw["roles"], raw.get("settings") or {}
        else:                                            # legacy flat roles-only shape
            raw_roles, raw_settings = raw, {}
        base = _default_roles()
        for role, items in raw_roles.items():
            if role in ROLES:
                base[role] = [Binding(**{k: v for k, v in i.items() if k in Binding.__annotations__}) for i in items]
        settings = dict(DEFAULT_SETTINGS)
        settings.update({k: v for k, v in raw_settings.items() if k in DEFAULT_SETTINGS})
        prof = cls(base, settings)
        prof.validate()
        return prof

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Profile":
        path = path or (APP_DIR / "profile.json")
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        self.validate()
        path = path or (APP_DIR / "profile.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def _int_tokens(meta: Any, key: str) -> int:
    value = meta.get(key, 0) if isinstance(meta, dict) else getattr(meta, key, 0)
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


class Runtime:
    """Resolves bindings -> live clients using vault-held keys."""

    def __init__(self, vault: KeyVault, profile: Profile):
        self.vault = vault
        self.profile = profile
        self.chaos: float = 0.0              # fault-injection probability for chat models (demo/testing)
        self.chaos_seed: Optional[int] = None
        self.failover_enabled: bool = True   # False = primary model only (ablation)
        self._ledger: Optional[SpendLedger] = None

    @property
    def settings(self) -> Dict[str, Any]:
        return self.profile.settings

    # -- key resolution ------------------------------------------------------
    def resolve_key(self, binding: Binding) -> Optional[str]:
        spec = PROVIDERS[binding.provider]
        if not spec.needs_key:
            return None
        if binding.key_alias:
            return self.vault.get(binding.key_alias)
        for entry in self.vault.list():
            if entry["provider"] == binding.provider:
                return self.vault.get(entry["alias"])
        if os.environ.get("JOBAGENT_ALLOW_ENV_KEYS") == "1":
            return os.environ.get(_ENV_FALLBACK.get(binding.provider, ""), None)
        return None

    def resolve_alias(self, binding: Binding) -> str:
        """Vault alias that serves this binding (falls back to the provider name)."""
        if binding.key_alias:
            return binding.key_alias
        try:
            for entry in self.vault.list():
                if entry["provider"] == binding.provider:
                    return str(entry["alias"])
        except Exception:  # noqa: BLE001 - locked vault etc.
            pass
        return binding.provider

    def missing_keys(self) -> List[str]:
        """Roles whose *primary* binding has no resolvable key."""
        if self.vault.is_locked:
            return ["Vault is locked - unlock it to use your keys"]
        out = []
        for role, bindings in self.profile.roles.items():
            b = bindings[0]
            if PROVIDERS[b.provider].needs_key and not self.resolve_key(b):
                out.append(f"{ROLES[role]['label']} ({b.provider})")
        return out

    # -- accounting ----------------------------------------------------------------
    def _get_ledger(self) -> SpendLedger:
        if self._ledger is None:
            self._ledger = SpendLedger(getattr(self.vault, "app_dir", None))
        return self._ledger

    def _make_usage_cb(self, role: str, bindings: Dict[str, Binding]) -> Callable[[str, Any], None]:
        def cb(label: str, response: Any) -> None:
            b = bindings[label]
            meta = getattr(response, "usage_metadata", None) or {}
            in_tok, out_tok = _int_tokens(meta, "input_tokens"), _int_tokens(meta, "output_tokens")
            cost = estimate_cost(b.model, in_tok, out_tok)
            alias = self.resolve_alias(b)
            record_usage({"role": role, "label": label, "alias": alias, "input_tokens": in_tok,
                          "output_tokens": out_tok, "cost_usd": cost})
            try:
                self._get_ledger().add(alias, cost)
            except Exception:  # noqa: BLE001 - ledger failure must never crash a call
                pass
        return cb

    def _make_precheck(self, bindings: Dict[str, Binding]) -> Callable[[str], None]:
        def check(label: str) -> None:
            alias = self.resolve_alias(bindings[label])
            try:
                cap = self.vault.get_cap(alias)
            except Exception:  # noqa: BLE001
                return
            if cap is None:
                return
            try:
                self._get_ledger().check(alias, cap)
            except BudgetExceeded:
                raise
            except Exception:  # noqa: BLE001
                return
        return check

    # -- clients -------------------------------------------------------------
    def chat(self, role: str) -> ResilientLLM:
        bindings = self.profile.roles[role]
        if not self.failover_enabled:
            bindings = bindings[:1]
        by_label: Dict[str, Binding] = {}
        cands = []
        for b in bindings:
            label = f"{b.provider}:{b.model}"
            by_label[label] = b

            def factory(b: Binding = b) -> Any:
                model = build_chat_model(b.provider, b.model, self.resolve_key(b), b.temperature)
                return ChaosModel(model, self.chaos, self.chaos_seed) if self.chaos > 0 else model

            cands.append((label, factory))
        return ResilientLLM(role, cands, usage_cb=self._make_usage_cb(role, by_label),
                            precheck=self._make_precheck(by_label))

    def embeddings(self) -> Any:
        b = self.profile.roles["embeddings"][0]  # no failover: vector dims must stay consistent
        return build_embeddings(b.provider, b.model, self.resolve_key(b))

    def tavily(self) -> Any:
        b = self.profile.roles["researcher"][0]
        return build_tavily(self.resolve_key(b))

    def pinecone(self) -> Any:
        b = self.profile.roles["vector_store"][0]
        return build_pinecone(self.resolve_key(b))

    def index_name(self) -> str:
        return self.profile.roles["vector_store"][0].model


_current: contextvars.ContextVar[Optional[Runtime]] = contextvars.ContextVar("runtime", default=None)


def get_runtime() -> Runtime:
    rt = _current.get()
    if rt is None:
        raise ProviderError("No runtime configured. Add API keys in the GUI/CLI first.")
    return rt


@contextlib.contextmanager
def use_runtime(rt: Runtime) -> Iterator[Runtime]:
    token = _current.set(rt)
    try:
        yield rt
    finally:
        _current.reset(token)


def activate_runtime(rt: Runtime) -> None:
    """Set the runtime for the current context (e.g. at CLI start)."""
    _current.set(rt)


def default_runtime() -> Runtime:
    """Runtime from the on-disk vault + saved profile (for scripts and CLI tools)."""
    return Runtime(KeyVault(), Profile.load())


# --- presets -----------------------------------------------------------------
def _preset(chat: List[Binding], emb: Binding, judge: List[Binding]) -> Dict[str, List[Binding]]:
    base = _default_roles()
    for r in ("jd_analyst", "writer", "critic", "editor", "verifier"):
        base[r] = copy.deepcopy(chat)
    base["judge"] = copy.deepcopy(judge)
    base["embeddings"] = [emb]
    return base


PRESETS: Dict[str, Dict[str, str]] = {
    "Balanced (OpenAI + Gemini failover)": {"desc": "gpt-4o-mini primary, Gemini Flash failover; Claude Opus judge"},
    "Cheap (Gemini Flash only)": {"desc": "Lowest cost, single provider; GPT-4o-mini judge"},
    "Best quality (Claude + GPT)": {"desc": "Claude primary, GPT failover; GPT-4o judge; higher cost"},
    "Fully local (Ollama)": {"desc": "No cloud LLM keys; embeddings stay local too (re-index needed if dimension differs)"},
}


def preset_profile(name: str) -> Profile:
    if name.startswith("Cheap"):
        roles = _preset([Binding("gemini", "gemini-2.5-flash")], Binding("openai", "text-embedding-3-small"),
                        [Binding("openai", "gpt-4o-mini")])
    elif name.startswith("Best"):
        roles = _preset([Binding("anthropic", "claude-sonnet-5-5"), Binding("openai", "gpt-4o")],
                        Binding("openai", "text-embedding-3-small"),
                        [Binding("openai", "gpt-4o"), Binding("gemini", "gemini-2.5-pro")])
    elif name.startswith("Fully local"):
        roles = _preset([Binding("ollama", "qwen2.5:7b"), Binding("ollama", "mistral:7b-instruct")],
                        Binding("ollama", "nomic-embed-text"),
                        [Binding("ollama", "llama3.1:8b"), Binding("ollama", "qwen2.5:7b")])
    else:
        roles = _default_roles()
    prof = Profile(roles)
    prof.validate()
    return prof


# --- connection test -----------------------------------------------------------
def check_connection(rt: Runtime, role: str) -> Dict[str, Any]:
    """Cheap live probe of a role's primary binding. Returns {ok, ms, detail} with secrets redacted."""
    import time

    from security import redact_secrets
    b = rt.profile.roles[role][0]
    kind = ROLES[role]["kind"]
    t0 = time.perf_counter()
    try:
        if PROVIDERS[b.provider].needs_key and not rt.resolve_key(b):
            return {"ok": False, "ms": 0, "detail": "No API key found for this binding."}
        if kind == "chat":
            build_chat_model(b.provider, b.model, rt.resolve_key(b), b.temperature).invoke("Reply with: OK")
        elif kind == "embedding":
            rt.embeddings().embed_query("ping")
        elif kind == "vector":
            rt.pinecone().list_indexes()
        elif kind == "search":
            rt.tavily().search(query="ping", max_results=1)
        return {"ok": True, "ms": round((time.perf_counter() - t0) * 1000), "detail": "Connected"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "ms": round((time.perf_counter() - t0) * 1000),
                "detail": redact_secrets(f"{type(e).__name__}: {e}")[:200]}
