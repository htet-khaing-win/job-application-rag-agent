"""Per-node model/key assignment (the "profile") and the runtime that resolves it.

The profile stores only *which provider/model/key alias* each node uses — never secrets.
Secrets are resolved from the encrypted vault at call time.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from providers import (PROVIDERS, ProviderError, build_chat_model, build_embeddings,
                       build_pinecone, build_tavily)
from resilience import ResilientLLM
from vault import APP_DIR, KeyVault

# role -> (kind, human label, description)
ROLES: Dict[str, Dict[str, str]] = {
    "jd_analyst":   {"kind": "chat",      "label": "JD Analyst",    "desc": "Validates and distils the job description"},
    "writer":       {"kind": "chat",      "label": "Copywriter",    "desc": "Drafts the cover letter"},
    "critic":       {"kind": "chat",      "label": "Hiring Manager", "desc": "Critiques the draft"},
    "editor":       {"kind": "chat",      "label": "Editor",        "desc": "Applies edits and fixes flow"},
    "verifier":     {"kind": "chat",      "label": "Fact Checker",  "desc": "Repairs letters that fail grounding checks"},
    "embeddings":   {"kind": "embedding", "label": "Embeddings",    "desc": "Resume/JD vectors (must match index dimension)"},
    "researcher":   {"kind": "search",    "label": "Researcher",    "desc": "Live company research"},
    "vector_store": {"kind": "vector",    "label": "Vector Store",  "desc": "Resume index"},
}

_ENV_FALLBACK = {  # only honoured when JOBAGENT_ALLOW_ENV_KEYS=1 (CI / headless)
    "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "tavily": "TAVILY_API_KEY", "pinecone": "PINECONE_API_KEY",
}


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
        "embeddings":   [Binding("openai", "text-embedding-3-small")],
        "researcher":   [Binding("tavily", "advanced")],
        "vector_store": [Binding("pinecone", "resume-index")],
    }


@dataclass
class Profile:
    roles: Dict[str, List[Binding]] = field(default_factory=_default_roles)

    def validate(self) -> None:
        for role, bindings in self.roles.items():
            if role not in ROLES:
                raise ValueError(f"Unknown role '{role}'.")
            if not bindings:
                raise ValueError(f"Role '{role}' needs at least one model.")
            for b in bindings:
                b.validate(ROLES[role]["kind"])

    def to_json(self) -> str:
        return json.dumps({r: [asdict(b) for b in bs] for r, bs in self.roles.items()}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Profile":
        raw = json.loads(text)
        base = _default_roles()
        for role, items in raw.items():
            if role in ROLES:
                base[role] = [Binding(**{k: v for k, v in i.items() if k in Binding.__annotations__}) for i in items]
        prof = cls(base)
        prof.validate()
        return prof

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Profile":
        path = path or (APP_DIR / "profile.json")
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        self.validate()
        path = path or (APP_DIR / "profile.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


class Runtime:
    """Resolves bindings -> live clients using vault-held keys."""

    def __init__(self, vault: KeyVault, profile: Profile):
        self.vault = vault
        self.profile = profile

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

    def missing_keys(self) -> List[str]:
        """Roles whose *primary* binding has no resolvable key."""
        out = []
        for role, bindings in self.profile.roles.items():
            b = bindings[0]
            if PROVIDERS[b.provider].needs_key and not self.resolve_key(b):
                out.append(f"{ROLES[role]['label']} ({b.provider})")
        return out

    # -- clients -------------------------------------------------------------
    def chat(self, role: str) -> ResilientLLM:
        cands = []
        for b in self.profile.roles[role]:
            label = f"{b.provider}:{b.model}"
            cands.append((label, lambda b=b: build_chat_model(
                b.provider, b.model, self.resolve_key(b), b.temperature)))
        return ResilientLLM(role, cands)

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
