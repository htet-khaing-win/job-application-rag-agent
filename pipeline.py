"""Single entry point used by the CLI, the GUI and the eval harness."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from graph import build_graph, validate_ablate
from resilience import drain_usage, set_token_sink, start_heal_log, start_usage_log
from runtime import Runtime, use_runtime
from security import MAX_COMPANY_CHARS, install_log_redaction, looks_like_injection, sanitize_text

Event = dict[str, Any]
_USAGE_FIELDS = ("node", "role", "model", "label", "alias", "tokens_in", "tokens_out",
                 "input_tokens", "output_tokens", "cost_usd", "usd")


def initial_state(job_description: str, company: str) -> dict[str, Any]:
    return {
        "job_description": sanitize_text(job_description),
        "company_name": sanitize_text(company, MAX_COMPANY_CHARS),
    }


def _num(entry: dict[str, Any], *keys: str) -> float:
    for k in keys:
        v = entry.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


_EVAL_ONLY_ROLES = ("Judge",)   # roles the generation pipeline never calls


def pipeline_missing_keys(rt: Runtime) -> list[str]:
    """Missing keys that actually block a run (the eval-only judge role is ignored)."""
    return [m for m in rt.missing_keys() if not m.startswith(tuple(f"{r} (" for r in _EVAL_ONLY_ROLES))]


def _alias_for(rt: Runtime, binding: Any) -> str | None:
    if getattr(binding, "key_alias", ""):
        return binding.key_alias
    for row in rt.vault.list():
        if row["provider"] == binding.provider:
            return row["alias"]
    return None


def check_budget(rt: Runtime, jd_chars: int) -> None:
    """Refuse to start a run whose estimated cost exceeds a key's remaining monthly cap."""
    try:
        from cost import SpendLedger, estimate_run_cost
    except ImportError:  # cost tracking not installed: nothing to enforce
        return
    caps = {r["alias"]: r["monthly_cap_usd"] for r in rt.vault.list() if r.get("monthly_cap_usd") is not None}
    if not caps:
        return
    est = estimate_run_cost(rt.profile, jd_chars)
    need: dict[str, float] = {}
    for role, info in est["per_role"].items():
        alias = _alias_for(rt, rt.profile.roles[role][0])
        if alias is not None and alias in caps:
            need[alias] = need.get(alias, 0.0) + float(info["usd"])
    ledger = SpendLedger(getattr(rt.vault, "app_dir", None))
    for alias, usd in need.items():
        remaining = float(caps[alias]) - ledger.month_total(alias)
        if usd > remaining:
            raise ValueError(f"budget: key '{alias}' has ${max(remaining, 0):.2f} left this month; "
                             f"this run is estimated at ${usd:.2f}.")


async def _execute(app: Any, state: dict[str, Any], config: dict[str, Any],
                   on_event: Callable[[Event], None] | None) -> dict[str, Any]:
    if on_event is None:
        return dict(await app.ainvoke(state, config=config))
    final: Any = None
    async for mode, chunk in app.astream(state, config=config, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
            continue
        for node_name, update in (chunk or {}).items():
            events = [{"node": t["node"], "ms": t["ms"]} for t in (update or {}).get("node_timings", [])]
            for ev in events or [{"node": node_name, "ms": None}]:
                try:
                    on_event(ev)
                except Exception:  # noqa: BLE001, S110
                    pass
    if final is None:
        raise RuntimeError("Pipeline produced no state.")
    return final.model_dump() if hasattr(final, "model_dump") else dict(final)


def _models(rt: Runtime) -> dict[str, str]:
    try:
        return {role: f"{bs[0].provider}:{bs[0].model}" for role, bs in rt.profile.roles.items()}
    except Exception:  # noqa: BLE001
        return {}


def _log(entry: dict[str, Any]) -> None:
    try:
        from runlog import log_run
        log_run(entry)
    except Exception:  # noqa: BLE001, S110
        pass


async def run_pipeline(rt: Runtime, job_description: str, company: str,
                       generator_llm: Any | None = None, critic_llm: Any | None = None, *,
                       ablate: Iterable[str] = frozenset(), timeout_s: float = 180.0,
                       token_sink: Callable[[str | None], None] | None = None,
                       on_event: Callable[[Event], None] | None = None) -> dict[str, Any]:
    """Validate input, run the self-healing graph under `rt`, return the final state dict.

    Extra keys on the result: run_id, usage, cost_usd, duration_ms (and node_timings / healing_log
    from the graph). `timeout_s` cancels the graph at its next await point; an LLM call that is
    blocking inside a synchronous provider SDK is bounded by that SDK's own timeout instead.
    """
    install_log_redaction()
    ablate = validate_ablate(ablate)
    state = initial_state(job_description, company)
    if not state["job_description"]:
        raise ValueError("Job description is empty.")
    if not state["company_name"]:
        raise ValueError("Company name is required.")
    pinned = generator_llm is not None or critic_llm is not None
    if not pinned:
        missing = pipeline_missing_keys(rt)
        if missing:
            raise ValueError("Missing API keys for: " + ", ".join(missing))
        check_budget(rt, len(state["job_description"]))

    run_id = uuid.uuid4().hex[:12]
    injection = looks_like_injection(state["job_description"])
    app = build_graph(generator_llm, critic_llm, ablate)
    prev_failover = getattr(rt, "failover_enabled", True)
    if "failover" in ablate:
        rt.failover_enabled = False
    t0 = time.perf_counter()
    leftover: list[dict[str, Any]] = []
    config = {"run_name": "job_application_run", "tags": ["langgraph", "async", "cover-letter", "self-healing"],
              "metadata": {"has_company": True, "run_id": run_id}}
    try:
        with use_runtime(rt):
            start_heal_log()
            start_usage_log()
            if token_sink is not None:
                set_token_sink("writer", token_sink)
            try:
                result = await asyncio.wait_for(_execute(app, state, config, on_event), timeout_s)
            finally:
                set_token_sink("writer", None)
                leftover = drain_usage()
    except TimeoutError:
        _log({"run_id": run_id, "ok": False, "error_type": "timeout", "ablate": sorted(ablate),
              "models": _models(rt), "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        raise TimeoutError(f"Pipeline exceeded {timeout_s:.0f}s and was cancelled.") from None
    except Exception as e:
        _log({"run_id": run_id, "ok": False, "error_type": type(e).__name__, "ablate": sorted(ablate),
              "models": _models(rt), "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        raise
    finally:
        rt.failover_enabled = prev_failover

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    usage = list(result.get("node_usage", [])) + [{**u, "node": "(unattributed)"} for u in leftover]
    usage = [{k: v for k, v in u.items() if k in _USAGE_FIELDS} for u in usage]
    result["usage"] = usage
    result["cost_usd"] = round(sum(_num(u, "cost_usd", "usd") for u in usage), 6)
    result["run_id"] = run_id
    result["duration_ms"] = duration_ms
    if injection:
        result.setdefault("healing_log", []).append(
            "[security] possible prompt-injection text detected in JD; treated as data")

    _log({
        "run_id": run_id, "ok": not result.get("is_fallback", False), "fallback": bool(result.get("is_fallback")),
        "error_type": result.get("error_type", ""), "models": _models(rt), "ablate": sorted(ablate),
        "node_timings": result.get("node_timings", []), "usage": usage, "cost_usd": result["cost_usd"],
        "tokens_in": int(sum(_num(u, "tokens_in", "input_tokens") for u in usage)),
        "tokens_out": int(sum(_num(u, "tokens_out", "output_tokens") for u in usage)),
        "healing_events": len(result.get("healing_log", [])), "duration_ms": duration_ms,
        "refinement_count": result.get("refinement_count", 0),
        "letter_verified": bool(result.get("letter_verified")),
    })
    return result
