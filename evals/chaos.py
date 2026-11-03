"""Fault-injection harness for the self-healing graph (`python -m evals chaos`).

Runs the REAL compiled graph with fake models that fail in controlled, seeded ways and measures
what the self-healing layer actually delivers:

  recovery_rate    letters delivered / trials, over scenarios where delivery is expected
  degraded_rate    delivered letters that needed graceful degradation (e.g. critic skipped)
  fallback_rate    UNEXPECTED fallbacks / trials (a clean fallback when every writer is dead is correct)
  time_to_recover  wall-clock mean / p95 of trials that hit at least one fault
  secret_leaks     occurrences of a planted API key in results, stdout, logs or healing events (must be 0)

Providers are fakes, so this proves the recovery *logic*; it does not prove real providers behave
like these faults. Real-provider chaos needs live keys.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from evals.fakes import (SECRET, FaultyModel, ScriptedModel, always, first_n, malformed_json_once,
                         stub_backends, with_probability, JD)
from evals.metrics import mean, percentile


@dataclass
class Scenario:
    name: str
    expected: str  # clean | recover | degrade | fallback
    build: Callable[[int], Tuple[Any, Any]]  # seed -> (generator_llm, critic_llm)
    description: str = ""


def _rl(role: str, *models: Any) -> Any:
    from resilience import ResilientLLM

    def factory(m: Any) -> Callable[[], Any]:
        return lambda: m

    return ResilientLLM(role, [(f"{role}-m{i}", factory(m)) for i, m in enumerate(models)])


def _good() -> ScriptedModel:
    return ScriptedModel(model_id="good")


def default_scenarios() -> List[Scenario]:
    timeout = TimeoutError(f"request timed out (key={SECRET})")
    rate = RuntimeError(f"429 rate limit exceeded for key {SECRET}")
    auth = RuntimeError(f"401 invalid api key {SECRET}")
    down = ConnectionError(f"503 service unavailable ({SECRET})")

    def faulty(inner: Any, fault: Any) -> FaultyModel:
        return FaultyModel(inner, fault)

    return [
        Scenario("control", "clean", lambda s: (_rl("writer", _good()), _rl("critic", _good())),
                 "no faults"),
        Scenario("timeout_burst", "recover",
                 lambda s: (_rl("writer", faulty(_good(), first_n(2, timeout))), _rl("critic", _good())),
                 "writer times out twice, third attempt succeeds"),
        Scenario("rate_limit_burst", "recover",
                 lambda s: (_rl("writer", _good()), _rl("critic", faulty(_good(), first_n(1, rate)))),
                 "critic-side models get a 429 on first call"),
        Scenario("malformed_json", "recover",
                 lambda s: (_rl("writer", _good()), _rl("critic", faulty(_good(), malformed_json_once()))),
                 "JSON stages return broken JSON once; repair prompt recovers"),
        Scenario("auth_failure_primary", "recover",
                 lambda s: (_rl("writer", faulty(_good(), always(auth)), _good()), _rl("critic", _good())),
                 "primary writer key rejected; fail over to secondary"),
        Scenario("dead_provider", "recover",
                 lambda s: (_rl("writer", faulty(_good(), always(down)), _good()),
                            _rl("critic", faulty(_good(), always(down)), _good())),
                 "primary provider down for every role; secondaries healthy"),
        Scenario("intermittent", "recover",
                 lambda s: (_rl("writer", faulty(_good(), with_probability(0.35, rate, s)), _good()),
                            _rl("critic", faulty(_good(), with_probability(0.35, timeout, s + 1)), _good())),
                 "35% random failures with a healthy secondary"),
        Scenario("critic_outage", "degrade",
                 lambda s: (_rl("writer", _good()), _rl("critic", faulty(_good(), always(down)))),
                 "all critic-side models dead; pipeline must still deliver"),
        Scenario("total_writer_outage", "fallback",
                 lambda s: (_rl("writer", faulty(_good(), always(down))), _rl("critic", _good())),
                 "every writer model dead; must end in a clean fallback"),
    ]


_RECOVER_MARKERS = ("recovered on", "repaired malformed", "repair attempt")
_DEGRADE_MARKERS = ("degraded gracefully", "skipping critique", "keyword heuristic", "keeping un-refined",
                    "applied deterministic")


def classify(result: Dict[str, Any]) -> str:
    letter = (result.get("cover_letter") or "").strip()
    if result.get("is_fallback") or not letter:
        return "fallback"
    events = " | ".join(result.get("healing_log", []) or []).lower()
    if any(m in events for m in _DEGRADE_MARKERS):
        return "degraded"
    if events and any(m in events for m in _RECOVER_MARKERS):
        return "recovered"
    return "recovered" if events else "clean"


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


def _run_graph(gen: Any, crit: Any, jd: str) -> Dict[str, Any]:
    from graph import build_graph
    from resilience import reset_breakers, start_heal_log
    reset_breakers()
    app = build_graph(gen, crit)

    async def go() -> Dict[str, Any]:
        start_heal_log()
        return dict(await app.ainvoke({"job_description": jd, "company_name": "Acme"}))

    return asyncio.run(go())


def run_chaos(trials: int = 10, seed: int = 0, scenarios: Optional[List[Scenario]] = None,
              backoff_s: float = 0.0, jd: str = JD, secret: str = SECRET) -> Dict[str, Any]:
    """Run every scenario `trials` times with seeds seed..seed+trials-1."""
    import resilience
    scenarios = scenarios or default_scenarios()
    old_backoff = resilience.BACKOFF_BASE_S
    resilience.BACKOFF_BASE_S = backoff_s
    cap = _LogCapture()
    root = logging.getLogger()
    root.addHandler(cap)
    per_scenario: Dict[str, Dict[str, Any]] = {}
    leaks = 0
    recover_times: List[float] = []
    try:
        with stub_backends():
            for sc in scenarios:
                outcomes: List[str] = []
                for t in range(trials):
                    gen, crit = sc.build(seed + t)
                    out_buf = io.StringIO()
                    cap.lines.clear()
                    t0 = time.perf_counter()
                    try:
                        with contextlib.redirect_stdout(out_buf):
                            result = _run_graph(gen, crit, jd)
                        crashed = None
                    except Exception as e:  # noqa: BLE001 - an unhandled crash is itself a finding
                        result, crashed = {"is_fallback": True, "error_message": str(e)}, repr(e)
                    dt = time.perf_counter() - t0
                    blob = "\n".join([json.dumps(result, default=str), out_buf.getvalue(),
                                      "\n".join(cap.lines), crashed or ""])
                    leaks += blob.count(secret) + blob.count(secret[3:23])
                    outcome = "crashed" if crashed else classify(result)
                    outcomes.append(outcome)
                    if outcome in ("recovered", "degraded") and sc.name != "control":
                        recover_times.append(dt)
                per_scenario[sc.name] = {
                    "expected": sc.expected, "description": sc.description, "trials": trials,
                    "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
                    "expectation_met_rate": round(sum(1 for o in outcomes if _meets(sc.expected, o)) / trials, 4),
                }
    finally:
        resilience.BACKOFF_BASE_S = old_backoff
        root.removeHandler(cap)

    delivering = [s for s in scenarios if s.expected != "fallback" and s.name != "control"]
    n = sum(per_scenario[s.name]["trials"] for s in delivering)

    def count(names: List[str], *outs: str) -> int:
        return sum(per_scenario[nm]["outcomes"].get(o, 0) for nm in names for o in outs)

    names = [s.name for s in delivering]
    expected_fb = [s.name for s in scenarios if s.expected == "fallback"]
    summary = {
        "trials_per_scenario": trials, "seed": seed, "backoff_s": backoff_s,
        "recovery_rate": round(count(names, "recovered", "degraded", "clean") / n, 4) if n else 1.0,
        "degraded_rate": round(count(names, "degraded") / n, 4) if n else 0.0,
        "fallback_rate": round((count(names, "fallback") + count(names, "crashed")) / n, 4) if n else 0.0,
        "expected_fallback_ok_rate": round(
            (sum(per_scenario[x]["outcomes"].get("fallback", 0) for x in expected_fb) /
             max(1, sum(per_scenario[x]["trials"] for x in expected_fb))), 4) if expected_fb else 1.0,
        "time_to_recover_mean_s": round(mean(recover_times), 4),
        "time_to_recover_p95_s": round(percentile(recover_times, 95), 4),
        "expectations_met_rate": round(mean([per_scenario[s.name]["expectation_met_rate"] for s in scenarios]), 4),
        "secret_leaks": leaks,
    }
    return {"summary": summary, "scenarios": per_scenario}


def _meets(expected: str, outcome: str) -> bool:
    if expected == "clean":
        return outcome == "clean"
    if expected == "recover":
        return outcome in ("recovered", "clean", "degraded")
    if expected == "degrade":
        return outcome in ("degraded", "recovered")
    if expected == "fallback":
        return outcome == "fallback"
    return False


def format_table(report: Dict[str, Any]) -> str:
    rows = [f"{'scenario':<24}{'expected':<10}{'outcomes':<44}{'met':>6}"]
    for name, s in report["scenarios"].items():
        outs = ", ".join(f"{k}:{v}" for k, v in s["outcomes"].items())
        rows.append(f"{name:<24}{s['expected']:<10}{outs:<44}{s['expectation_met_rate']:>6}")
    rows.append("")
    rows += [f"{k:<28}{v}" for k, v in report["summary"].items()]
    return "\n".join(rows)
