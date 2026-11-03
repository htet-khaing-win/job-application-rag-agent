"""Ablation study (`python -m evals ablate`).

Switch pipeline components off one at a time and measure what each buys you in quality, cost and
latency. The unit of analysis is the CASE (a JD + resume pair); repeats are averaged within a case
so the statistics are paired across cases, not inflated by repeated draws.

The runner is injectable: the live default calls `pipeline.run_pipeline(..., ablate=...)`, tests pass a
fake. Letters are kept in memory only (they derive from resumes); the report never contains them.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from evals.faithfulness import claim_support_rate
from evals.metrics import bootstrap_ci, cohens_dz, mean, paired_permutation_pvalue, wilcoxon_signed_rank_pvalue
from evals.nli import Entailer

COMPONENTS = ("fact_checker", "critique_refine", "research", "failover")

DEFAULT_ABLATIONS: List[FrozenSet[str]] = (
    [frozenset()] + [frozenset({c}) for c in COMPONENTS] + [frozenset(COMPONENTS)]
)

Runner = Callable[[Mapping[str, Any], FrozenSet[str]], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]


def ablation_name(ablate: Iterable[str]) -> str:
    a = sorted(ablate)
    if not a:
        return "full"
    return "all_off" if set(a) == set(COMPONENTS) else "no_" + "+".join(a)


def live_runner(rt: Any, timeout_s: float = 180.0) -> Runner:
    """Runner that executes the real pipeline against the vault-keyed runtime."""
    async def run(case: Mapping[str, Any], ablate: FrozenSet[str]) -> Dict[str, Any]:
        from pipeline import run_pipeline
        return await run_pipeline(rt, case["jd"], case.get("company", "Acme"),
                                  ablate=frozenset(ablate), timeout_s=timeout_s)
    return run


async def _call(runner: Runner, case: Mapping[str, Any], ablate: FrozenSet[str]) -> Dict[str, Any]:
    out = runner(case, ablate)
    return await out if inspect.isawaitable(out) else out  # type: ignore[return-value]


def _latency_s(result: Mapping[str, Any], wall: float) -> float:
    timings = result.get("node_timings") or []
    return round(sum(t.get("ms", 0.0) for t in timings) / 1000.0, 4) if timings else round(wall, 4)


async def run_ablation_async(cases: Sequence[Mapping[str, Any]], runner: Runner,
                             ablations: Optional[Sequence[FrozenSet[str]]] = None, repeats: int = 1,
                             entailer: Optional[Entailer] = None, judge_llm: Any = None,
                             generator_ids: Optional[Iterable[str]] = None, seed: int = 0) -> Dict[str, Any]:
    """cases: [{id, jd, resume_text, company?}]. Returns {"records", "summary", "fact_checker_effect", ...}."""
    ablations = list(ablations if ablations is not None else DEFAULT_ABLATIONS)
    records: List[Dict[str, Any]] = []
    letters: Dict[Tuple[str, str], str] = {}  # (ablation, case) -> first-repeat letter, in memory only
    for ab in ablations:
        name = ablation_name(ab)
        for case in cases:
            for r in range(repeats):
                t0 = time.perf_counter()
                try:
                    res = await _call(runner, case, frozenset(ab))
                    error = None
                except Exception as e:  # noqa: BLE001 - a crash is a data point, not a harness failure
                    res, error = {}, f"{type(e).__name__}"
                wall = time.perf_counter() - t0
                letter = (res.get("cover_letter") or "").strip()
                delivered = bool(letter) and not res.get("is_fallback")
                cs = claim_support_rate(letter, case.get("resume_text", ""), entailer) if delivered else None
                if r == 0 and delivered:
                    letters[(name, str(case["id"]))] = letter
                records.append({
                    "ablation": name, "case": str(case["id"]), "repeat": r, "delivered": delivered,
                    "hallucination_rate": cs["hallucination_rate"] if cs else None,
                    "support_rate": cs["support_rate"] if cs else None,
                    "n_claims": cs["n_claims"] if cs else 0,
                    "cost_usd": float(res.get("cost_usd") or 0.0),
                    "latency_s": _latency_s(res, wall),
                    "error": error,
                })
    summary = _summarise(records, ablations)
    out: Dict[str, Any] = {"records": records, "summary": summary, "repeats": repeats,
                           "fact_checker_effect": fact_checker_effect(records)}
    if judge_llm is not None:
        out["judge"] = _judge_vs_full(judge_llm, cases, letters, ablations, generator_ids, seed)
    return out


def run_ablation(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(run_ablation_async(*args, **kwargs))


def _case_means(records: Sequence[Mapping[str, Any]], ablation: str, field: str) -> Dict[str, float]:
    by_case: Dict[str, List[float]] = {}
    for r in records:
        if r["ablation"] == ablation and r.get(field) is not None:
            by_case.setdefault(r["case"], []).append(float(r[field]))
    return {c: mean(v) for c, v in by_case.items()}


def _summarise(records: Sequence[Mapping[str, Any]], ablations: Sequence[FrozenSet[str]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ab in ablations:
        name = ablation_name(ab)
        rows = [r for r in records if r["ablation"] == name]
        row: Dict[str, Any] = {"n_runs": len(rows),
                               "delivery_rate": round(mean([1.0 if r["delivered"] else 0.0 for r in rows]), 4)}
        for f in ("hallucination_rate", "support_rate", "cost_usd", "latency_s"):
            per_case = list(_case_means(records, name, f).values())
            m, lo, hi = bootstrap_ci(per_case)
            row[f] = {"mean": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n_cases": len(per_case)}
        out[name] = row
    return out


def fact_checker_effect(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Before/after the Fact Checker: hallucination rate with it ablated vs the full pipeline (paired by case)."""
    after = _case_means(records, "full", "hallucination_rate")
    before = _case_means(records, "no_fact_checker", "hallucination_rate")
    cases = sorted(set(after) & set(before))
    if not cases:
        return {"n_cases": 0, "note": "need both 'full' and 'no_fact_checker' runs with delivered letters"}
    b = [before[c] for c in cases]
    a = [after[c] for c in cases]
    diffs = [x - y for x, y in zip(b, a)]  # reduction (positive = Fact Checker helps)
    m, lo, hi = bootstrap_ci(diffs)
    dz = cohens_dz(a, b)
    return {
        "n_cases": len(cases),
        "hallucination_before_fact_checker": round(mean(b), 4),
        "hallucination_after_fact_checker": round(mean(a), 4),
        "reduction": round(m, 4), "reduction_ci_lo": round(lo, 4), "reduction_ci_hi": round(hi, 4),
        "p_wilcoxon": round(wilcoxon_signed_rank_pvalue(a, b), 4),
        "p_permutation": round(paired_permutation_pvalue(a, b), 4),
        "dz": round(dz, 4) if dz not in (float("inf"), float("-inf")) else dz,
    }


def _judge_vs_full(judge_llm: Any, cases: Sequence[Mapping[str, Any]], letters: Mapping[Tuple[str, str], str],
                   ablations: Sequence[FrozenSet[str]], generator_ids: Optional[Iterable[str]],
                   seed: int) -> Dict[str, Any]:
    from evals.judge import run_pairwise_eval
    out: Dict[str, Any] = {}
    for ab in ablations:
        name = ablation_name(ab)
        if name == "full":
            continue
        pairs = [{"id": str(c["id"]), "jd": c["jd"], "x": letters[("full", str(c["id"]))],
                  "y": letters[(name, str(c["id"]))], "resume_text": c.get("resume_text", "")}
                 for c in cases if ("full", str(c["id"])) in letters and (name, str(c["id"])) in letters]
        if pairs:
            r = run_pairwise_eval(judge_llm, pairs, seed=seed, generator_ids=generator_ids)
            out[name] = {k: r[k] for k in ("n", "win_rate_x", "consistency_rate", "position_bias")}
    return out


def format_table(result: Mapping[str, Any]) -> str:
    rows = [f"{'ablation':<34}{'deliv':>7}{'halluc':>9}{'cost $':>9}{'lat s':>8}"]
    for name, s in result["summary"].items():
        rows.append(f"{name:<34}{s['delivery_rate']:>7}{s['hallucination_rate']['mean']:>9}"
                    f"{s['cost_usd']['mean']:>9}{s['latency_s']['mean']:>8}")
    fc = result.get("fact_checker_effect", {})
    if fc.get("n_cases"):
        rows += ["", f"Fact Checker: hallucination {fc['hallucination_before_fact_checker']} -> "
                     f"{fc['hallucination_after_fact_checker']}  (reduction {fc['reduction']} "
                     f"[{fc['reduction_ci_lo']}, {fc['reduction_ci_hi']}], Wilcoxon p={fc['p_wilcoxon']}, "
                     f"n={fc['n_cases']})"]
    if result["summary"] and next(iter(result["summary"].values()))["hallucination_rate"]["n_cases"] < 20:
        rows.append("WARNING: fewer than 20 cases; intervals are wide and p-values fragile.")
    return "\n".join(rows)
