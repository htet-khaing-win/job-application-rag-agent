"""Latency / token / cost summaries from the run log (`python -m evals perf`).

Reads run records written by the pipeline (via runlog.read_runs()). Each record is a dict; this module
accepts `node_timings` (or `timings`): [{node, ms}], `usage`: [{role, model, input_tokens, output_tokens,
cost_usd, ms?}], optional `total_ms`, `cost_usd`, `run_id`. Missing fields are treated as absent, not zero.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from evals.metrics import mean, percentile


def _stats(xs: List[float]) -> Dict[str, float]:
    return {"n": len(xs), "mean": round(mean(xs), 2), "p50": round(percentile(xs, 50), 2),
            "p95": round(percentile(xs, 95), 2), "max": round(max(xs), 2) if xs else 0.0}


def summarize(runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    runs = list(runs)
    by_node: Dict[str, List[float]] = {}
    by_model: Dict[str, Dict[str, Any]] = {}
    totals_ms: List[float] = []
    costs: List[float] = []
    for r in runs:
        timings = r.get("node_timings") or r.get("timings") or []
        for t in timings:
            by_node.setdefault(str(t.get("node")), []).append(float(t.get("ms", 0.0)))
        tot = r.get("total_ms")
        if tot is None and timings:
            tot = sum(float(t.get("ms", 0.0)) for t in timings)
        if tot is not None:
            totals_ms.append(float(tot))
        run_cost = 0.0
        saw_cost = False
        for u in r.get("usage") or []:
            key = f"{u.get('role', '?')} / {u.get('model', '?')}"
            m = by_model.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                                          "_ms": []})
            m["calls"] += 1
            m["input_tokens"] += int(u.get("input_tokens") or 0)
            m["output_tokens"] += int(u.get("output_tokens") or 0)
            c = float(u.get("cost_usd") or 0.0)
            m["cost_usd"] += c
            run_cost += c
            saw_cost = saw_cost or u.get("cost_usd") is not None
            if u.get("ms") is not None:
                m["_ms"].append(float(u["ms"]))
        if r.get("cost_usd") is not None:
            run_cost, saw_cost = float(r["cost_usd"]), True
        if saw_cost:
            costs.append(run_cost)
    models = {}
    for k, v in sorted(by_model.items()):
        ms = v.pop("_ms")
        v["cost_usd"] = round(v["cost_usd"], 6)
        if ms:
            v["latency_ms"] = _stats(ms)
        models[k] = v
    return {
        "runs": len(runs),
        "total_ms": _stats(totals_ms),
        "cost_usd_per_run": {"mean": round(mean(costs), 6), "p50": round(percentile(costs, 50), 6),
                             "p95": round(percentile(costs, 95), 6), "total": round(sum(costs), 6),
                             "n": len(costs)},
        "nodes": {k: _stats(v) for k, v in sorted(by_node.items())},
        "models": models,
    }


def format_table(s: Mapping[str, Any]) -> str:
    rows = [f"runs: {s['runs']}   total latency ms p50={s['total_ms']['p50']} p95={s['total_ms']['p95']}   "
            f"cost/run mean=${s['cost_usd_per_run']['mean']} p95=${s['cost_usd_per_run']['p95']}",
            "", f"{'node':<22}{'n':>5}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}"]
    for k, v in s["nodes"].items():
        rows.append(f"{k:<22}{v['n']:>5}{v['p50']:>10}{v['p95']:>10}{v['max']:>10}")
    rows += ["", f"{'role / model':<40}{'calls':>6}{'in tok':>9}{'out tok':>9}{'cost $':>10}"]
    for k, v in s["models"].items():
        rows.append(f"{k:<40}{v['calls']:>6}{v['input_tokens']:>9}{v['output_tokens']:>9}{v['cost_usd']:>10}")
    return "\n".join(rows)
