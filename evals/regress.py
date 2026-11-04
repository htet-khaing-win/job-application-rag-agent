"""Regression gate: fail CI when a tracked metric gets worse than the stored baseline.

  python -m evals regress --baseline evals/baseline.json --current evals/reports/report_X.json --tolerance 0.02
  python -m evals regress --baseline evals/baseline.json --current <report> --update-baseline

Exit codes: 0 ok / baseline updated, 1 regression, 2 usage or file problem.
Direction comes from the report: 'higher' metrics (recall, nDCG, recovery) regress when they DROP by more
than the tolerance; 'lower' metrics (hallucination rate, secret leaks, cost) regress when they RISE by more.
A metric may carry its own `tolerance` (secret_leaks uses 0). A metric that vanished is a regression
unless --allow-missing.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

EXIT_OK, EXIT_REGRESSION, EXIT_USAGE = 0, 1, 2


def check(baseline: Mapping[str, Any], current: Mapping[str, Any], tolerance: float = 0.02,
          allow_missing: bool = False) -> Dict[str, Any]:
    base_m = baseline.get("metrics", {})
    cur_m = current.get("metrics", {})
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    missing: List[str] = []
    unchanged = 0
    for name, b in base_m.items():
        if name not in cur_m:
            missing.append(name)
            continue
        bv, cv = b.get("value"), cur_m[name].get("value")
        if bv is None or cv is None:
            continue
        direction = b.get("direction", "higher")
        tol = float(b.get("tolerance", tolerance))
        delta = cv - bv
        worse = delta < -tol if direction == "higher" else delta > tol
        better = delta > tol if direction == "higher" else delta < -tol
        row = {"metric": name, "baseline": bv, "current": cv, "delta": round(delta, 6),
               "direction": direction, "tolerance": tol}
        if worse:
            regressions.append(row)
        elif better:
            improvements.append(row)
        else:
            unchanged += 1
    new_metrics = sorted(set(cur_m) - set(base_m))
    ok = not regressions and (allow_missing or not missing)
    return {"ok": ok, "regressions": regressions, "improvements": improvements, "missing": missing,
            "new_metrics": new_metrics, "unchanged": unchanged, "tolerance": tolerance}


def format_result(res: Mapping[str, Any]) -> str:
    lines = []
    for r in res["regressions"]:
        lines.append(f"REGRESSION  {r['metric']}: {r['baseline']} -> {r['current']} "
                     f"({r['direction']} is better, tol {r['tolerance']})")
    for m in res["missing"]:
        lines.append(f"MISSING     {m} (in baseline, absent from current)")
    for r in res["improvements"]:
        lines.append(f"improved    {r['metric']}: {r['baseline']} -> {r['current']}")
    if res["new_metrics"]:
        lines.append(f"new metrics (not gated): {', '.join(res['new_metrics'])}")
    lines.append(f"unchanged within tolerance: {res['unchanged']}")
    lines.append("RESULT: " + ("OK" if res["ok"] else "FAIL"))
    return "\n".join(lines)


def run(baseline_path: str, current_path: str, tolerance: float = 0.02, update: bool = False,
        allow_missing: bool = False) -> int:
    cur = Path(current_path)
    if not cur.exists():
        print(f"current report not found: {current_path}")
        return EXIT_USAGE
    if update:
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cur, baseline_path)
        print(f"baseline updated: {baseline_path}")
        return EXIT_OK
    base = Path(baseline_path)
    if not base.exists():
        print(f"baseline not found: {baseline_path} (create it with --update-baseline)")
        return EXIT_USAGE
    try:
        res = check(json.loads(base.read_text(encoding="utf-8")), json.loads(cur.read_text(encoding="utf-8")),
                    tolerance, allow_missing)
    except (json.JSONDecodeError, OSError) as e:
        print(f"could not read reports: {e}")
        return EXIT_USAGE
    print(format_result(res))
    return EXIT_OK if res["ok"] else EXIT_REGRESSION
