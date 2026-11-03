"""Eval runner.

  python -m evals --gold evals/gold_set.json --runs-a baseline.json --runs-b hybrid.json
  python -m evals --gold evals/gold_set.json --collect hybrid --out hybrid.json     (live; needs vault keys)
  python -m evals --letters letters.json                                             (faithfulness)

gold_set.json : {"queries": [{"id": "q1", "jd": "...", "namespace": "resume.pdf",
                              "relevant_chunk_ids": ["resume.pdf#3", "resume.pdf#7"]}]}
runs file     : {"q1": ["resume.pdf#3", "resume.pdf#1", ...]}   (ranked chunk ids)
letters file  : [{"letter": "...", "resume_text": "...", "skills": ["python", "kubernetes"]}]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from evals.faithfulness import aggregate, score_letter
from evals.metrics import compare, evaluate_run


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def collect(gold: Dict[str, Any], method: str, k: int) -> Dict[str, List[str]]:
    """Run live retrieval for every gold query (baseline = dense only, hybrid = dense+BM25)."""
    from database import get_bm25, get_embeddings, get_pinecone_index
    from runtime import activate_runtime, default_runtime
    activate_runtime(default_runtime())
    emb, bm25, index = get_embeddings(), get_bm25(), get_pinecone_index()
    out: Dict[str, List[str]] = {}
    for q in gold["queries"]:
        dense = emb.embed_query(q["jd"])
        kwargs: Dict[str, Any] = dict(vector=dense, top_k=k, include_metadata=False, namespace=q["namespace"])
        if method == "hybrid":
            kwargs.update(sparse_vector=bm25.encode_queries(q["jd"]), alpha=0.4)
        res = index.query(**kwargs)
        out[q["id"]] = [m["id"] for m in res["matches"]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="evals")
    ap.add_argument("--gold")
    ap.add_argument("--runs-a")
    ap.add_argument("--runs-b")
    ap.add_argument("--collect", choices=["baseline", "hybrid"])
    ap.add_argument("--out")
    ap.add_argument("--letters")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--report-dir", default="evals/reports")
    args = ap.parse_args()
    report: Dict[str, Any] = {"git_sha": _git_sha(), "date": dt.datetime.now(dt.timezone.utc).isoformat(), "k": args.k}

    if args.collect:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        runs = collect(gold, args.collect, args.k)
        Path(args.out or f"{args.collect}.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
        print(f"Saved {len(runs)} ranked lists.")
        return 0

    if args.gold and args.runs_a and args.runs_b:
        gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        truth = {q["id"]: q["relevant_chunk_ids"] for q in gold["queries"]}
        a = evaluate_run(json.loads(Path(args.runs_a).read_text(encoding="utf-8")), truth, args.k)
        b = evaluate_run(json.loads(Path(args.runs_b).read_text(encoding="utf-8")), truth, args.k)
        rows = compare(a, b)
        report["retrieval"] = {"n_queries": len(truth), "rows": rows}
        print(f"{'metric':<14}{'base':>8}{'cand':>8}{'95% CI':>20}{'delta':>8}{'p':>8}")
        for r in rows:
            print(f"{r['metric']:<14}{r['baseline']:>8}{r['candidate']:>8}"
                  f"{'[%s, %s]' % (r['ci_lo'], r['ci_hi']):>20}{r['delta']:>8}{r['p_value']:>8}")
        if len(truth) < 30:
            print(f"\nWARNING: only {len(truth)} queries; confidence intervals will be wide.")

    if args.letters:
        items = json.loads(Path(args.letters).read_text(encoding="utf-8"))
        scores = [score_letter(i["letter"], i["resume_text"], i["skills"]) for i in items]
        report["faithfulness"] = aggregate(scores)
        print("Faithfulness:", report["faithfulness"])

    if len(report) > 3:
        d = Path(args.report_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{report['git_sha']}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written: {path}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
