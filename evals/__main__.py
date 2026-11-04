"""Eval CLI.   python -m evals <command> ...

  retrieval  compare two ranked-list files against a gold set (CIs, Wilcoxon, Holm, effect sizes),
             or collect them live:   --collect {baseline,hybrid} [--alpha A] [--cold-bm25]
  letters    claim-level faithfulness of generated letters (--entailer lexical|llm|model)
  chaos      fault-injection run of the self-healing graph with fake models (no keys needed)
  ablate     switch pipeline components off one at a time (LIVE: spends provider money)
  sweep      hybrid alpha sweep with cross-validated selection (LIVE)
  label      make-sheet | build-gold    gold-set tooling (labelling itself is human work)
  perf       p50/p95 latency, tokens and dollars per node/model from the run log
  regress    fail (exit 1) when tracked metrics regress against evals/baseline.json

Legacy flag-style calls still work and are routed with a notice:
  python -m evals --gold g.json --runs-a a.json --runs-b b.json   ->  retrieval ...
  python -m evals --collect hybrid --gold g.json --out h.json     ->  retrieval --collect ...
  python -m evals --letters letters.json                          ->  letters ...

gold_set.json : {"queries": [{"id", "jd", "namespace", "relevant_chunk_ids": [...]}]}
runs file     : {"<query id>": ["<chunk id>", ...]}   ranked
letters file  : [{"letter", "resume_text", "skills": [...]}]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evals import regress
from evals.metrics import compare, evaluate_run, flatten_rows
from evals.report import build_report, save_report
from evals.spans import evaluate_run_spans


def _read(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _runtime() -> Any:
    from runtime import default_runtime
    return default_runtime()


def _apply_bm25_mode(cold: bool) -> str:
    if not cold:
        return "fitted"
    import database
    if hasattr(database, "set_bm25_mode"):
        database.set_bm25_mode("cold")
    else:  # older database.py: install a cold encoder directly
        from pinecone_text.sparse import BM25Encoder
        database._bm25 = BM25Encoder.default()
    return "cold"


def _emit(report: Dict[str, Any], args: argparse.Namespace) -> None:
    for w in report["warnings"]:
        print(f"\nWARNING: {w}")
    if not getattr(args, "no_report", False):
        print(f"\nReport written: {save_report(report, args.report_dir)}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--report-dir", default="evals/reports")
    p.add_argument("--no-report", action="store_true")


# ------------------------------------------------------------------------------------ retrieval
def collect(gold: Dict[str, Any], method: str, k: int, alpha: float) -> Dict[str, Any]:
    """Live ranked lists. Hybrid scales dense by alpha and sparse by (1 - alpha), the documented Pinecone
    approach (Pinecone's query API has no `alpha` argument)."""
    from database import get_bm25, get_embeddings, get_pinecone_index
    from runtime import activate_runtime
    activate_runtime(_runtime())
    emb, bm25, index = get_embeddings(), get_bm25(), get_pinecone_index()
    out: Dict[str, Any] = {}
    spans_mode = _is_span_gold(gold)   # span gold needs chunk text to match against, not just ids
    for q in gold["queries"]:
        dense = emb.embed_query(q["jd"])
        kw: Dict[str, Any] = dict(top_k=k, include_metadata=spans_mode, namespace=q["namespace"])
        if method == "hybrid":
            sp = bm25.encode_queries(q["jd"])
            kw.update(vector=[v * alpha for v in dense],
                      sparse_vector={"indices": sp["indices"], "values": [v * (1 - alpha) for v in sp["values"]]})
        else:
            kw.update(vector=dense)
        matches = index.query(**kw)["matches"]
        out[q["id"]] = ([{"id": m["id"], "text": (m.get("metadata") or {}).get("text", "")} for m in matches]
                        if spans_mode else [m["id"] for m in matches])
    return out


def _is_span_gold(gold: Dict[str, Any]) -> bool:
    """Schema v2 gold (evals.goldgen): relevance = evidence spans matched against chunk text."""
    return any("relevant_spans" in q for q in gold["queries"])


def _bucket_means(gold: Dict[str, Any], per_query: Dict[str, List[float]], metric: str) -> Dict[str, float]:
    scorable = [q for q in gold["queries"] if q.get("relevant_spans")]
    out: Dict[str, float] = {}
    for b in ("high", "medium", "low"):
        vals = [v for q, v in zip(scorable, per_query[metric]) if q.get("bucket") == b]
        if vals:
            out[f"{b} (n={len(vals)})"] = round(sum(vals) / len(vals), 4)
    return out


def cmd_retrieval(a: argparse.Namespace) -> int:
    mode = _apply_bm25_mode(a.cold_bm25)
    if a.collect:
        gold = _read(a.gold)
        runs = collect(gold, a.collect, a.k, a.alpha)
        Path(a.out or f"{a.collect}.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
        print(f"Saved {len(runs)} ranked lists (alpha={a.alpha if a.collect == 'hybrid' else 'n/a'}, bm25={mode}).")
        return 0
    if not (a.gold and a.runs_a and a.runs_b):
        print("retrieval needs --gold, --runs-a and --runs-b (or --collect).")
        return 2
    gold = _read(a.gold)
    if _is_span_gold(gold):
        truth = {q["id"]: q["relevant_spans"] for q in gold["queries"] if q.get("relevant_spans")}
        base = evaluate_run_spans(_read(a.runs_a), gold["queries"], a.k)
        cand = evaluate_run_spans(_read(a.runs_b), gold["queries"], a.k)
    else:
        truth = {q["id"]: q["relevant_chunk_ids"] for q in gold["queries"]}
        base = evaluate_run(_read(a.runs_a), truth, a.k)
        cand = evaluate_run(_read(a.runs_b), truth, a.k)
    rows = compare(base, cand)
    print(f"{'metric':<14}{'base':>8}{'cand':>8}{'95% CI':>20}{'delta':>8}{'p_perm':>8}{'p_wilc':>8}{'p_holm':>8}"
          f"{'dz':>7}{'sig':>5}")
    for r in rows:
        print(f"{r['metric']:<14}{r['baseline']:>8}{r['candidate']:>8}"
              f"{'[%s, %s]' % (r['ci_lo'], r['ci_hi']):>20}{r['delta']:>8}{r['p_value']:>8}{r['p_wilcoxon']:>8}"
              f"{r['p_holm']:>8}{r['dz']:>7}{'*' if r['significant'] else '':>5}")
    warns = []
    if _is_span_gold(gold):
        nd = f"ndcg@{a.k}"
        print()
        print(f"{nd} by coverage bucket   baseline: {_bucket_means(gold, base, nd)}")
        print(f"{'':<{len(nd) + 22}}candidate: {_bucket_means(gold, cand, nd)}")
        warns.append(f"labels are {gold.get('meta', {}).get('labeling', 'unspecified')}.")
    if len(truth) < 30:
        warns.append(f"only {len(truth)} queries; confidence intervals will be wide.")
    report = build_report("retrieval", {"n_queries": len(truth), "rows": rows}, flatten_rows(rows),
                          seeds={"bootstrap": 0, "permutation": 0}, bm25_mode=mode,
                          evaluated_namespaces=sorted({q["namespace"] for q in gold["queries"]}), warnings=warns)
    _emit(report, a)
    return 0


# ------------------------------------------------------------------------------------ letters
def cmd_letters(a: argparse.Namespace) -> int:
    from evals.faithfulness import aggregate, aggregate_claims, claim_support_rate, score_letter
    from evals.nli import get_entailer
    llm = _runtime().chat("verifier") if a.entailer == "llm" else None
    ent = get_entailer(a.entailer, llm)
    items = _read(a.letters)
    claims = [claim_support_rate(i["letter"], i["resume_text"], ent) for i in items]
    agg = aggregate_claims(claims)
    sections: Dict[str, Any] = {"claims": agg}
    metrics = {"faithfulness.mean_hallucination_rate": {"value": agg["mean_hallucination_rate"], "direction": "lower"},
               "faithfulness.mean_support_rate": {"value": agg["mean_support_rate"], "direction": "higher"}}
    if all("skills" in i for i in items):
        sections["skill_vocab"] = aggregate([score_letter(i["letter"], i["resume_text"], i["skills"]) for i in items])
    print("Claim-level faithfulness:", json.dumps(agg))
    if "skill_vocab" in sections:
        print("Skill-vocabulary check:  ", json.dumps(sections["skill_vocab"]))
    _emit(build_report("letters", sections, metrics, seeds={}, warnings=[
        "lexical entailment is conservative; use --entailer llm for paraphrase-robust scoring"]
        if a.entailer == "lexical" else []), a)
    return 0


# ------------------------------------------------------------------------------------ chaos
def cmd_chaos(a: argparse.Namespace) -> int:
    from evals.chaos import format_table, run_chaos
    res = run_chaos(trials=a.trials, seed=a.seed, backoff_s=a.backoff)
    print(format_table(res))
    s = res["summary"]
    metrics = {"chaos.recovery_rate": {"value": s["recovery_rate"], "direction": "higher"},
               "chaos.fallback_rate": {"value": s["fallback_rate"], "direction": "lower"},
               "chaos.secret_leaks": {"value": s["secret_leaks"], "direction": "lower", "tolerance": 0}}
    _emit(build_report("chaos", res, metrics, seeds={"seed": a.seed, "trials": a.trials}), a)
    return 1 if s["secret_leaks"] else 0


# ------------------------------------------------------------------------------------ ablate
def cmd_ablate(a: argparse.Namespace) -> int:
    from evals import ablation
    from evals.adversarial import stuffing_cases
    from evals.nli import get_entailer
    rt = _runtime()
    cases = stuffing_cases() if a.adversarial else _read(a.cases)
    cases = cases["cases"] if isinstance(cases, dict) else cases
    n_runs = len(cases) * len(ablation.DEFAULT_ABLATIONS) * a.repeats
    print(f"LIVE run: {n_runs} pipeline executions with your provider keys (this costs money).")
    ent = get_entailer(a.entailer, rt.chat("verifier") if a.entailer == "llm" else None)
    judge = rt.chat("judge") if a.judge else None
    gen_ids = [f"{b.provider}:{b.model}" for b in rt.profile.roles["writer"]]
    res = ablation.run_ablation(cases, ablation.live_runner(rt), repeats=a.repeats, entailer=ent,
                                judge_llm=judge, generator_ids=gen_ids, seed=a.seed)
    print(ablation.format_table(res))
    full = res["summary"].get("full", {})
    metrics: Dict[str, Dict[str, Any]] = {}
    if full:
        metrics["ablation.full.hallucination_rate"] = {"value": full["hallucination_rate"]["mean"], "direction": "lower"}
        metrics["ablation.full.delivery_rate"] = {"value": full["delivery_rate"], "direction": "higher"}
    fc = res["fact_checker_effect"]
    if fc.get("n_cases"):
        metrics["ablation.fact_checker.reduction"] = {"value": fc["reduction"], "direction": "higher"}
    cost = sum(r["cost_usd"] for r in res["records"])
    res = {k: v for k, v in res.items() if k != "records"} | {"records_n": len(res["records"])}  # no letters/PII
    _emit(build_report("ablation", res, metrics, seeds={"seed": a.seed, "repeats": a.repeats}, profile=rt.profile,
                       cost_usd=cost), a)
    return 0


# ------------------------------------------------------------------------------------ sweep
def cmd_sweep(a: argparse.Namespace) -> int:
    from evals import sweep
    mode = _apply_bm25_mode(a.cold_bm25)
    gold = _read(a.gold)
    alphas = [float(x) for x in a.alphas.split(",")]
    res = sweep.sweep_alpha(gold, alphas, a.k, sweep.live_retriever(_runtime()), folds=a.folds, seed=a.seed)
    print(sweep.format_table(res))
    metrics = {}
    if res["cv_ndcg"] is not None:
        metrics[f"sweep.cv_ndcg@{a.k}"] = {"value": res["cv_ndcg"], "direction": "higher"}
    _emit(build_report("sweep", res, metrics, seeds={"seed": a.seed, "folds": a.folds}, bm25_mode=mode,
                       evaluated_namespaces=sorted({q["namespace"] for q in gold["queries"]})), a)
    return 0


# ------------------------------------------------------------------------------------ label
def cmd_label(a: argparse.Namespace) -> int:
    from evals import label
    if a.action == "make-sheet":
        rt = _runtime()
        jds = _read(a.jds)
        namespaces = a.namespaces
        if not namespaces:
            from database import list_stored_resumes
            from runtime import activate_runtime
            activate_runtime(rt)
            namespaces = list_stored_resumes()
        sheet = label.make_sheet(jds, namespaces, label.live_fetcher(rt), a.top_n)
        label.save(sheet, a.out)
        print(f"Sheet with {len(sheet['items'])} item(s) written to {a.out}. A human must now fill every "
              "'relevant' and 'supported' field (true/false).")
        return 0
    sheets = [label.load(p) for p in a.sheets]
    try:
        gold = label.build_gold(sheets, adjudicate=a.adjudicate, min_queries=a.min_queries)
    except label.LabelError as e:
        print(f"ERROR: {e}")
        return 2
    label.save(gold, a.out)
    for w in gold["meta"]["warnings"]:
        print(f"WARNING: {w}")
    if gold["meta"]["agreement"]:
        ag = gold["meta"]["agreement"]
        print(f"Inter-annotator: chunk kappa={ag['chunk_relevance']['kappa']} "
              f"skill kappa={ag['skill_support']['kappa']} disagreements={len(ag['disagreements'])}")
    print(f"Gold set: {len(gold['queries'])} queries -> {a.out}")
    return 0


# ------------------------------------------------------------------------------------ perf
def cmd_perf(a: argparse.Namespace) -> int:
    from evals import perf
    if a.file:
        runs = [json.loads(ln) for ln in Path(a.file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        try:
            from runlog import read_runs
        except ImportError:
            print("runlog module not available; pass --file runs.jsonl")
            return 2
        runs = list(read_runs())
    if not runs:
        print("no runs recorded yet")
        return 0
    s = perf.summarize(runs)
    print(perf.format_table(s))
    return 0


def cmd_regress(a: argparse.Namespace) -> int:
    return regress.run(a.baseline, a.current, a.tolerance, a.update_baseline, a.allow_missing)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m evals", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("retrieval", help="compare / collect ranked lists")
    p.add_argument("--gold"); p.add_argument("--runs-a"); p.add_argument("--runs-b")
    p.add_argument("--collect", choices=["baseline", "hybrid"]); p.add_argument("--out")
    p.add_argument("-k", type=int, default=5); p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--cold-bm25", action="store_true", help="use an unfitted BM25 encoder (no corpus leakage)")
    _add_common(p); p.set_defaults(fn=cmd_retrieval)

    p = sub.add_parser("letters", help="claim-level faithfulness")
    p.add_argument("--letters", required=True)
    p.add_argument("--entailer", choices=["lexical", "llm", "model"], default="lexical")
    _add_common(p); p.set_defaults(fn=cmd_letters)

    p = sub.add_parser("chaos", help="fault-injection harness (fake models)")
    p.add_argument("--trials", type=int, default=10); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backoff", type=float, default=0.0, help="retry backoff base seconds (0 = instant)")
    _add_common(p); p.set_defaults(fn=cmd_chaos)

    p = sub.add_parser("ablate", help="component ablations (LIVE)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cases"); g.add_argument("--adversarial", action="store_true")
    p.add_argument("--repeats", type=int, default=3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge", action="store_true"); p.add_argument("--entailer", choices=["lexical", "llm", "model"], default="lexical")
    _add_common(p); p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("sweep", help="alpha sweep (LIVE)")
    p.add_argument("--gold", required=True); p.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("-k", type=int, default=5); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--cold-bm25", action="store_true")
    _add_common(p); p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("label", help="gold-set tooling")
    ls = p.add_subparsers(dest="action", required=True)
    m = ls.add_parser("make-sheet")
    m.add_argument("--jds", required=True, help='JSON list of {"id","jd"}'); m.add_argument("--namespaces", nargs="*")
    m.add_argument("--top-n", type=int, default=10); m.add_argument("--out", default="evals/label_sheet.json")
    b = ls.add_parser("build-gold")
    b.add_argument("--sheets", nargs="+", required=True); b.add_argument("--out", default="evals/gold_set.json")
    b.add_argument("--adjudicate", choices=["error", "first", "intersection"], default="error")
    b.add_argument("--min-queries", type=int, default=30)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("perf", help="latency / tokens / cost percentiles")
    p.add_argument("--file", help="JSONL of run records (default: runlog)"); p.set_defaults(fn=cmd_perf)

    p = sub.add_parser("regress", help="CI regression gate")
    p.add_argument("--baseline", default="evals/baseline.json"); p.add_argument("--current", required=True)
    p.add_argument("--tolerance", type=float, default=0.02); p.add_argument("--update-baseline", action="store_true")
    p.add_argument("--allow-missing", action="store_true"); p.set_defaults(fn=cmd_regress)
    return ap


def _migrate_legacy(argv: List[str]) -> List[str]:
    if not argv or not argv[0].startswith("-") or argv[0] in ("-h", "--help"):
        return argv
    if "--letters" in argv:
        target = "letters"
    elif any(f in argv for f in ("--collect", "--runs-a", "--runs-b", "--gold")):
        target = "retrieval"
    else:
        return argv
    print(f"[evals] flag-style invocation is deprecated; running as `python -m evals {target} ...`", file=sys.stderr)
    return [target] + argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(_migrate_legacy(list(sys.argv[1:] if argv is None else argv)))
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
