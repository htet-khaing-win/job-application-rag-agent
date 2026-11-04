"""Sweep, gold-set tooling, perf summaries, run reports, regression gate and the CLI."""
import json
import random

import pytest

from evals import label, perf, regress, sweep
from evals.__main__ import _apply_bm25_mode, _migrate_legacy, main
from evals.report import bm25_warning, build_report, model_ids, save_report


# ----------------------------------------------------------------- sweep
def _gold(n):
    return {"queries": [{"id": f"q{i}", "jd": f"jd {i}", "namespace": "r.pdf", "relevant_chunk_ids": [f"r.pdf#{i}"]}
                        for i in range(n)]}


def retriever_best_at(best, noisy=False, seed=0):
    rng = random.Random(seed)

    def retrieve(q, alpha, k):
        rel = q["relevant_chunk_ids"][0]
        if noisy:
            return [rel] + ["x"] if rng.random() < 0.5 else ["x", rel]
        return [rel, "x", "y"] if abs(alpha - best) < 1e-9 else ["x", "y", "z"]
    return retrieve


def test_kfold_partitions_every_index_exactly_once():
    folds = sweep.kfold_indices(23, 5, seed=1)
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(23)) and len(folds) == 5
    assert sweep.kfold_indices(23, 5, 1) == folds and sweep.kfold_indices(23, 5, 2) != folds


def test_sweep_finds_the_alpha_that_actually_works():
    res = sweep.sweep_alpha(_gold(20), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 3, retriever_best_at(0.6))
    assert res["best_alpha_in_sample"] == 0.6 and res["in_sample_ndcg"] == 1.0
    assert res["cv_ndcg"] == 1.0 and res["cv_selected_alphas"] == [0.6] * 5
    assert res["optimism"] == 0.0
    row = next(r for r in res["table"] if r["alpha"] == 0.6)
    assert row["ndcg@3"] == 1.0 and row["ci_lo"] == row["ci_hi"] == 1.0
    assert next(r for r in res["table"] if r["alpha"] == 0.0)["ndcg@3"] == 0.0


def test_sweep_cv_exposes_selection_optimism_on_pure_noise():
    """With no real signal, the in-sample 'best' alpha looks better than what held-out queries get."""
    res = sweep.sweep_alpha(_gold(12), [0.0, 0.25, 0.5, 0.75, 1.0], 2,
                            lambda q, a, k: ([q["relevant_chunk_ids"][0], "x"]
                                             if random.Random(hash((q["id"], a)) & 0xFFFF).random() < 0.5 else ["x", "y"]))
    assert res["optimism"] is not None and res["optimism"] >= 0
    assert "optimistically biased" in res["caveat"] and "below 30" in res["caveat"]


def test_sweep_small_gold_set_skips_cv_and_validates_input():
    res = sweep.sweep_alpha(_gold(3), [0.4, 0.6], 3, retriever_best_at(0.6))
    assert res["cv_ndcg"] is None and res["optimism"] is None
    with pytest.raises(ValueError):
        sweep.sweep_alpha({"queries": []}, [0.4], 3, retriever_best_at(0.4))


def test_sweep_table_text_mentions_caveat():
    res = sweep.sweep_alpha(_gold(10), [0.2, 0.6], 3, retriever_best_at(0.6))
    text = sweep.format_table(res)
    assert "cross-validated ndcg" in text and "NOTE:" in text


# ----------------------------------------------------------------- label round trip
JDS = [{"id": "j1", "jd": "Python and SQL engineer. Python pipelines."},
       {"id": "j2", "jd": "Kubernetes platform engineer"}]


def fake_fetch(jd, ns, n):
    return [{"chunk_id": f"{ns}#{i}", "text": f"chunk {i} of {ns}"} for i in range(n)]


def fill(sheet, relevant_ids, supported=()):
    for it in sheet["items"]:
        for c in it["chunks"]:
            c["relevant"] = c["chunk_id"] in relevant_ids
        for s in it["skills"]:
            s["supported"] = s["skill"] in supported
    return sheet


def test_make_sheet_structure_and_unlabelled_state():
    sheet = label.make_sheet(JDS, ["a.pdf", "b.pdf"], fake_fetch, top_n=3)
    assert len(sheet["items"]) == 4 and sheet["version"] == label.SHEET_VERSION
    first = sheet["items"][0]
    assert [c["chunk_id"] for c in first["chunks"]] == ["a.pdf#0", "a.pdf#1", "a.pdf#2"]
    assert all(c["relevant"] is None for c in first["chunks"]) and "python" in [s["skill"] for s in first["skills"]]
    errors, _ = label.validate_sheet(sheet)
    assert errors and "unlabelled" in errors[0]


def test_build_gold_refuses_unlabelled_sheet():
    with pytest.raises(label.LabelError, match="not fully labelled"):
        label.build_gold([label.make_sheet(JDS, ["a.pdf"], fake_fetch, 2)])


def test_build_gold_single_annotator():
    sheet = fill(label.make_sheet(JDS, ["a.pdf"], fake_fetch, 3), {"a.pdf#0", "a.pdf#2"}, supported={"python"})
    gold = label.build_gold([sheet], min_queries=2)
    q = gold["queries"][0]
    assert q["id"] == "j1::a.pdf" and q["relevant_chunk_ids"] == ["a.pdf#0", "a.pdf#2"]
    assert "python" in q["supported_skills"] and "sql" in q["unsupported_skills"]
    assert gold["meta"]["agreement"] is None and gold["meta"]["warnings"] == []


def test_build_gold_warns_on_few_queries_and_drops_items_without_relevant_chunks():
    sheet = fill(label.make_sheet(JDS, ["a.pdf"], fake_fetch, 3), {"a.pdf#1"})
    sheet["items"][1]["chunks"][1]["relevant"] = False  # j2 item: nothing relevant
    gold = label.build_gold([sheet])
    assert len(gold["queries"]) == 1 and gold["meta"]["dropped_no_relevant"] == ["j2::a.pdf"]
    assert any("only 1 labelled queries" in w for w in gold["meta"]["warnings"])
    for it in sheet["items"]:
        for c in it["chunks"]:
            c["relevant"] = False
    with pytest.raises(label.LabelError, match="nothing to evaluate"):
        label.build_gold([sheet])


def _two_annotators():
    s1 = fill(label.make_sheet(JDS, ["a.pdf"], fake_fetch, 4), {"a.pdf#0", "a.pdf#1"}, supported={"python"})
    s2 = fill(label.make_sheet(JDS, ["a.pdf"], fake_fetch, 4), {"a.pdf#0"}, supported={"python"})  # disagrees on #1
    return s1, s2


def test_inter_annotator_agreement_and_adjudication_modes():
    s1, s2 = _two_annotators()
    ag = label.agreement(s1, s2)
    assert ag["shared_items"] == 2 and ag["chunk_relevance"]["n"] == 8
    assert ag["chunk_relevance"]["agreement"] == 0.75 and len(ag["disagreements"]) == 2
    assert 0 < ag["chunk_relevance"]["kappa"] < 1
    with pytest.raises(label.LabelError, match="disagreements"):
        label.build_gold([s1, s2], min_queries=1)
    first = label.build_gold([s1, s2], adjudicate="first", min_queries=1)
    inter = label.build_gold([s1, s2], adjudicate="intersection", min_queries=1)
    assert first["queries"][0]["relevant_chunk_ids"] == ["a.pdf#0", "a.pdf#1"]
    assert inter["queries"][0]["relevant_chunk_ids"] == ["a.pdf#0"]
    assert first["meta"]["agreement"]["chunk_relevance"]["kappa"] == ag["chunk_relevance"]["kappa"]


def test_validate_sheet_catches_duplicates_and_bad_version():
    sheet = fill(label.make_sheet(JDS[:1], ["a.pdf"], fake_fetch, 2), set())
    sheet["items"].append(json.loads(json.dumps(sheet["items"][0])))
    sheet["version"] = 99
    errors, _ = label.validate_sheet(sheet)
    assert any("duplicate item" in e for e in errors) and any("version" in e for e in errors)


def test_label_cli_round_trip(tmp_path, capsys):
    sheet = fill(label.make_sheet(JDS, ["a.pdf"], fake_fetch, 3), {"a.pdf#0"})
    sp, gp = tmp_path / "sheet.json", tmp_path / "gold.json"
    label.save(sheet, str(sp))
    assert main(["label", "build-gold", "--sheets", str(sp), "--out", str(gp), "--min-queries", "1"]) == 0
    gold = json.loads(gp.read_text(encoding="utf-8"))
    assert {q["id"] for q in gold["queries"]} == {"j1::a.pdf", "j2::a.pdf"}
    # the gold file feeds straight into the retrieval comparison
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({q["id"]: q["relevant_chunk_ids"] for q in gold["queries"]}), encoding="utf-8")
    assert main(["retrieval", "--gold", str(gp), "--runs-a", str(runs), "--runs-b", str(runs),
                 "--report-dir", str(tmp_path / "rep")]) == 0
    unlabelled = tmp_path / "raw.json"
    label.save(label.make_sheet(JDS, ["a.pdf"], fake_fetch, 2), str(unlabelled))
    assert main(["label", "build-gold", "--sheets", str(unlabelled), "--out", str(tmp_path / "x.json")]) == 2
    assert "not fully labelled" in capsys.readouterr().out


# ----------------------------------------------------------------- perf
def test_perf_percentiles_tokens_and_cost():
    runs = [{"node_timings": [{"node": "Copywriter", "ms": ms}, {"node": "Critic", "ms": 10}],
             "usage": [{"role": "writer", "model": "gpt-x", "input_tokens": 1000, "output_tokens": 200,
                        "cost_usd": 0.01, "ms": ms}], "cost_usd": 0.01 * (i + 1)}
            for i, ms in enumerate([100, 200, 300, 400, 1000])]
    s = perf.summarize(runs)
    w = s["nodes"]["Copywriter"]
    assert w["n"] == 5 and w["p50"] == 300 and w["p95"] == pytest.approx(880) and w["max"] == 1000
    m = s["models"]["writer / gpt-x"]
    assert m["calls"] == 5 and m["input_tokens"] == 5000 and m["output_tokens"] == 1000
    assert m["cost_usd"] == pytest.approx(0.05) and m["latency_ms"]["p50"] == 300
    assert s["cost_usd_per_run"]["total"] == pytest.approx(0.15) and s["cost_usd_per_run"]["p50"] == pytest.approx(0.03)
    assert s["total_ms"]["n"] == 5 and s["total_ms"]["p50"] == 310
    assert "Copywriter" in perf.format_table(s)


def test_perf_handles_missing_fields_and_empty_input():
    s = perf.summarize([{"run_id": "a"}, {"timings": [{"node": "n", "ms": 5}]}])
    assert s["runs"] == 2 and s["nodes"]["n"]["n"] == 1 and s["cost_usd_per_run"]["n"] == 0
    assert perf.summarize([])["runs"] == 0


def test_perf_cli_reads_jsonl(tmp_path, capsys):
    f = tmp_path / "runs.jsonl"
    f.write_text(json.dumps({"node_timings": [{"node": "Editor", "ms": 50}]}) + "\n\n", encoding="utf-8")
    assert main(["perf", "--file", str(f)]) == 0
    assert "Editor" in capsys.readouterr().out


# ----------------------------------------------------------------- run tracking
def test_report_is_self_describing_and_never_contains_keys():
    from runtime import Binding, Profile
    prof = Profile()
    prof.roles["writer"] = [Binding("anthropic", "claude-sonnet-5-5", key_alias="my-secret-alias")]
    rep = build_report("retrieval", {"x": 1}, {"m": {"value": 1.0, "direction": "higher"}}, seeds={"s": 3},
                       profile=prof, index_stats=lambda: {"vectors": 123}, cost_usd=0.25)
    assert rep["models"]["writer"] == ["anthropic:claude-sonnet-5-5"]
    assert rep["index_stats"] == {"vectors": 123} and rep["seeds"] == {"s": 3} and rep["cost_usd_total"] == 0.25
    assert rep["git_sha"] and rep["date"].endswith("+00:00") and "python" in rep["versions"]
    assert "my-secret-alias" not in json.dumps(rep)
    assert model_ids(None) == {}


def test_report_survives_failing_index_stats():
    def boom():
        raise RuntimeError("pinecone down sk-SECRET")
    rep = build_report("x", {}, {}, index_stats=boom)
    assert rep["index_stats"] == {"error": "RuntimeError"} and "sk-SECRET" not in json.dumps(rep)


def test_bm25_leakage_warning(tmp_path):
    fitted = tmp_path / "fitted_bm25.json"
    assert bm25_warning("fitted", ["a.pdf"], str(fitted)) is None  # nothing fitted -> nothing leaked
    fitted.write_text("{}")
    assert "optimistic" in bm25_warning("fitted", ["a.pdf"], str(fitted))
    assert bm25_warning("cold", ["a.pdf"], str(fitted)) is None
    assert bm25_warning("fitted", [], str(fitted)) is None


def test_save_report_names_include_kind_and_sha(tmp_path):
    path = save_report(build_report("chaos", {}, {}), str(tmp_path))
    assert path.exists() and path.name.startswith("report_chaos_") and json.loads(path.read_text())["kind"] == "chaos"


def test_cold_bm25_flag_selects_unfitted_encoder(monkeypatch):
    import database
    calls = []
    monkeypatch.setattr(database, "set_bm25_mode", lambda m: calls.append(m), raising=False)
    assert _apply_bm25_mode(False) == "fitted" and calls == []
    assert _apply_bm25_mode(True) == "cold" and calls == ["cold"]


# ----------------------------------------------------------------- regression gate
def metric(v, d="higher", **kw):
    return {"value": v, "direction": d, **kw}


BASE = {"metrics": {"retrieval.mrr": metric(0.80), "faith.halluc": metric(0.05, "lower"),
                    "chaos.secret_leaks": metric(0, "lower", tolerance=0), "retrieval.ndcg@5": metric(0.70)}}


def test_check_passes_within_tolerance_and_reports_improvements():
    cur = {"metrics": {"retrieval.mrr": metric(0.79), "faith.halluc": metric(0.04, "lower"),
                       "chaos.secret_leaks": metric(0, "lower", tolerance=0), "retrieval.ndcg@5": metric(0.80)}}
    res = regress.check(BASE, cur, 0.02)
    assert res["ok"] and not res["regressions"]
    assert [r["metric"] for r in res["improvements"]] == ["retrieval.ndcg@5"] and res["unchanged"] == 3


def test_check_flags_drop_in_higher_better_and_rise_in_lower_better():
    cur = {"metrics": {"retrieval.mrr": metric(0.70), "faith.halluc": metric(0.20, "lower"),
                       "chaos.secret_leaks": metric(0, "lower", tolerance=0), "retrieval.ndcg@5": metric(0.70)}}
    res = regress.check(BASE, cur, 0.02)
    assert not res["ok"] and {r["metric"] for r in res["regressions"]} == {"retrieval.mrr", "faith.halluc"}


def test_secret_leak_has_zero_tolerance_despite_global_tolerance():
    cur = {"metrics": dict(BASE["metrics"], **{"chaos.secret_leaks": metric(1, "lower", tolerance=0)})}
    res = regress.check(BASE, cur, tolerance=0.5)
    assert [r["metric"] for r in res["regressions"]] == ["chaos.secret_leaks"]


def test_vanished_metric_is_a_regression_unless_allowed():
    cur = {"metrics": {k: v for k, v in BASE["metrics"].items() if k != "retrieval.mrr"}}
    assert not regress.check(BASE, cur)["ok"] and regress.check(BASE, cur)["missing"] == ["retrieval.mrr"]
    assert regress.check(BASE, cur, allow_missing=True)["ok"]


def test_new_metrics_are_reported_not_gated():
    cur = {"metrics": dict(BASE["metrics"], **{"brand.new": metric(0.1)})}
    res = regress.check(BASE, cur)
    assert res["ok"] and res["new_metrics"] == ["brand.new"]


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_regress_exit_codes(tmp_path, capsys):
    base = write(tmp_path, "base.json", BASE)
    good = write(tmp_path, "good.json", BASE)
    bad = write(tmp_path, "bad.json", {"metrics": dict(BASE["metrics"], **{"retrieval.mrr": metric(0.5)})})
    assert regress.run(base, good) == regress.EXIT_OK
    assert regress.run(base, bad) == regress.EXIT_REGRESSION
    assert "REGRESSION  retrieval.mrr" in capsys.readouterr().out
    assert regress.run(str(tmp_path / "nope.json"), good) == regress.EXIT_USAGE
    assert regress.run(base, str(tmp_path / "nope.json")) == regress.EXIT_USAGE
    (tmp_path / "corrupt.json").write_text("{not json")
    assert regress.run(str(tmp_path / "corrupt.json"), good) == regress.EXIT_USAGE


def test_regress_cli_and_update_baseline(tmp_path):
    cur = write(tmp_path, "cur.json", BASE)
    baseline = str(tmp_path / "evals" / "baseline.json")
    assert main(["regress", "--baseline", baseline, "--current", cur]) == 2          # no baseline yet
    assert main(["regress", "--baseline", baseline, "--current", cur, "--update-baseline"]) == 0
    assert json.loads(open(baseline, encoding="utf-8").read()) == BASE
    worse = write(tmp_path, "worse.json", {"metrics": dict(BASE["metrics"], **{"faith.halluc": metric(0.5, "lower")})})
    assert main(["regress", "--baseline", baseline, "--current", worse, "--tolerance", "0.02"]) == 1
    assert main(["regress", "--baseline", baseline, "--current", cur]) == 0


# ----------------------------------------------------------------- CLI plumbing
def test_legacy_flag_style_is_migrated():
    assert _migrate_legacy(["--letters", "l.json"]) == ["letters", "--letters", "l.json"]
    assert _migrate_legacy(["--gold", "g", "--runs-a", "a", "--runs-b", "b"])[0] == "retrieval"
    assert _migrate_legacy(["--collect", "hybrid", "--gold", "g"])[0] == "retrieval"
    assert _migrate_legacy(["chaos"]) == ["chaos"] and _migrate_legacy([]) == []
    assert _migrate_legacy(["--help"]) == ["--help"]


def test_legacy_retrieval_invocation_still_produces_the_comparison(tmp_path, capsys):
    gold = write(tmp_path, "gold.json", {"queries": [{"id": "q1", "jd": "j", "namespace": "n", "relevant_chunk_ids": ["c1"]}]})
    a = write(tmp_path, "a.json", {"q1": ["x"]})
    b = write(tmp_path, "b.json", {"q1": ["c1"]})
    assert main(["--gold", gold, "--runs-a", a, "--runs-b", b, "--report-dir", str(tmp_path / "r")]) == 0
    out = capsys.readouterr()
    assert "p_holm" in out.out and "only 1 queries" in out.out and "deprecated" in out.err
    report = json.loads(next((tmp_path / "r").glob("report_retrieval_*.json")).read_text(encoding="utf-8"))
    assert report["metrics"]["retrieval.mrr"]["value"] == 1.0 and report["warnings"]


def test_letters_cli_scores_claims(tmp_path, capsys):
    letters = write(tmp_path, "letters.json", [
        {"letter": "I built python services handling 5000 requests per second. I deployed Kubernetes clusters.",
         "resume_text": "Built python services handling 5000 requests per second.", "skills": ["python", "kubernetes"]}])
    assert main(["letters", "--letters", letters, "--report-dir", str(tmp_path / "r")]) == 0
    out = capsys.readouterr().out
    assert '"mean_hallucination_rate": 0.5' in out and "WARNING" in out  # lexical-entailer caveat
    report = json.loads(next((tmp_path / "r").glob("report_letters_*.json")).read_text(encoding="utf-8"))
    assert report["metrics"]["faithfulness.mean_hallucination_rate"] == {"value": 0.5, "direction": "lower"}


def test_unknown_command_exits_via_argparse():
    with pytest.raises(SystemExit):
        main(["definitely-not-a-command"])
