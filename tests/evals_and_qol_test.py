import asyncio

import pytest

import runtime
from evals.faithfulness import aggregate, score_letter
from evals.metrics import (bootstrap_ci, compare, evaluate_run, mrr, ndcg_at_k,
                           paired_permutation_pvalue, precision_at_k, recall_at_k)
from runtime import PRESETS, Profile, Runtime, check_connection, preset_profile
from vault import KeyVault


# --- metrics -------------------------------------------------------------------
def test_ranking_metrics_known_values():
    ranked, rel = ["a", "b", "c", "d"], ["b", "d"]
    assert recall_at_k(ranked, rel, 2) == 0.5
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert mrr(ranked, rel) == 0.5
    assert ndcg_at_k(["b", "d"], rel, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], rel, 2) == 0.0
    assert recall_at_k([], [], 3) == 0.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]
    m, lo, hi = bootstrap_ci(vals)
    assert lo <= m <= hi and bootstrap_ci(vals) == (m, lo, hi)


def test_permutation_test_detects_real_difference_not_noise():
    a = [0.1] * 20
    assert paired_permutation_pvalue(a, [0.9] * 20) < 0.01
    assert paired_permutation_pvalue(a, a) == 1.0
    with pytest.raises(ValueError):
        paired_permutation_pvalue([1], [1, 2])


def test_compare_reports_delta_and_pvalue():
    gold = {"q1": ["x"], "q2": ["y"], "q3": ["z"]}
    base = evaluate_run({"q1": ["a"], "q2": ["a"], "q3": ["a"]}, gold, k=3)
    good = evaluate_run({"q1": ["x"], "q2": ["y"], "q3": ["z"]}, gold, k=3)
    row = next(r for r in compare(base, good) if r["metric"] == "mrr")
    assert row["delta"] == 1.0 and 0 < row["p_value"] <= 1


# --- faithfulness --------------------------------------------------------------
def test_faithfulness_flags_unsupported_claims_but_not_hedged_ones():
    resume = "Built python services with postgres."
    bad = score_letter("I led Kubernetes migrations. I built python services.", resume, ["python", "kubernetes"])
    ok = score_letter("I am prepared to learn Kubernetes. I built python services.", resume, ["python", "kubernetes"])
    assert bad["hallucination_rate"] > 0 and bad["unsupported_claims"] == ["kubernetes"]
    assert ok["hallucination_rate"] == 0
    assert aggregate([bad, ok])["letters_with_any_hallucination"] == 1


# --- presets -------------------------------------------------------------------
@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_is_valid(name):
    prof = preset_profile(name)
    prof.validate()
    assert prof.roles["writer"]


def test_local_preset_needs_no_chat_keys(tmp_path):
    rt = Runtime(KeyVault(tmp_path, use_keyring=False), preset_profile("Fully local (Ollama)"))
    assert not any(m.startswith(("JD", "Copy", "Hiring", "Editor", "Fact")) for m in rt.missing_keys())


# --- connection test -------------------------------------------------------------
def test_check_connection_reports_missing_key_and_redacts_errors(tmp_path, monkeypatch):
    vault = KeyVault(tmp_path, use_keyring=False)
    rt = Runtime(vault, Profile())
    assert check_connection(rt, "writer")["ok"] is False

    vault.set("g", "gemini", "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1")

    class Dead:
        def invoke(self, *_):
            raise RuntimeError("401 key AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1 rejected")

    monkeypatch.setattr(runtime, "build_chat_model", lambda *a, **k: Dead())
    res = check_connection(rt, "writer")
    assert res["ok"] is False and "AIzaSyFAKE" not in res["detail"]


def test_check_connection_success(tmp_path, monkeypatch):
    vault = KeyVault(tmp_path, use_keyring=False)
    vault.set("g", "gemini", "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1")
    monkeypatch.setattr(runtime, "build_chat_model",
                        lambda *a, **k: type("OK", (), {"invoke": lambda s, p: "OK"})())
    res = check_connection(Runtime(vault, Profile()), "writer")
    assert res["ok"] and res["detail"] == "Connected"


# --- per-node timings --------------------------------------------------------------
def test_pipeline_records_node_timings(monkeypatch):
    import node
    import resilience
    import subagents
    from graph import build_graph
    from resilience import reset_breakers, start_heal_log
    from self_healing_pipeline_test import JD, Scripted

    reset_breakers()
    monkeypatch.setattr(resilience, "BACKOFF_BASE_S", 0)
    monkeypatch.setattr(node, "extract_jd_keywords", lambda jd, top_k=20: ["python"])

    async def retrieve(state):
        return {"retrieved_chunks": [{"text": "python", "source": "r", "score": 1, "chunk_id": "1"}],
                "vector_relevance_score": 90.0}

    async def research(state):
        return {"company_research": "x", "company_research_success": True}

    monkeypatch.setattr(subagents, "retrieve_resumes_node", retrieve)
    monkeypatch.setattr(subagents, "research_company_node", research)
    m = Scripted()

    async def go():
        start_heal_log()
        return await build_graph(m, m).ainvoke({"job_description": JD, "company_name": "Acme"})

    out = asyncio.run(go())
    names = {t["node"] for t in out["node_timings"]}
    assert {"JD Analyst", "Copywriter", "Fact Checker"} <= names
    assert all(t["ms"] >= 0 for t in out["node_timings"])
