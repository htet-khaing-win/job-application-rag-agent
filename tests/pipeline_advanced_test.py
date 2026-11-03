import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from self_healing_pipeline_test import JD, Scripted

import database
import node
import resilience
import runlog
import subagents
from cost import estimate_run_cost
from database import retrieve_resumes_node, select_fair_matches, set_bm25_mode
from graph import build_graph, validate_ablate
from pipeline import run_pipeline
from resilience import reset_breakers
from runtime import Profile, Runtime
from state import GraphState
from vault import KeyVault

LETTER_UNGROUNDED = "I have deep Kubernetes expertise. I built Python services."


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBAGENT_HOME", str(tmp_path))
    monkeypatch.setattr(resilience, "BACKOFF_BASE_S", 0)
    reset_breakers()
    monkeypatch.setattr(node, "extract_jd_keywords", lambda jd, top_k=20: ["python", "kubernetes"])
    calls = {"research": 0, "retrieve": 0}

    async def retrieve(state):
        calls["retrieve"] += 1
        return {"retrieved_chunks": [{"text": "Built python services", "source": "r.pdf", "score": 0.9, "chunk_id": "1"}],
                "vector_relevance_score": 90.0}

    async def research(state):
        calls["research"] += 1
        return {"company_research": "Acme builds things", "company_research_success": True}

    monkeypatch.setattr(subagents, "retrieve_resumes_node", retrieve)
    monkeypatch.setattr(subagents, "research_company_node", research)
    return calls


@pytest.fixture
def rt(tmp_path):
    return Runtime(KeyVault(tmp_path / "vault", use_keyring=False), Profile())


class Counting(Scripted):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.critic_calls = 0

    def invoke(self, prompt, *a, **k):
        if "Hiring Manager" in prompt:
            self.critic_calls += 1
        return super().invoke(prompt, *a, **k)


def go(rt, gen, crit, **kw):
    return asyncio.run(run_pipeline(rt, JD, "Acme", gen, crit, **kw))


# ------------------------------------------------------------------ retrieval --
def _match(i, score):
    return {"id": f"r#{i}", "score": score, "metadata": {"text": f"chunk {i}", "source": "r"}}


def _retrieval_patches(namespaces, per_ns, settings=None):
    index = MagicMock()
    index.query.side_effect = lambda **kw: {"matches": per_ns.get(kw["namespace"], [])}
    rt_obj = SimpleNamespace(settings=settings) if settings is not None else None
    return index, rt_obj


def _run_retrieval(namespaces, per_ns, settings, jd):
    index, rt_obj = _retrieval_patches(namespaces, per_ns, settings)
    emb = MagicMock()
    emb.embed_query.return_value = [1.0] * 8
    bm25 = MagicMock()
    bm25.encode_queries.return_value = {"indices": [1], "values": [1.0]}

    def fake_runtime():
        if rt_obj is None:
            raise RuntimeError("no runtime")
        return rt_obj

    with patch("database.list_stored_resumes", return_value=namespaces), \
            patch("database.get_embeddings", return_value=emb), patch("database.get_bm25", return_value=bm25), \
            patch("database.get_pinecone_index", return_value=index), patch("database.get_runtime", fake_runtime):
        out = asyncio.run(retrieve_resumes_node(GraphState(job_description=jd, cleaned_jd=jd, is_valid_jd=True)))
    return out, index


def test_retrieval_queries_every_resume_not_just_two():
    ns = [f"r{i}.pdf" for i in range(5)]
    out, index = _run_retrieval(ns, {n: [_match(i, 0.5 + i / 100)] for i, n in enumerate(ns)}, None, "jd all five")
    assert sorted(c.kwargs["namespace"] for c in index.query.call_args_list) == sorted(ns)
    assert all("alpha" not in c.kwargs for c in index.query.call_args_list)   # Pinecone has no alpha arg
    assert {tuple(c.kwargs["vector"]) for c in index.query.call_args_list} == {(0.4,) * 8}   # default alpha
    assert {tuple(c.kwargs["sparse_vector"]["values"]) for c in index.query.call_args_list} == {(0.6,)}
    assert {c.kwargs["top_k"] for c in index.query.call_args_list} == {10}
    assert out["vector_relevance_score"] == pytest.approx(54.0)


def test_settings_alpha_topk_and_max_resumes_are_plumbed():
    ns = [f"r{i}.pdf" for i in range(5)]
    cfg = {"alpha": 0.7, "top_k": 4, "max_resumes": 3}
    out, index = _run_retrieval(ns, {n: [_match(i, 0.5)] for i, n in enumerate(ns)}, cfg, "jd settings")
    calls = index.query.call_args_list
    assert len(calls) == 3
    assert {tuple(c.kwargs["vector"]) for c in calls} == {(0.7,) * 8}
    assert {round(c.kwargs["sparse_vector"]["values"][0], 6) for c in calls} == {0.3}
    assert {c.kwargs["top_k"] for c in calls} == {4}
    assert len(out["retrieved_chunks"]) <= 4


def test_one_failing_namespace_does_not_sink_retrieval():
    index = MagicMock()

    def q(**kw):
        if kw["namespace"] == "bad.pdf":
            raise RuntimeError("503 sk-abcdefghijklmnopqrstuvwx")
        return {"matches": [_match(1, 0.8)]}

    index.query.side_effect = q
    emb = MagicMock()
    emb.embed_query.return_value = [0.0]
    with patch("database.list_stored_resumes", return_value=["bad.pdf", "ok.pdf"]), \
            patch("database.get_embeddings", return_value=emb), patch("database.get_bm25", return_value=MagicMock()), \
            patch("database.get_pinecone_index", return_value=index):
        out = asyncio.run(retrieve_resumes_node(GraphState(job_description="x", cleaned_jd="jd failing ns")))
    assert out["vector_relevance_score"] == 80.0


def test_fair_merge_keeps_best_chunk_of_each_resume_and_sorts():
    by_ns = {"A": [_match(i, 0.9 - i * 0.01) for i in range(6)], "B": [_match(100, 0.3)], "C": [_match(200, 0.2)]}
    picked = select_fair_matches(by_ns, top_k=4)
    ids = [m["id"] for m in picked]
    assert "r#100" in ids and "r#200" not in ids        # top_k//2 = 2 reserved slots: A's best + B's best
    assert [m["score"] for m in picked] == sorted((m["score"] for m in picked), reverse=True)
    assert len(picked) == 4 and select_fair_matches({}, 5) == []


def test_bm25_mode_switch_forces_cold_encoder():
    with pytest.raises(ValueError):
        set_bm25_mode("warm")
    database._bm25 = object()
    set_bm25_mode("cold")
    assert database._bm25 is None and database._bm25_mode == "cold"
    set_bm25_mode("fitted")
    assert database._bm25_mode == "fitted"


# ------------------------------------------------------------ reducer / refine loop --
def test_refine_loop_terminates_and_flag_resets():
    assert "needs_refinement" in GraphState.model_fields
    assert not GraphState.model_fields["needs_refinement"].metadata           # no reducer any more
    critique = json.dumps({"edits": [{"type": "replace", "target": "Python services",
                                      "replacement": "Python microservices", "reason": "specific"}],
                           "ready_to_submit": False})
    m = Counting(critique=critique, letter="I built Python services at scale.")
    out = asyncio.run(run_pipeline(SimpleNamespace(missing_keys=lambda: [], profile=None, vault=None,
                                                   failover_enabled=True), JD, "Acme", m, m))
    assert "microservices" in out["cover_letter"]
    assert out["refinement_count"] == 1 and out["needs_refinement"] is False
    assert m.critic_calls == 1                       # refine -> END, no second critique


# ---------------------------------------------------------------------- ablations --
def test_unknown_ablation_rejected_everywhere(rt):
    with pytest.raises(ValueError, match="Unknown ablation"):
        validate_ablate({"fact_checker", "nope"})
    with pytest.raises(ValueError):
        build_graph(Scripted(), Scripted(), ablate={"bogus"})
    with pytest.raises(ValueError):
        go(rt, Scripted(), Scripted(), ablate={"bogus"})


def test_ablate_fact_checker_ships_ungrounded_claim_unverified(rt):
    base = go(rt, Scripted(letter=LETTER_UNGROUNDED), Scripted(letter=LETTER_UNGROUNDED))
    assert "Kubernetes expertise" not in base["cover_letter"] and base["letter_verified"] is True
    abl = go(rt, Scripted(letter=LETTER_UNGROUNDED), Scripted(letter=LETTER_UNGROUNDED), ablate={"fact_checker"})
    assert "Kubernetes expertise" in abl["cover_letter"] and abl["letter_verified"] is False
    assert any("ablation" in e for e in abl["healing_log"])


def test_ablate_critique_refine_skips_those_nodes(rt):
    critique = json.dumps({"edits": [{"type": "replace", "target": "Python services",
                                      "replacement": "Python microservices"}], "ready_to_submit": False})
    full = Counting(critique=critique)
    out_full = go(rt, full, full)
    assert full.critic_calls == 1 and "microservices" in out_full["cover_letter"]

    skipped = Counting(critique=critique)
    out = go(rt, skipped, skipped, ablate={"critique_refine"})
    names = {t["node"] for t in out["node_timings"]}
    assert skipped.critic_calls == 0 and "microservices" not in out["cover_letter"]
    assert "Hiring Manager" not in names and "Editor" not in names and "Fact Checker" in names


def test_ablate_research_never_calls_the_search_node(rt, env):
    out = go(rt, Scripted(), Scripted())
    assert env["research"] == 1 and "Acme builds things" in out["company_research"]
    env["research"] = 0
    out = go(rt, Scripted(), Scripted(), ablate={"research"})
    assert env["research"] == 0 and "Unable to fetch" in out["company_research"]
    assert out["cover_letter"]        # pipeline still delivers without research


def test_ablate_failover_disables_then_restores_flag(rt, monkeypatch):
    seen = []

    async def research(state):
        from runtime import get_runtime
        seen.append(get_runtime().failover_enabled)
        return {"company_research": "x", "company_research_success": True}

    monkeypatch.setattr(subagents, "research_company_node", research)
    go(rt, Scripted(), Scripted(), ablate={"failover"})
    go(rt, Scripted(), Scripted())
    assert seen == [False, True] and rt.failover_enabled is True


# ------------------------------------------------------------------------ timeout --
def test_timeout_cancels_graph_and_logs(rt, monkeypatch):
    async def slow(state):
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(subagents, "research_company_node", slow)
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="cancelled"):
        go(rt, Scripted(), Scripted(), timeout_s=0.3)
    assert time.perf_counter() - t0 < 3
    assert any(r.get("error_type") == "timeout" and r["ok"] is False for r in runlog.read_runs())


# ----------------------------------------------------------- streaming equivalence --
def test_event_streaming_returns_same_state_as_ainvoke(rt):
    events = []
    plain = go(rt, Scripted(), Scripted())
    streamed = go(rt, Scripted(), Scripted(), on_event=events.append)
    for key in ("cover_letter", "verified_skills", "unverified_skills", "refinement_count", "is_valid_jd",
                "letter_verified", "company_research", "vector_relevance_score", "is_fallback"):
        assert plain.get(key) == streamed.get(key), key
    assert sorted(t["node"] for t in plain["node_timings"]) == sorted(t["node"] for t in streamed["node_timings"])
    assert {e["node"] for e in events} >= {"JD Analyst", "Copywriter", "Fact Checker"}
    assert all(e["ms"] is None or e["ms"] >= 0 for e in events)


def test_broken_event_callback_does_not_fail_run(rt):
    def boom(ev):
        raise RuntimeError("ui exploded")

    assert go(rt, Scripted(), Scripted(), on_event=boom)["cover_letter"]


# ------------------------------------------------------------------- token sink --
def test_token_sink_registered_for_writer_and_always_cleared(rt, monkeypatch):
    during = {}

    class Streamy(Scripted):
        def invoke(self, prompt, *a, **k):
            if "Write a factual cover letter" in prompt:
                during["sinks"] = dict(resilience._sinks.get() or {})
            return super().invoke(prompt, *a, **k)

    async def scenario(fail):
        sink = lambda t: None  # noqa: E731
        try:
            await run_pipeline(rt, JD, "Acme", Streamy(), Streamy(), token_sink=sink,
                               timeout_s=0.3 if fail else 30)
        except TimeoutError:
            pass
        return dict(resilience._sinks.get() or {})

    assert "writer" not in asyncio.run(scenario(False))
    assert "writer" in during["sinks"]                       # visible to the writer while running

    async def slow(state):
        await asyncio.sleep(5)
        return {}

    monkeypatch.setattr(subagents, "research_company_node", slow)
    assert "writer" not in asyncio.run(scenario(True))      # cleared on the failure path too


# ------------------------------------------------------- usage / cost / run log --
def test_usage_attributed_to_nodes_summed_and_sanitised(rt):
    class Billing(Scripted):
        def invoke(self, prompt, *a, **k):
            if "Write a factual cover letter" in prompt:
                resilience.record_usage({"role": "writer", "model": "gpt-x", "tokens_in": 100, "tokens_out": 50,
                                         "cost_usd": 0.002, "prompt": "TOP SECRET PROMPT", "api_key": "sk-leak"})
            return super().invoke(prompt, *a, **k)

    out = go(rt, Billing(), Billing())
    assert out["cost_usd"] == pytest.approx(0.002)
    entry = next(u for u in out["usage"] if u["model"] == "gpt-x")
    assert entry["node"] == "Copywriter" and "prompt" not in entry and "api_key" not in entry
    row = runlog.read_runs()[-1]
    assert row["tokens_in"] == 100 and row["tokens_out"] == 50 and row["run_id"] == out["run_id"]


def test_run_log_never_contains_letter_jd_or_company(rt, home=None):
    marker_jd = JD + " ZEBRA-MARKER-JD"
    m = Scripted(letter="Dear team, QUOKKA-LETTER-MARKER built Python services.")
    asyncio.run(run_pipeline(rt, marker_jd, "Wombat Industries", m, m))
    raw = runlog.log_path().read_text(encoding="utf-8")
    for leaked in ("ZEBRA-MARKER-JD", "QUOKKA-LETTER-MARKER", "Wombat"):
        assert leaked not in raw
    row = json.loads(raw.strip().splitlines()[-1])
    assert row["ok"] is True and row["ablate"] == [] and row["node_timings"] and "models" in row


def test_run_ids_unique_and_result_carries_metadata(rt):
    a, b = go(rt, Scripted(), Scripted()), go(rt, Scripted(), Scripted())
    assert a["run_id"] != b["run_id"] and len(a["run_id"]) == 12
    assert a["duration_ms"] > 0 and "usage" in a


# ----------------------------------------------------------------------- budget --
def test_budget_cap_blocks_run_before_any_model_call(tmp_path, monkeypatch):
    vault = KeyVault(tmp_path / "v", use_keyring=False)
    profile = Profile()
    est = estimate_run_cost(profile, len(JD))["total_usd"]
    assert est > 0
    for prov in ("openai", "gemini"):
        vault.set(f"{prov}-key", prov, f"{prov}-secret-key-1234567890", monthly_cap_usd=est / 100)
    vault.set("tav", "tavily", "tvly-secret-key-1234567890")
    vault.set("pin", "pinecone", "pcsk_secret-key-1234567890")
    runtime = Runtime(vault, profile)
    called = []
    monkeypatch.setattr(subagents, "ingest_jd_node", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="budget"):
        asyncio.run(run_pipeline(runtime, JD, "Acme"))
    assert not called


def test_hybrid_scale_endpoints_and_validation():
    from database import hybrid_scale
    d, sp = hybrid_scale([1.0, 2.0], {"indices": [3], "values": [4.0]}, 1.0)
    assert d == [1.0, 2.0] and sp["values"] == [0.0]
    d, sp = hybrid_scale([1.0, 2.0], {"indices": [3], "values": [4.0]}, 0.0)
    assert d == [0.0, 0.0] and sp["values"] == [4.0] and sp["indices"] == [3]
    with pytest.raises(ValueError):
        hybrid_scale([1.0], {"indices": [], "values": []}, 1.5)
