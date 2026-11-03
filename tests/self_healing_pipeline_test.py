import asyncio
import json

import pytest

import node
import resilience
import subagents
from graph import build_graph
from resilience import AllCandidatesFailed, ResilientLLM, invoke_json, reset_breakers, start_heal_log

JD = ("Requirements: Python and Kubernetes experience. Responsibilities: build and operate backend services, "
      "collaborate with product and design, and mentor juniors. Qualifications: BSc in computer science or equivalent "
      "practical experience shipping production software.")


class R:
    def __init__(self, content):
        self.content = content


class Scripted:
    """Fake chat model routed by prompt content."""

    def __init__(self, letter="Dear team, I built Python services at scale.", critique=None, ingest=None):
        self.letter = letter
        self.critique = critique if critique is not None else json.dumps({"edits": [], "ready_to_submit": True})
        self.ingest = ingest if ingest is not None else [json.dumps({"valid": True, "cleaned_jd": JD})]

    def invoke(self, prompt, *a, **k):
        if "not valid JSON" in prompt:
            return R(json.dumps({"valid": True, "cleaned_jd": JD}))
        if "processing a job description" in prompt:
            return R(self.ingest[0])
        if "Write a factual cover letter" in prompt:
            return R(self.letter)
        if "Rewrite ONLY the listed sentences" in prompt:
            return R("I built Python services. I am prepared to learn Kubernetes via my Python background.")
        if "Hiring Manager" in prompt:
            return R(self.critique)
        return R("ok")


class Boom:
    def invoke(self, *a, **k):
        raise RuntimeError("503 upstream overloaded")


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(resilience, "BACKOFF_BASE_S", 0)
    reset_breakers()
    monkeypatch.setattr(node, "extract_jd_keywords", lambda jd, top_k=20: ["python", "kubernetes"])

    async def fake_retrieve(state):
        return {"retrieved_chunks": [{"text": "Built python services", "source": "r.pdf",
                                      "score": 0.9, "chunk_id": "1"}],
                "vector_relevance_score": 90.0}

    async def fake_research(state):
        return {"company_research": "Acme builds things", "company_research_success": True}

    monkeypatch.setattr(subagents, "retrieve_resumes_node", fake_retrieve)
    monkeypatch.setattr(subagents, "research_company_node", fake_research)


def run(gen, crit, jd=JD):
    app = build_graph(gen, crit)

    async def go():
        start_heal_log()
        return await app.ainvoke({"job_description": jd, "company_name": "Acme"})

    return asyncio.run(go())


def test_happy_path():
    m = Scripted()
    out = run(m, m)
    assert not out.get("is_fallback") and "Python" in out["cover_letter"] and out["letter_verified"]


def test_writer_failover_to_second_model():
    good = Scripted()
    gen = ResilientLLM("writer", [("a", lambda: Boom()), ("b", lambda: good)])
    out = run(gen, good)
    assert not out.get("is_fallback")
    assert any("recovered on b" in e for e in out["healing_log"])


def test_ungrounded_claim_is_repaired():
    m = Scripted(letter="I have deep Kubernetes expertise. I built Python services.")
    out = run(m, m)
    assert "prepared to learn Kubernetes" in out["cover_letter"]
    assert any("repair attempt" in e for e in out["healing_log"])


def test_ungrounded_claim_removed_when_repair_fails():
    class Stubborn(Scripted):
        def invoke(self, prompt, *a, **k):
            if "Rewrite ONLY the listed sentences" in prompt:
                return R("I have deep Kubernetes expertise. I built Python services.")
            return super().invoke(prompt, *a, **k)

    m = Stubborn(letter="I have deep Kubernetes expertise. I built Python services.")
    out = run(m, m)
    assert "Kubernetes" not in out["cover_letter"] and "Python" in out["cover_letter"]
    assert any("deterministically" in e for e in out["healing_log"])


def test_critic_outage_degrades_but_still_delivers():
    out = run(Scripted(), Boom())  # pinned critic drives jd_analyst, verifier and critic
    assert not out.get("is_fallback") and out["cover_letter"]
    assert any("heuristic" in e for e in out["healing_log"])


def test_total_writer_outage_routes_to_fallback():
    out = run(Boom(), Scripted())
    assert out.get("is_fallback") and "automatic recovery" in out["final_response"]


def test_malformed_json_is_repaired():
    m = Scripted(ingest=["{not json"])
    out = run(m, m)
    assert out["is_valid_jd"]
    assert any("repaired malformed JSON" in e for e in out["healing_log"])


def test_invalid_jd_goes_to_fallback():
    m = Scripted(ingest=[json.dumps({"valid": False, "cleaned_jd": ""})])
    out = run(m, m, jd="hello world")
    assert out.get("is_fallback")


def test_errors_are_redacted_and_breaker_opens():
    class Leaky:
        def invoke(self, *a, **k):
            raise RuntimeError("bad sk-abcdefghijklmnopqrstuv")

    llm = ResilientLLM("t", [("x", lambda: Leaky())])
    log = start_heal_log()
    for _ in range(2):
        with pytest.raises(AllCandidatesFailed) as ei:
            llm.invoke("hi")
        assert "sk-abcdef" not in str(ei.value)
    assert all("sk-abcdef" not in e for e in log)
    assert resilience._breaker_open("x")


def test_invoke_json_gives_up_cleanly():
    class Bad:
        def invoke(self, p):
            return R("never json")

    with pytest.raises(json.JSONDecodeError):
        invoke_json(Bad(), "give json")
