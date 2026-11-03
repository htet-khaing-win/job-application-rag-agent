"""Routing gates of the current self-healing graph."""
import asyncio

from langgraph.graph import END

from graph import (route_after_ingest, route_after_join, route_after_refine,
                   route_after_retrieval, route_after_write, should_refine_letter)
from state import GraphState


def S(**kw):
    return GraphState(job_description="JD", **kw)


def test_invalid_jd_routes_to_fallback():
    assert route_after_ingest(S(is_valid_jd=False)) == "fallback_handler"


def test_valid_jd_fans_out_to_retrieval_and_research():
    assert set(route_after_ingest(S(is_valid_jd=True))) == {"retrieve_resumes", "research_company"}


def test_low_or_missing_score_routes_to_fallback():
    assert asyncio.run(route_after_retrieval(S())) == "fallback_handler"
    assert asyncio.run(route_after_retrieval(S(vector_relevance_score=2.0))) == "fallback_handler"
    assert asyncio.run(route_after_retrieval(S(vector_relevance_score=80.0))) == "generate_summary"


def test_join_and_write_failures_route_to_fallback():
    assert route_after_join(S(error_type="incomplete_context")) == "fallback_handler"
    assert route_after_join(S()) == "write_cover_letter"
    assert route_after_write(S(error_type="generation_failed", cover_letter="x")) == "fallback_handler"
    assert route_after_write(S(cover_letter="")) == "fallback_handler"
    assert route_after_write(S(cover_letter="letter")) == "verify_letter"


def test_refinement_gate_is_bounded():
    assert asyncio.run(should_refine_letter(S(needs_refinement=False))) == END
    assert asyncio.run(should_refine_letter(S(needs_refinement=True, refinement_count=1,
                                              refinement_edits=[{"type": "replace"}]))) == END
    assert asyncio.run(should_refine_letter(S(needs_refinement=True, refinement_count=0,
                                              refinement_edits=[{"type": "replace"}]))) == "refine_letter"
    assert route_after_refine(S(refinement_count=1)) == END
    assert route_after_refine(S(refinement_count=0)) == "critique_letter"
