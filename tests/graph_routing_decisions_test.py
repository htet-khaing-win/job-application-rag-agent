"""
Module: tests/graph_routing_decisions_test.py

PURPOSE:
This suite validates the 'Traffic Controller' logic of the LangGraph workflow. 
It ensures that the conditional edges correctly route the GraphState based on 
validation flags, quality scores, and iteration limits.

TESTED LOGIC GATES:
1. Retrieval Gate: Routes to Pinecone or Fallback based on JD validity.
2. Quality Gate (Self-Correction): Decides whether to proceed to summary 
   generation, retry a query rewrite, or give up after MAX_REWRITES (2).
3. Refinement Gate (Polishing): Manages the loop between Critique and Refinement 
   nodes, enforcing a hard stop after 3 iterations to prevent infinite loops.

SCENARIOS COVERED:
- Happy Path (Valid JD, High Score, No Refinement)
- Rejection Path (Invalid JD)
- Loop Exhaustion (Maxing out rewrite/refinement counters)

USAGE: pytest tests/graph_routing_decisions_test.py -v
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from state import GraphState
from graph import (
    should_proceed_with_retrieval,
    should_rewrite_query,
    should_refine_letter,
)
from langgraph.graph import END

# should_proceed_with_retrieval

def test_invalid_jd_routes_to_fallback():
    state = GraphState(
        job_description="Not a JD",
        is_valid_jd=False
    )

    decision = should_proceed_with_retrieval(state)

    assert decision == "fallback_handler"


def test_valid_jd_proceeds_to_retrieval():
    state = GraphState(
        job_description="JD",
        is_valid_jd=True
    )

    decision = should_proceed_with_retrieval(state)

    assert decision == "retrieve_resumes"



# should_rewrite_query


def test_no_rewrite_needed_proceeds_to_generate_summary():
    state = GraphState(
        job_description="JD",
        needs_rewrite=False
    )

    decision = should_rewrite_query(state)

    assert decision == "generate_summary"


def test_rewrite_routes_to_rewrite_query_under_limit():
    state = GraphState(
        job_description="JD",
        needs_rewrite=True,
        refinement_count=0
    )

    decision = should_rewrite_query(state)

    assert decision == "rewrite_query"


def test_rewrite_exhaustion_routes_to_fallback():
    state = GraphState(
        job_description="JD",
        needs_rewrite=True,
        refinement_count=2  # MAX_REWRITES reached
    )

    decision = should_rewrite_query(state)

    assert decision == "fallback_handler"



# should_refine_letter

def test_no_refinement_needed_ends_graph():
    state = GraphState(
        job_description="JD",
        needs_refinement=False
    )

    decision = should_refine_letter(state)

    assert decision == END


def test_refinement_routes_to_refine_letter_under_limit():
    state = GraphState(
        job_description="JD",
        needs_refinement=True,
        refinement_count=1
    )

    decision = should_refine_letter(state)

    assert decision == "refine_letter"


def test_refinement_stops_at_iteration_limit():
    state = GraphState(
        job_description="JD",
        needs_refinement=True,
        refinement_count=3
    )

    decision = should_refine_letter(state)

    assert decision == END
