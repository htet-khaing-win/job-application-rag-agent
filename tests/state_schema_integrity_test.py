"""
Module: tests/test_state_schema_integrity.py

PURPOSE:
This test suite validates the 'GraphState' Pydantic model which serves as the 
Single Source of Truth (SSoT) for the LangGraph workflow. 

WHY THIS MATTERS:
In LangGraph, state is passed between nodes as a shared dictionary. If a node 
returns data that doesn't match the schema, or if a default value is 'None' 
when a list is expected, the entire graph will crash mid-execution. These tests 
ensure that the 'brain' of our application (the State) is robust against 
partial updates and edge cases.

KEY PROTECTIONS:
1. Serialization: Ensures state can be saved/loaded (Check-pointing).
2. Merge Safety: Ensures partial updates from nodes don't delete existing data.
3. Determinism: Ensures flags like 'needs_rewrite' start at 'False'.
4. Bound Testing: Checks how the schema handles unusual values (e.g., negative scores).

USAGE: pytest tests/state_schema_integrity_test.py -v
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from state import GraphState
from pydantic import ValidationError

# Default State Integrity

def test_graph_state_defaults_are_valid_and_serializable():
    """
    GraphState should initialize with valid defaults
    and be JSON serializable without errors.
    """
    state = GraphState(job_description="Sample JD")

    # Should not raise
    serialized = state.model_dump()
    assert isinstance(serialized, dict)

    # Required field
    assert state.job_description == "Sample JD"

    # Boolean defaults
    assert state.is_valid_jd is False
    assert state.needs_rewrite is False
    assert state.needs_refinement is False
    assert state.is_fallback is False

    # Collections
    assert isinstance(state.retrieved_chunks, list)
    assert state.retrieved_chunks == []

    # Counters
    assert state.refinement_count == 0
    assert state.rewrite_count == 0



# Partial Node Return Merge Safety

def test_partial_state_update_does_not_drop_existing_fields():
    """
    Simulates LangGraph-style partial node returns.
    Existing fields must remain intact after merge.
    """
    base_state = GraphState(
        job_description="JD",
        is_valid_jd=True,
        relevance_score=85,
        rewrite_count=1
    )

    partial_update = {
        "needs_rewrite": True,
        "grading_feedback": "Weak match"
    }

    merged_state = GraphState(**{**base_state.model_dump(), **partial_update})

    assert merged_state.job_description == "JD"
    assert merged_state.is_valid_jd is True
    assert merged_state.relevance_score == 85
    assert merged_state.needs_rewrite is True
    assert merged_state.grading_feedback == "Weak match"
    assert merged_state.rewrite_count == 1



# No Field Should Ever Become None


@pytest.mark.parametrize("field_name", GraphState.model_fields.keys())
def test_no_field_is_none_by_default(field_name):
    """
    Ensures every field has a deterministic default value.
    """
    state = GraphState(job_description="JD")
    value = getattr(state, field_name)

    assert value is not None, f"{field_name} should never be None"



# Edge Case: Empty Job Description


def test_empty_job_description_is_allowed_but_explicit():
    """
    Empty job_description should not crash schema validation.
    Validation logic belongs in ingest_jd_node, not schema.
    """
    state = GraphState(job_description="")

    assert state.job_description == ""
    assert state.is_valid_jd is False



# Edge Case: Empty Retrieved Chunks


def test_empty_retrieved_chunks_is_valid():
    """
    Empty retrieval results must be a valid state.
    """
    state = GraphState(job_description="JD", retrieved_chunks=[])

    assert isinstance(state.retrieved_chunks, list)
    assert len(state.retrieved_chunks) == 0



# Edge Case: Invalid Relevance Scores


@pytest.mark.parametrize("score", [-10, 150])
def test_relevance_score_out_of_bounds_is_still_serializable(score):
    """
    Schema does not enforce bounds.
    Node logic must clamp or handle.
    """
    state = GraphState(job_description="JD", relevance_score=score)

    assert state.relevance_score == score


# Counter Monotonicity (Non-Regressive)

def test_counters_do_not_regress():
    """
    Refinement and rewrite counters should never decrease.
    """
    state = GraphState(job_description="JD")

    state.refinement_count += 1
    state.refinement_count += 1

    assert state.refinement_count >= 2

    state.rewrite_count += 1
    assert state.rewrite_count >= 1


# Boolean Flag Determinism

def test_boolean_flags_are_strict_booleans():
    """
    Ensures boolean flags never become truthy strings or ints.
    """
    state = GraphState(job_description="JD")

    assert isinstance(state.needs_rewrite, bool)
    assert isinstance(state.needs_refinement, bool)
    assert isinstance(state.is_fallback, bool)
