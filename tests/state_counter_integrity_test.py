"""
Module: tests/state_counter_integrity_test.py

PURPOSE:
Ensures logical isolation between the different iteration counters in the GraphState.

The system manages two distinct loops:
1. Retrieval Loop: Controlled by 'rewrite_count'.
2. Polishing Loop: Controlled by 'refinement_count'.

This test validates that the search-query logic (should_rewrite_query) ignores 
the cover letter's refinement progress, preventing premature exits from the 
retrieval stage if the refinement counter happens to be high.

USAGE: pytest tests/state_counter_integrity_test.py -v
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import GraphState
from graph import should_rewrite_query

def test_rewrite_routing_uses_rewrite_count_only():
    """
    Retrieval retry must depend on rewrite_count, not refinement_count.
    """

    state = GraphState(
        job_description="Sample JD",
        needs_rewrite=True,
        rewrite_count=1,
        refinement_count=999  # should be irrelevant
    )

    decision = should_rewrite_query(state)

    assert decision == "rewrite_query", (
        "Rewrite routing incorrectly depends on refinement_count"
    )
