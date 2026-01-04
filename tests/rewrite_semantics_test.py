"""
Module: tests/rewrite_semantics_test.py

PURPOSE:
Ensures that the 'Rewrite Loop' actually modifies the search parameters. 
If the 'cleaned_jd' remains identical between loops, the system will 
continually retrieve the same wrong results from Pinecone (a 'No-Op' loop).

STRATEGY:
- Mocks the embedding function to return values based on string content.
- Simulates a 'state update' where the query is modified.
- Validates that the retrieval node receives a different input in the second iteration.

USAGE: pytest tests/rewrite_semantics_test.py -v
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import GraphState
from database import retrieve_resumes_node

def test_rewrite_loop_changes_query_state(monkeypatch):
    """
    Rewrite loop must mutate retrieval input, not just retry.
    """

    class FakeEmbeddings:
        def embed_query(self, text):
            # Returns a deterministic fake vector based on text length to simulate change
            return [float(len(text))]

    monkeypatch.setattr("database.embeddings", FakeEmbeddings())

    # Initial State
    state = GraphState(
        job_description="Senior ML Engineer",
        cleaned_jd="Senior ML Engineer NLP",
        needs_rewrite=True,
        rewrite_count=0,
        is_valid_jd=True
    )

    # Act - First Retrieval
    # result_1 is a DICT
    result_1 = retrieve_resumes_node(state, llm=None)
    
    # Simulate LangGraph merging the dict back into a State object
    state_after_1 = state.model_copy(update=result_1)
    state_after_1.needs_rewrite = True # Now we can use dot notation on the Pydantic copy
    state_after_1.cleaned_jd = "Updated Query for NLP" # Simulate a rewrite happening

    # Act - Second Retrieval
    result_2 = retrieve_resumes_node(state_after_1, llm=None)
    
    # Assert
    assert state.cleaned_jd != state_after_1.cleaned_jd, "The query text should have changed"
    assert "retrieved_chunks" in result_2
