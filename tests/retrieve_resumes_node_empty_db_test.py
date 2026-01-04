"""
Module: tests/retrieve_resumes_node_empty_db_test.py

PURPOSE:
This test validates the system's "Safe Fail" logic when no resumes are found 
in the Vector Database. It ensures the assistant handles an empty database 
gracefully rather than crashing or providing irrelevant matches.

TESTING STRATEGY: DOUBLE-MOCKING
We mock both the 'list_stored_resumes' helper and the Pinecone 'index' object. 
By simulating an empty list return, we verify that the node is smart enough 
to skip the expensive Vector Search entirely.

KEY PROTECTIONS:
1. Resource Efficiency: Confirms 'index.query' is NOT called if no resumes exist.
2. State Integrity: Ensures 'retrieved_chunks' is explicitly set to an empty list.
3. User Feedback: Verifies that 'error_type' is set to 'no_resumes' so the 
   UI can prompt the user to upload a resume.

SCENARIO:
User provides a valid JD, but this is a fresh installation with 0 resumes indexed.

USAGE: pytest tests/retrieve_resumes_node_empty_db_test.py -v
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import Mock, patch
from state import GraphState
from database import retrieve_resumes_node


# Empty Pinecone Database Handling

@patch("database.list_stored_resumes")
@patch("database.embeddings")
@patch("database.index")
def test_retrieve_resumes_empty_database(
    mock_index,
    mock_embeddings,
    mock_list_resumes
):
    """
    If no resumes exist in Pinecone:
    - retrieved_chunks must be empty
    - error_type must be 'no_resumes'
    - Pinecone query must NEVER be called
    """

    # Arrange
    mock_list_resumes.return_value = []
    mock_embeddings.embed_query.return_value = [0.0] * 768

    state = GraphState(
        job_description="Senior Backend Engineer",
        cleaned_jd="Backend Engineer with Python and APIs",
        is_valid_jd=True
    )

    # Act
    result = retrieve_resumes_node(state, llm=Mock())

    # Assert: Core outputs
    assert result["retrieved_chunks"] == []
    assert result["error_type"] == "no_resumes"
    assert "upload" in result["grading_feedback"].lower()

    # Assert: Pinecone never queried
    mock_index.query.assert_not_called()
