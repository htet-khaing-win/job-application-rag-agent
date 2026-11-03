"""Empty vector DB: retrieval must report no_resumes and never query Pinecone."""
import asyncio
from unittest.mock import Mock, patch

from state import GraphState
from database import retrieve_resumes_node


@patch("database.get_bm25")
@patch("database.get_pinecone_index")
@patch("database.get_embeddings")
@patch("database.list_stored_resumes")
def test_retrieve_resumes_empty_database(mock_list, mock_emb, mock_index, mock_bm25):
    mock_list.return_value = []
    mock_emb.return_value.embed_query.return_value = [0.0] * 1536
    index = mock_index.return_value

    state = GraphState(job_description="Senior Backend Engineer",
                       cleaned_jd="Backend Engineer with Python and APIs", is_valid_jd=True)
    result = asyncio.run(retrieve_resumes_node(state))

    assert result["retrieved_chunks"] == []
    assert result["error_type"] == "no_resumes"
    assert result["vector_relevance_score"] == 0.0
    index.query.assert_not_called()
