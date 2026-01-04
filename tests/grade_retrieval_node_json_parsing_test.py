"""
Module: tests/grade_retrieval_node_json_parsing_test.py

PURPOSE:
LLMs are notoriously unreliable at returning perfectly formatted JSON. This 
suite tests the 'Resilience' of the Grading Node—ensuring the graph doesn't 
crash if the LLM adds conversational filler, misses a key, or breaks syntax.

TESTING STRATEGY: RAG RECOVERY
We simulate four common LLM failure modes:
1. Perfect JSON: The ideal case.
2. Noisy JSON: JSON wrapped in conversational text (e.g., "Sure, here it is...").
3. Partial JSON: Valid JSON but missing the critical 'needs_rewrite' key.
4. Malformed JSON: Broken syntax that requires Regex extraction fallback.

KEY VALIDATIONS:
- Logic Safety: If a score is low (< 70), the node should flag a rewrite even 
   if the LLM forgot the boolean flag.
- Type Integrity: Relevance scores must always be integers, never strings.
- Feedback Continuity: Ensure reasoning is captured for the UI.

USAGE: pytest tests/grade_retrieval_node_json_parsing_test.py -v
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import Mock
from state import GraphState
from node import grade_retrieval_node

# Helper: Mock LLM

def make_llm_mock(response_text: str):
    llm = Mock()
    llm.invoke.return_value = Mock(content=response_text)
    return llm

# Perfect JSON Response

def test_grade_retrieval_perfect_json():
    llm = make_llm_mock(
        """
        {
            "score": 85,
            "needs_rewrite": false,
            "reasoning": "Strong alignment with job requirements."
        }
        """
    )

    state = GraphState(
        job_description="JD",
        cleaned_jd="Python backend role",
        retrieved_chunks=[{"text": "Python API experience"}]
    )

    result = grade_retrieval_node(state, llm)

    assert isinstance(result["relevance_score"], int)
    assert result["relevance_score"] == 85
    assert result["needs_rewrite"] is False
    assert result["grading_feedback"] != ""

# JSON with Extra Text (Regex Fallback)

def test_grade_retrieval_json_with_extra_text():
    llm = make_llm_mock(
        """
        Sure! Here's my assessment:

        {
            "score": 62,
            "needs_rewrite": true,
            "reasoning": "Missing key skills."
        }

        Hope this helps!
        """
    )

    state = GraphState(
        job_description="JD",
        cleaned_jd="ML Engineer role",
        retrieved_chunks=[{"text": "Some experience"}]
    )

    result = grade_retrieval_node(state, llm)

    assert isinstance(result["relevance_score"], int)
    assert result["relevance_score"] == 62
    assert result["needs_rewrite"] is True
    assert result["grading_feedback"] != ""

# Missing needs_rewrite Key

def test_grade_retrieval_missing_needs_rewrite_key():
    llm = make_llm_mock(
        """
        {
            "score": 40,
            "reasoning": "Very weak match."
        }
        """
    )

    state = GraphState(
        job_description="JD",
        cleaned_jd="Data Scientist",
        retrieved_chunks=[{"text": "Unrelated experience"}]
    )

    result = grade_retrieval_node(state, llm)

    assert result["relevance_score"] == 40
    # Default behavior: low score forces rewrite
    assert result["needs_rewrite"] is True
    assert result["grading_feedback"] != ""

# Malformed JSON (No Quotes / Broken Syntax)

def test_grade_retrieval_malformed_json():
    llm = make_llm_mock(
        """
        {score: 80, needs_rewrite: false, reasoning: Strong candidate}
        """
    )

    state = GraphState(
        job_description="JD",
        cleaned_jd="Full Stack Engineer",
        retrieved_chunks=[{"text": "React and Python"}]
    )

    result = grade_retrieval_node(state, llm)

    # Fallback regex extraction
    assert isinstance(result["relevance_score"], int)
    assert 0 <= result["relevance_score"] <= 100

    # If rewrite flag missing, fallback logic applies
    assert isinstance(result["needs_rewrite"], bool)
    assert result["grading_feedback"] != ""
