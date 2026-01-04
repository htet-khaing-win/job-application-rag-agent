"""
Module: tests/ingest_jd_node_validation_test.py

PURPOSE:
This suite validates the 'ingest_jd_node', which is the entry point of the 
Job Assistant graph. It ensures that the system can correctly distinguish 
between valid job descriptions and irrelevant user input.

TESTING STRATEGY: MOCKING & ISOLATION
Instead of calling a live LLM (which is slow and costly), we use 'unittest.mock' 
to simulate various LLM responses (e.g., "VALID", "INVALID"). This allows us 
to test the node's branching logic deterministically.

KEY VALIDATIONS:
1. Classification: Does the node set 'is_valid_jd' correctly based on LLM output?
2. Error Mapping: Does it populate 'error_type' and 'error_message' on failure?
3. Efficiency: Does it skip the expensive 'cleaning' LLM call if validation fails?
4. Resilience: Can it handle LLM outputs with extra whitespace or mixed casing?

TECHNICAL NOTE:
In accordance with LangGraph best practices, these tests verify that the node 
returns a 'dict' representing the state delta, rather than a full GraphState object.

USAGE: pytest tests/ingest_jd_node_validation_test.py -v
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import Mock
from state import GraphState
from node import ingest_jd_node

# Helper: Mock LLM

def make_llm_mock(response_text: str):
    llm = Mock()
    llm.invoke.return_value = Mock(content=response_text)
    return llm


# VALID Job Description Path

def test_valid_job_description_sets_is_valid_true_and_cleans_jd():
    llm = make_llm_mock("VALID")

    state = GraphState(job_description="We are hiring a Python Engineer...")

    result = ingest_jd_node(state, llm)

    assert result["is_valid_jd"] is True
    assert result["cleaned_jd"] != ""
    assert result.get("error_type", "") == ""
    assert result.get("error_message", "") == ""



# INVALID Job Description (Uppercase)


def test_invalid_job_description_sets_error_fields():
    llm = make_llm_mock("INVALID")

    state = GraphState(job_description="Tell me a joke")

    result = ingest_jd_node(state, llm)

    assert result["is_valid_jd"] is False
    assert result["error_type"] == "invalid_input"
    assert "job description" in result["error_message"].lower()
    assert result.get("cleaned_jd", "") == ""



# INVALID Job Description (Lowercase)

def test_invalid_job_description_lowercase_invalid():
    llm = make_llm_mock("invalid")

    state = GraphState(job_description="hello how are you")

    result = ingest_jd_node(state, llm)

    assert result["is_valid_jd"] is False
    assert result["error_type"] == "invalid_input"



# INVALID with Extra Text (LLM Misbehavior)

def test_invalid_job_description_with_extra_text():
    llm = make_llm_mock("INVALID\nReason: Not a job posting")

    state = GraphState(job_description="What's your name?")

    result = ingest_jd_node(state, llm)

    assert result["is_valid_jd"] is False
    assert result["error_type"] == "invalid_input"



# Question-Like Job Description

def test_question_like_input_is_invalid():
    llm = make_llm_mock("INVALID")

    state = GraphState(job_description="Should I apply to Google?")

    result = ingest_jd_node(state, llm)

    assert result["is_valid_jd"] is False
    assert result["error_type"] == "invalid_input"



# Ensure Cleaning Is NOT Called on Invalid Input

def test_cleaning_not_triggered_for_invalid_input():
    llm = Mock()

    # First call → INVALID
    llm.invoke.return_value = Mock(content="INVALID")

    state = GraphState(job_description="Random text")

    result = ingest_jd_node(state, llm)

    # Only ONE invoke call (validation prompt)
    assert llm.invoke.call_count == 1
    assert result["is_valid_jd"] is False
    assert result.get("cleaned_jd", "") == ""
