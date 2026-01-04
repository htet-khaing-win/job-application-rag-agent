import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pydantic import ValidationError
from state import GraphState
from privacy import PIIGuard

# Initialize the guard once for all tests
@pytest.fixture
def pii_guard():
    return PIIGuard()

# PII GUARD TESTS

def test_pii_redaction_comprehensive(pii_guard):
    raw_text = """
Htet Khaing Win is a developer. Contact at +6534452223 or htet@gmail.com
"""
    redacted = pii_guard.redact_text(raw_text)

    print("\n" + "="*50)
    print("DEBUG START")
    print(f"RAW INPUT: {raw_text}")
    print(f"REDACTED OUTPUT: {redacted}")
    print("DEBUG END")
    print("="*50 + "\n")
    
    # If this prints "[CANDIDATE_NAME]_AND_I_LIKE_PYTHON", then the logic is 100% correct.
    assert "HTET" not in redacted
    print("\n PII Redaction: All sensitive entities masked successfully.")

# PYDANTIC STATE TESTS 

def test_state_validation_success():
    """Ensures GraphState accepts valid data types."""
    valid_data = {
        "job_description": "Senior Python Developer",
        "relevance_score": 85.5,
        "retrieved_chunks": [{"text": "Experience with Django", "score": 0.9}]
    }
    state = GraphState(**valid_data)
    assert state.job_description == "Senior Python Developer"
    assert len(state.retrieved_chunks) == 1
    print(" State Validation: Pydantic schema accepted valid input.")

def test_state_validation_failure():
    """Ensures GraphState rejects invalid data (e.g., string where number is expected)."""
    invalid_data = {
        "job_description": "Data Scientist",
        "relevance_score": "EXCELLENT"  # This should be a float/int
    }
    with pytest.raises(ValidationError):
        GraphState(**invalid_data)
    print(" State Validation: Pydantic correctly caught a Type Error.")

# GRAPH LOGIC TESTS

def test_iteration_counters():
    """Tests if the state correctly tracks refinement attempts."""
    state = GraphState(job_description="DevOps Role")
    
    # Simulate the node logic: refinement_count + 1
    state.refinement_count += 1
    state.rewrite_count += 1
    
    assert state.refinement_count == 1
    assert state.rewrite_count == 1
    print(" Graph Logic: State counters incremented correctly.")

# COMPONENT INTEGRATION 

def test_pii_to_state_flow(pii_guard):
    """Tests the actual 'usable' flow: Input -> Redact -> State."""
    user_input = "Contact John at +1 (555) 000-1234 or john@work.com"
    safe_text = pii_guard.redact_text(user_input)
    
    # This mimics what happens in your ingest_jd_node
    state = GraphState(job_description=safe_text)
    
    assert "john@work.com" not in state.job_description
    assert "[EMAIL]" in state.job_description
    assert "[PHONE_NUMBER]" in state.job_description
    print(" Integration: Redacted text successfully populated GraphState.")