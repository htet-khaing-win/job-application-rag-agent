"""
Module: tests/critique_parsing_test.py

PURPOSE:
Validates the NLP parsing logic within the Critique Node. This node 
does not return JSON; it returns a natural language assessment.

TEST STRATEGY:
- Simulated "Submit?" checks: Ensures "Yes" (ready) and "NO" (needs work) 
  are correctly captured from raw LLM text blocks.
- Case Sensitivity: Checks if the parser handles varied casing (Yes vs NO).
- Validation Safety: Ensures Pydantic state constraints are met during testing.

USAGE: pytest tests/critique_parsing_test.py -v
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from state import GraphState
from node import critique_letter_node

class FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, prompt):
        class R:
            content = self._content
        return R()


def test_critique_ready_yes_does_not_trigger_refinement():
    llm = FakeLLM(
        """
        Strengths: Clear and specific.
        Issues to Fix: None.
        Recommended Changes: None.
        Ready to Submit? Yes.
        """
    )

    state = GraphState(
        job_description="dummy",
        cover_letter="dummy",
        cleaned_jd="dummy"
    )

    new_state = critique_letter_node(state, llm=llm)

    assert new_state["needs_refinement"] is False


def test_critique_ready_no_triggers_refinement():
    llm = FakeLLM(
        """
        Strengths: Good structure.
        Issues to Fix: Add metrics.
        Recommended Changes: Rewrite paragraph 2.
        Ready to Submit? NO
        """
    )

    state = GraphState(
        job_description="dummy",
        cover_letter="dummy",
        cleaned_jd="dummy"
    )

    new_state = critique_letter_node(state, llm=llm)

    assert new_state["needs_refinement"] is True
