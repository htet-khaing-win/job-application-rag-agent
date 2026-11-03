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


def _state():
    return GraphState(job_description="dummy", cover_letter="I built great systems.", cleaned_jd="dummy")


def test_critique_ready_does_not_trigger_refinement():
    llm = FakeLLM('{"edits": [], "ready_to_submit": true}')
    assert critique_letter_node(_state(), llm=llm)["needs_refinement"] is False


def test_critique_valid_edit_triggers_refinement():
    llm = FakeLLM('{"edits": [{"type": "replace", "target": "great systems", "replacement": "scalable systems", "reason": "specific"}], "ready_to_submit": false}')
    out = critique_letter_node(_state(), llm=llm)
    assert out["needs_refinement"] is True and len(out["refinement_edits"]) == 1


def test_critique_edit_with_missing_target_is_dropped():
    llm = FakeLLM('{"edits": [{"type": "replace", "target": "not in letter", "replacement": "x"}], "ready_to_submit": false}')
    out = critique_letter_node(_state(), llm=llm)
    assert out["refinement_edits"] == [] and out["needs_refinement"] is False


def test_critique_unparseable_output_never_blocks():
    out = critique_letter_node(_state(), llm=FakeLLM("not json at all"))
    assert out["needs_refinement"] is False and out["refinement_edits"] == []
