"""Deterministic faithfulness scoring for generated letters.

A letter is faithful when every skill it *claims* (outside learning/transfer hedges)
is evidenced in the resume text. No LLM involved, so it is reproducible and free.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from subagents import find_ungrounded_sentences


def _present(skill: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", text.lower()))


def score_letter(letter: str, resume_text: str, skill_vocab: Sequence[str]) -> Dict[str, object]:
    """skill_vocab: skills that may be mentioned (e.g. JD keywords). Returns claim-level stats."""
    unsupported = [s for s in skill_vocab if not _present(s, resume_text)]
    bad = find_ungrounded_sentences(letter, unsupported)
    claimed = [s for s in skill_vocab if _present(s, letter)]
    unsupported_claims = [s for s in claimed if s in unsupported and any(_present(s, b) for b in bad)]
    n = max(1, len(claimed))
    return {
        "claimed_skills": claimed,
        "unsupported_claims": unsupported_claims,
        "hallucination_rate": round(len(unsupported_claims) / n, 4),
        "ungrounded_sentences": bad,
    }


def aggregate(scores: List[Dict[str, object]]) -> Dict[str, float]:
    rates = [float(s["hallucination_rate"]) for s in scores]
    return {
        "letters": len(scores),
        "mean_hallucination_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "letters_with_any_hallucination": sum(1 for r in rates if r > 0),
    }
