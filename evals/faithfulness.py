"""Deterministic faithfulness scoring for generated letters.

A letter is faithful when every skill it *claims* (outside learning/transfer hedges)
is evidenced in the resume text. No LLM involved, so it is reproducible and free.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from evals.nli import Entailer, verdicts
from subagents import find_ungrounded_sentences


def _present(skill: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", text.lower()))


def score_letter(letter: str, resume_text: str, skill_vocab: Sequence[str]) -> Dict[str, Any]:
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


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    rates = [float(s["hallucination_rate"]) for s in scores]
    return {
        "letters": len(scores),
        "mean_hallucination_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "letters_with_any_hallucination": sum(1 for r in rates if r > 0),
    }


# --- claim-level (entailment) faithfulness -------------------------------------------------
def claim_support_rate(letter: str, resume_text: str, entailer: Optional[Entailer] = None) -> Dict[str, Any]:
    """Share of the letter's factual claims the resume supports.

    Returns per-claim verdicts plus `support_rate`, `contradiction_rate` and
    `hallucination_rate` (= claims not entailed / claims). A letter with no checkable claim
    scores support 1.0 / hallucination 0.0 and is flagged `vacuous`.
    """
    vs = verdicts(letter, resume_text, entailer)
    n = len(vs)
    if n == 0:
        return {"n_claims": 0, "support_rate": 1.0, "contradiction_rate": 0.0,
                "hallucination_rate": 0.0, "vacuous": True, "verdicts": []}
    ent = sum(1 for v in vs if v["label"] == "entailed")
    con = sum(1 for v in vs if v["label"] == "contradicted")
    return {"n_claims": n, "support_rate": round(ent / n, 4), "contradiction_rate": round(con / n, 4),
            "hallucination_rate": round((n - ent) / n, 4), "vacuous": False, "verdicts": vs}


def aggregate_claims(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    ok = [r for r in results if not r.get("vacuous")]
    rates = [r["hallucination_rate"] for r in ok]
    return {
        "letters": len(results),
        "scored_letters": len(ok),
        "mean_hallucination_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "mean_support_rate": round(sum(r["support_rate"] for r in ok) / len(ok), 4) if ok else 1.0,
        "letters_with_any_unsupported_claim": sum(1 for r in ok if r["hallucination_rate"] > 0),
    }
