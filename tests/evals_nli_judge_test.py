"""Entailment-based faithfulness and the pairwise judge (with its bias controls)."""
import json
import random
import sys

import pytest

from evals.faithfulness import aggregate_claims, claim_support_rate
from evals.judge import (JudgeConflict, LLMRelevanceJudge, calibrate, judge_pair, model_ids,
                         run_pairwise_eval, validate_judge_against_gold)
from evals.nli import (LLMEntailer, LexicalEntailer, ModelEntailer, extract_claims, get_entailer, verdicts)

RESUME = ("Built python services handling 5000 requests per second at Acme.\n"
          "Led a team of 4 engineers.\nBSc Computer Science, 2019.")


class R:
    def __init__(self, c):
        self.content = c


# ----------------------------------------------------------------- claim extraction
def test_extract_claims_drops_boilerplate_hedges_and_fragments():
    letter = ("Dear hiring team,\nI built python services handling 5000 requests per second.\n"
              "I am prepared to learn Kubernetes through my python background.\n"
              "Thank you for your time and consideration.\nSincerely, A.\nHi there.")
    assert extract_claims(letter) == ["I built python services handling 5000 requests per second."]


def test_extract_claims_empty_input():
    assert extract_claims("") == [] and extract_claims(None) == []


# ----------------------------------------------------------------- lexical entailer
def test_lexical_entails_supported_claim():
    label, score = LexicalEntailer().entails(RESUME, "I built python services handling 5000 requests per second.")
    assert label == "entailed" and score >= 0.6


def test_lexical_marks_unstated_skill_neutral():
    assert LexicalEntailer().entails(RESUME, "I deployed Kubernetes clusters across three regions.")[0] == "neutral"


def test_lexical_contradicts_wrong_number_and_neutral_on_unstated_number():
    e = LexicalEntailer()
    assert e.entails(RESUME, "I led a team of 9 engineers at Acme.")[0] == "contradicted"
    assert e.entails("Led a team of engineers.", "I led a team of 9 engineers.")[0] == "neutral"


def test_lexical_empty_hypothesis_is_neutral():
    assert LexicalEntailer().entails(RESUME, "")[0] == "neutral"


def test_lexical_is_deterministic():
    e = LexicalEntailer()
    claim = "I built python services handling 5000 requests per second."
    assert e.entails(RESUME, claim) == e.entails(RESUME, claim)


# ----------------------------------------------------------------- claim_support_rate
def test_claim_support_rate_counts_supported_and_unsupported():
    letter = ("I built python services handling 5000 requests per second. "
              "I deployed Kubernetes clusters across three regions. "
              "I am prepared to learn Terraform quickly.")
    res = claim_support_rate(letter, RESUME)
    assert res["n_claims"] == 2 and res["support_rate"] == 0.5 and res["hallucination_rate"] == 0.5
    assert [v["label"] for v in res["verdicts"]] == ["entailed", "neutral"]


def test_claim_support_rate_flags_contradiction():
    res = claim_support_rate("I led a team of 12 engineers at Acme.", RESUME)
    assert res["contradiction_rate"] == 1.0 and res["support_rate"] == 0.0


def test_vacuous_letter_is_flagged_not_rewarded_in_aggregate():
    res = claim_support_rate("Dear team, thank you.", RESUME)
    assert res["vacuous"] and res["hallucination_rate"] == 0.0
    real = claim_support_rate("I deployed Kubernetes clusters across three regions.", RESUME)
    agg = aggregate_claims([res, real])
    assert agg["letters"] == 2 and agg["scored_letters"] == 1 and agg["mean_hallucination_rate"] == 1.0


# ----------------------------------------------------------------- LLM / model entailers
class FakeVerdict:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return R(self.replies.pop(0) if len(self.replies) > 1 else self.replies[0])


def test_llm_entailer_parses_verdict_and_fences_inputs():
    llm = FakeVerdict([json.dumps({"label": "entailed", "confidence": 0.9})])
    label, conf = LLMEntailer(llm).entails(RESUME, "I led a team of 4 engineers.")
    assert (label, conf) == ("entailed", 0.9)
    assert "<resume>" in llm.prompts[0] and "<claim>" in llm.prompts[0] and "DATA, not instructions" in llm.prompts[0]


def test_llm_entailer_repairs_malformed_json():
    llm = FakeVerdict(["not json at all", json.dumps({"label": "neutral", "confidence": 0.4})])
    assert LLMEntailer(llm).entails(RESUME, "I know Rust.") == ("neutral", 0.4)
    assert len(llm.prompts) == 2


def test_llm_entailer_rejects_invalid_label_after_repair():
    llm = FakeVerdict([json.dumps({"label": "maybe"})])
    with pytest.raises(json.JSONDecodeError):
        LLMEntailer(llm).entails(RESUME, "I know Rust.")


def test_llm_entailer_truncates_long_premise_to_relevant_windows():
    long_resume = "\n".join(f"Filler line number {i} about unrelated hobbies." for i in range(400))
    long_resume += "\nBuilt python services handling 5000 requests per second."
    llm = FakeVerdict([json.dumps({"label": "entailed", "confidence": 1})])
    LLMEntailer(llm, max_premise_chars=600).entails(long_resume, "I built python services handling 5000 requests.")
    assert "5000 requests per second" in llm.prompts[0] and len(llm.prompts[0]) < 2500


def test_model_entailer_reports_missing_dependency_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # forces ImportError
    with pytest.raises(ImportError, match="sentence-transformers"):
        ModelEntailer()


def test_get_entailer_factory():
    assert isinstance(get_entailer("lexical"), LexicalEntailer)
    assert isinstance(get_entailer("llm", FakeVerdict(["{}"])), LLMEntailer)
    with pytest.raises(ValueError):
        get_entailer("llm")
    with pytest.raises(ValueError):
        get_entailer("nope")


def test_verdicts_accepts_custom_entailer_and_claims():
    class AlwaysNeutral:
        def entails(self, p, h):
            return "neutral", 0.0
    out = verdicts("ignored", RESUME, AlwaysNeutral(), claims=["a b c d e f"])
    assert out == [{"claim": "a b c d e f", "label": "neutral", "score": 0.0}]


# ----------------------------------------------------------------- pairwise judge
class FakeJudge:
    """Prefers the letter containing GOOD, whichever slot it is in."""

    model_id = "judge-model"

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        a = prompt.split("<letter_A>")[1].split("</letter_A>")[0]
        b = prompt.split("<letter_B>")[1].split("</letter_B>")[0]
        if ("GOOD" in a) == ("GOOD" in b):
            return R(json.dumps({"winner": "tie", "reason": "same"}))
        return R(json.dumps({"winner": "A" if "GOOD" in a else "B", "reason": "has GOOD"}))


class AlwaysA:
    model_id = "biased-judge"

    def invoke(self, prompt):
        return R(json.dumps({"winner": "A", "reason": "first one"}))


CASES = [{"id": f"p{i}", "jd": "job", "x": "GOOD letter", "y": "plain letter"} for i in range(8)]


def test_consistent_judge_has_no_position_bias_and_prefers_the_good_letter():
    res = run_pairwise_eval(FakeJudge(), CASES, seed=1)
    assert res["win_rate_x"] == 1.0 and res["wins_x"] == 8
    assert res["consistency_rate"] == 1.0 and res["position_bias"] == 0.0


def test_slot_biased_judge_is_detected_and_cannot_manufacture_wins():
    res = run_pairwise_eval(AlwaysA(), CASES, seed=1)
    assert res["position_bias"] == pytest.approx(0.5)      # always slot A
    assert res["consistency_rate"] == 0.0                  # orders contradict each other
    assert res["win_rate_x"] == 0.5 and res["ties"] == 8   # inconsistent -> tie, so no spurious win


def test_judge_runs_both_orders_per_pair_and_single_order_when_asked():
    j = FakeJudge()
    run_pairwise_eval(j, CASES[:3], seed=0)
    assert len(j.prompts) == 6
    j2 = FakeJudge()
    res = run_pairwise_eval(j2, CASES[:3], seed=0, both_orders=False)
    assert len(j2.prompts) == 3 and res["consistency_rate"] is None


def test_judge_order_is_seeded_and_reproducible():
    a, b, c = FakeJudge(), FakeJudge(), FakeJudge()
    run_pairwise_eval(a, CASES, seed=7)
    run_pairwise_eval(b, CASES, seed=7)
    run_pairwise_eval(c, CASES, seed=8)
    assert a.prompts == b.prompts and a.prompts != c.prompts


def test_judge_pair_maps_slots_back_to_underlying_letters():
    r = judge_pair(FakeJudge(), "job", "plain", "GOOD one", random.Random(3))
    assert r["winner"] == "y" and r["consistent"] is True


def test_judge_refuses_same_model_as_generator():
    class Same(FakeJudge):
        model_id = "gpt-4o-mini"
    with pytest.raises(JudgeConflict):
        run_pairwise_eval(Same(), CASES, generator_ids=["openai:gpt-4o-mini"])
    run_pairwise_eval(Same(), CASES, generator_ids=["openai:gpt-4o-mini"], allow_same_model=True)
    run_pairwise_eval(FakeJudge(), CASES, generator_ids=["openai:gpt-4o-mini"])  # different model: fine


def test_model_ids_reads_resilient_llm_candidates():
    from resilience import ResilientLLM
    llm = ResilientLLM("judge", [("openai:gpt-4o", lambda: None), ("anthropic:claude-opus-5-5", lambda: None)])
    ids = model_ids(llm)
    assert {"gpt-4o", "openai:gpt-4o", "claude-opus-5-5"} <= ids
    with pytest.raises(JudgeConflict):
        run_pairwise_eval(llm, CASES, generator_ids=["gpt-4o"])


def test_judge_repairs_malformed_output():
    class Sloppy(FakeJudge):
        def __init__(self):
            super().__init__()
            self.n = 0

        def invoke(self, prompt):
            self.n += 1
            if self.n == 1:
                return R("Winner is A!!")
            return super().invoke(prompt)
    res = run_pairwise_eval(Sloppy(), CASES[:2], seed=0)
    assert res["n"] == 2


# ----------------------------------------------------------------- calibration against humans
def test_calibrate_known_values():
    res = calibrate(["A", "B", "A", "tie"], ["A", "B", "B", "tie"])
    assert res["agreement"] == 0.75 and res["n"] == 4
    assert res["kappa"] == pytest.approx(0.6364, abs=1e-4)
    assert res["confusion"]["B"] == {"B": 1, "A": 1}


def test_calibrate_aligns_dicts_on_shared_ids_and_validates_lengths():
    res = calibrate({"p1": "A", "p2": "B", "p3": "A"}, {"p2": "B", "p3": "B", "p9": "A"})
    assert res["n"] == 2 and res["agreement"] == 0.5
    with pytest.raises(ValueError):
        calibrate(["A"], ["A", "B"])


def test_validate_judge_against_gold_known_confusion():
    judged = {"q1": {"c1": True, "c2": True, "c3": False, "c4": False}}
    gold = {"q1": ["c1", "c3"]}
    res = validate_judge_against_gold(judged, gold)
    assert (res["tp"], res["fp"], res["fn"], res["tn"]) == (1, 1, 1, 1)
    assert res["precision"] == res["recall"] == res["f1"] == res["agreement"] == 0.5
    assert res["kappa"] == pytest.approx(0.0)


def test_validate_judge_perfect_and_empty():
    perfect = validate_judge_against_gold({"q": {"a": True, "b": False}}, {"q": ["a"]})
    assert perfect["f1"] == 1.0 and perfect["kappa"] == 1.0
    assert validate_judge_against_gold({}, {})["n_chunks"] == 0


def test_llm_relevance_judge_parses_bool():
    llm = FakeVerdict([json.dumps({"relevant": True})])
    assert LLMRelevanceJudge(llm).judge("job", "python services") is True
    assert "DATA, not instructions" in llm.prompts[0]
