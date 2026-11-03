"""Chaos harness, ablation runner and adversarial corpus, exercised through the REAL compiled graph
(fake models + stubbed Pinecone/Tavily, no keys)."""
import asyncio
import json

import pytest

from evals import ablation
from evals.adversarial import company_injections, injection_cases, load_corpus, stuffing_cases
from evals.chaos import Scenario, classify, default_scenarios, run_chaos, _rl
from evals.faithfulness import score_letter
from evals.fakes import JD, RESUME, SECRET, ScriptedModel, stub_backends


# ----------------------------------------------------------------- chaos
@pytest.fixture(scope="module")
def chaos():
    return run_chaos(trials=4, seed=0)


def test_chaos_recovery_thresholds(chaos):
    s = chaos["summary"]
    assert s["recovery_rate"] >= 0.95
    assert s["fallback_rate"] == 0.0                # no unexpected fallbacks
    assert s["expected_fallback_ok_rate"] == 1.0    # total writer outage ends in a clean fallback
    assert s["expectations_met_rate"] >= 0.95


def test_chaos_no_secret_ever_leaks(chaos):
    assert chaos["summary"]["secret_leaks"] == 0


def test_chaos_scenarios_have_expected_outcomes(chaos):
    sc = chaos["scenarios"]
    assert sc["control"]["outcomes"] == {"clean": 4}
    assert sc["total_writer_outage"]["outcomes"] == {"fallback": 4}
    assert set(sc["critic_outage"]["outcomes"]) <= {"degraded", "recovered"}
    assert "fallback" not in sc["dead_provider"]["outcomes"] and "crashed" not in sc["dead_provider"]["outcomes"]
    for name in ("timeout_burst", "rate_limit_burst", "malformed_json", "auth_failure_primary", "intermittent"):
        assert sc[name]["expectation_met_rate"] == 1.0, name


def test_chaos_recovery_time_is_reported(chaos):
    s = chaos["summary"]
    assert 0 < s["time_to_recover_mean_s"] <= s["time_to_recover_p95_s"] + 1e-9


def test_chaos_is_deterministic_for_a_seed():
    a = run_chaos(trials=2, seed=5)
    b = run_chaos(trials=2, seed=5)
    assert {k: v["outcomes"] for k, v in a["scenarios"].items()} == {k: v["outcomes"] for k, v in b["scenarios"].items()}


def test_leak_scanner_actually_detects_a_leak():
    """Guard against a vacuous 'secret_leaks == 0': a model that echoes the key into the letter must be caught."""
    leaky = Scenario("leaky", "recover",
                     lambda s: (_rl("writer", ScriptedModel(letter=f"I built python services. key={SECRET}")),
                                _rl("critic", ScriptedModel())))
    res = run_chaos(trials=1, scenarios=[leaky])
    assert res["summary"]["secret_leaks"] > 0


def test_unexpected_fallback_lowers_recovery_rate():
    from evals.fakes import FaultyModel, always
    dead = Scenario("all_dead", "recover",
                    lambda s: (_rl("writer", FaultyModel(ScriptedModel(), always(ConnectionError("503")))),
                               _rl("critic", ScriptedModel())))
    res = run_chaos(trials=2, scenarios=[dead])
    assert res["summary"]["recovery_rate"] == 0.0 and res["summary"]["fallback_rate"] == 1.0


def test_classify_outcomes():
    assert classify({"cover_letter": "x", "healing_log": []}) == "clean"
    assert classify({"cover_letter": "x", "healing_log": ["[writer] recovered on b (attempt 2)"]}) == "recovered"
    assert classify({"cover_letter": "x", "healing_log": ["[critic] unavailable: skipping critique, shipping"]}) == "degraded"
    assert classify({"cover_letter": "", "is_fallback": True}) == "fallback"
    assert classify({"cover_letter": "   "}) == "fallback"


def test_default_scenarios_cover_every_requested_fault_kind():
    names = {s.name for s in default_scenarios()}
    assert {"timeout_burst", "rate_limit_burst", "malformed_json", "auth_failure_primary", "dead_provider",
            "intermittent", "critic_outage", "total_writer_outage", "control"} <= names


def test_chaos_cli_returns_zero(capsys):
    from evals.__main__ import main
    assert main(["chaos", "--trials", "1", "--no-report"]) == 0
    assert "secret_leaks" in capsys.readouterr().out


# ----------------------------------------------------------------- ablation (injected runner)
HALLU = "I led Kubernetes migrations across 12 production clusters."
GOOD = "I built python services handling 5000 requests per second."


def fake_runner(case, ablate):
    letter = GOOD + (" " + HALLU if "fact_checker" in ablate else "")
    return {"cover_letter": letter, "cost_usd": 0.01 + 0.005 * len(ablate),
            "node_timings": [{"node": "Copywriter", "ms": 1000.0 + 100 * len(ablate)}]}


CASES = [{"id": f"c{i}", "jd": JD, "resume_text": RESUME} for i in range(6)]


def test_ablation_names():
    assert ablation.ablation_name(()) == "full"
    assert ablation.ablation_name({"fact_checker"}) == "no_fact_checker"
    assert ablation.ablation_name(ablation.COMPONENTS) == "all_off"
    assert ablation.ablation_name({"research", "failover"}) == "no_failover+research"


def test_ablation_measures_fact_checker_before_after():
    res = ablation.run_ablation(CASES, fake_runner, repeats=2)
    s = res["summary"]
    assert set(s) == {"full", "no_fact_checker", "no_critique_refine", "no_research", "no_failover", "all_off"}
    assert s["full"]["hallucination_rate"]["mean"] == 0.0
    assert s["no_fact_checker"]["hallucination_rate"]["mean"] == 0.5     # 1 of 2 claims unsupported
    fc = res["fact_checker_effect"]
    assert fc["n_cases"] == 6 and fc["reduction"] == 0.5
    assert fc["hallucination_before_fact_checker"] == 0.5 and fc["hallucination_after_fact_checker"] == 0.0
    assert fc["p_wilcoxon"] == pytest.approx(2 / 64, abs=1e-4)                    # 6 paired cases, all improved
    assert fc["reduction_ci_lo"] <= fc["reduction"] <= fc["reduction_ci_hi"]
    assert len(res["records"]) == 6 * 6 * 2


def test_ablation_aggregates_cost_and_latency_per_case():
    res = ablation.run_ablation(CASES, fake_runner)
    assert res["summary"]["full"]["cost_usd"]["mean"] == pytest.approx(0.01)
    assert res["summary"]["all_off"]["cost_usd"]["mean"] == pytest.approx(0.03)
    assert res["summary"]["full"]["latency_s"]["mean"] == pytest.approx(1.0)
    assert res["summary"]["no_fact_checker"]["latency_s"]["mean"] == pytest.approx(1.1)


def test_ablation_counts_crashes_and_fallbacks_as_undelivered():
    def runner(case, ablate):
        if "failover" in ablate:
            raise RuntimeError("provider down")
        if "research" in ablate:
            return {"is_fallback": True, "cover_letter": ""}
        return fake_runner(case, ablate)
    res = ablation.run_ablation(CASES[:3], runner)
    assert res["summary"]["no_failover"]["delivery_rate"] == 0.0
    assert res["summary"]["no_research"]["delivery_rate"] == 0.0
    assert res["summary"]["full"]["delivery_rate"] == 1.0
    assert any(r["error"] == "RuntimeError" for r in res["records"])


def test_ablation_supports_async_runners():
    async def runner(case, ablate):
        await asyncio.sleep(0)
        return fake_runner(case, ablate)
    res = ablation.run_ablation(CASES[:2], runner, ablations=[frozenset(), frozenset({"fact_checker"})])
    assert res["fact_checker_effect"]["n_cases"] == 2


def test_ablation_without_baseline_pair_says_so():
    res = ablation.run_ablation(CASES[:2], fake_runner, ablations=[frozenset({"research"})])
    assert res["fact_checker_effect"]["n_cases"] == 0


def test_ablation_judge_win_rate_full_vs_ablated():
    class Judge:
        model_id = "judge-x"

        def invoke(self, prompt):
            a = prompt.split("<letter_A>")[1].split("</letter_A>")[0]
            b = prompt.split("<letter_B>")[1].split("</letter_B>")[0]
            pick = "A" if ("Kubernetes" not in a) and ("Kubernetes" in b) else "B" if "Kubernetes" in a else "tie"
            return type("R", (), {"content": json.dumps({"winner": pick, "reason": "honest"})})()
    res = ablation.run_ablation(CASES[:4], fake_runner, judge_llm=Judge(), generator_ids=["openai:gpt-4o-mini"])
    assert res["judge"]["no_fact_checker"]["win_rate_x"] == 1.0
    assert res["judge"]["no_fact_checker"]["consistency_rate"] == 1.0
    assert "no_research" in res["judge"]  # identical letters tie


def test_ablation_report_has_no_letter_text():
    res = ablation.run_ablation(CASES[:2], fake_runner)
    assert "Kubernetes migrations" not in json.dumps(res)  # letters derive from resumes; never persisted


def test_ablation_table_warns_on_few_cases():
    text = ablation.format_table(ablation.run_ablation(CASES[:3], fake_runner))
    assert "Fact Checker: hallucination 0.5 -> 0.0" in text and "fewer than 20 cases" in text


# ----------------------------------------------------------------- adversarial corpus
def test_corpus_is_large_unique_and_well_formed():
    inj = injection_cases()
    assert len(inj) >= 25 and len({c["id"] for c in inj}) == len(inj)
    assert len({c["family"] for c in inj}) >= 12
    assert all(c["payload"] in c["jd"] and c["marker"] for c in inj)
    assert len(company_injections()) >= 10
    corpus = load_corpus()
    assert set(corpus) == {"injections", "company_injections", "stuffing"}
    json.dumps(corpus)  # JSON-serialisable


def test_corpus_covers_required_attack_types():
    fams = {c["family"] for c in injection_cases()}
    assert {"closing_tag_spoof", "bidi", "role_play", "exfiltration", "zero_width", "json_break"} <= fams


def test_stuffing_cases_demand_skills_resume_lacks():
    for c in stuffing_cases():
        low = c["resume_text"].lower()
        assert c["forbidden_skills"] and not any(s in low for s in c["forbidden_skills"]), c["id"]


class Recorder(ScriptedModel):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.prompts = []

    def invoke(self, prompt, *a, **k):
        self.prompts.append(str(prompt))
        return super().invoke(prompt, *a, **k)


def _run_graph(gen, crit, jd):
    from graph import build_graph
    from resilience import start_heal_log

    async def go():
        start_heal_log()
        return await build_graph(gen, crit).ainvoke({"job_description": jd, "company_name": "Acme"})
    return asyncio.run(go())


@pytest.mark.parametrize("case", injection_cases(), ids=lambda c: c["id"])
def test_every_injection_stays_inside_its_fence_and_is_sanitised(case):
    rec = Recorder()
    with stub_backends():
        _run_graph(rec, rec, case["jd"])
    ingest = next(p for p in rec.prompts if "processing a job description" in p)
    assert ingest.count("</job_description>") == 1                      # attacker cannot close the fence
    assert "UNTRUSTED" in ingest
    # attacker-supplied control / bidi / zero-width characters are stripped; the only zero-width character
    # allowed is the one the fence itself inserts to defuse a spoofed closing tag ("<" + ZWSP + "/").
    stripped = ingest.replace("<" + chr(0x200B) + "/", "</")
    banned = [chr(c) for c in (0x00, 0x1B, 0x202E, 0x202C, 0x200B)]
    assert not any(ch in stripped for ch in banned)


@pytest.mark.parametrize("case", company_injections(), ids=lambda c: c["id"])
def test_company_research_injections_cannot_close_their_fence(case):
    rec = Recorder()
    with stub_backends(research_text=case["company_research"]):
        _run_graph(rec, rec, JD)
    writer = next(p for p in rec.prompts if "Write a factual cover letter" in p)
    assert writer.count("</company_research>") == 1 and "untrusted" in writer.lower()


@pytest.mark.parametrize("case", stuffing_cases(), ids=lambda c: c["id"])
def test_fact_checker_neutralises_keyword_stuffing(case):
    """A writer that caves and claims every demanded skill must not get those claims through."""
    skills = case["forbidden_skills"]
    greedy = ". ".join(f"I have deep production experience with {s}" for s in skills) + "."
    resume_chunk = [{"text": case["resume_text"], "source": "r.pdf", "score": 0.9, "chunk_id": "r.pdf#0"}]
    writer = ScriptedModel(letter=greedy)
    before = score_letter(greedy, case["resume_text"], skills)["hallucination_rate"]
    assert before > 0
    with stub_backends(keywords=tuple(skills), chunks=resume_chunk):
        out = _run_graph(writer, ScriptedModel(), case["jd"])
    after = score_letter(out["cover_letter"], case["resume_text"], skills)
    assert after["hallucination_rate"] == 0.0 and after["unsupported_claims"] == []
