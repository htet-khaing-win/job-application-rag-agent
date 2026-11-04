import json

import pytest

import resilience
import runtime
from cost import BudgetExceeded
from resilience import (
    AllCandidatesFailed,
    ChaosModel,
    ResilientLLM,
    _breaker_open,
    drain_usage,
    reset_breakers,
    set_token_sink,
    start_heal_log,
    start_usage_log,
)
from runtime import DEFAULT_SETTINGS, PRESETS, ROLES, Binding, Profile, Runtime, preset_profile
from vault import KeyVault


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(resilience, "BACKOFF_BASE_S", 0)
    reset_breakers()
    yield
    set_token_sink("writer", None)


class Msg:
    def __init__(self, content, usage=None):
        self.content = content
        self.usage_metadata = usage

    def __add__(self, other):
        return Msg(self.content + other.content, other.usage_metadata or self.usage_metadata)


class Good:
    def __init__(self, text="ok", usage=None):
        self.text, self.usage = text, usage
        self.calls = 0

    def invoke(self, prompt, *a, **k):
        self.calls += 1
        return Msg(self.text, self.usage)


# --- budget -> failover -------------------------------------------------------
def test_budget_exceeded_fails_over_without_tripping_breaker():
    primary, backup = Good("primary"), Good("backup")

    def precheck(label):
        if label == "a":
            raise BudgetExceeded("Monthly budget reached for key 'a'")

    llm = ResilientLLM("writer", [("a", lambda: primary), ("b", lambda: backup)], precheck=precheck)
    log = start_heal_log()
    for _ in range(5):
        assert llm.invoke("hi").content == "backup"
    assert primary.calls == 0 and backup.calls == 5
    assert not _breaker_open("a")                      # budget is not a provider outage
    assert any("over budget" in e for e in log)


def test_budget_exceeded_on_every_candidate_raises_cleanly():
    def precheck(label):
        raise BudgetExceeded("budget gone")

    llm = ResilientLLM("writer", [("a", lambda: Good()), ("b", lambda: Good())], precheck=precheck)
    with pytest.raises(AllCandidatesFailed, match="budget"):
        llm.invoke("hi")


def test_usage_cb_runs_and_its_failure_never_breaks_the_call():
    seen = []
    llm = ResilientLLM("writer", [("a", lambda: Good("x", {"input_tokens": 3}))],
                       usage_cb=lambda label, r: seen.append((label, r.content)))
    assert llm.invoke("p").content == "x" and seen == [("a", "x")]

    def boom(label, r):
        raise RuntimeError("accounting exploded")

    assert ResilientLLM("writer", [("a", lambda: Good("y"))], usage_cb=boom).invoke("p").content == "y"


def test_usage_log_roundtrip():
    log = start_usage_log()
    resilience.record_usage({"role": "writer", "cost_usd": 0.1})
    assert log and drain_usage() == [{"role": "writer", "cost_usd": 0.1}] and drain_usage() == []


# --- streaming -----------------------------------------------------------------
class Streamer:
    def __init__(self, chunks, fail_after=None, error="401 unauthorized"):
        self.chunks, self.fail_after, self.error = chunks, fail_after, error

    def stream(self, prompt, *a, **k):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError(self.error)
            yield Msg(c)

    def invoke(self, prompt, *a, **k):
        raise AssertionError("invoke must not be used when a sink is registered")


def test_streaming_calls_sink_per_chunk_and_returns_accumulated_message():
    got = []
    set_token_sink("writer", got.append)
    out = ResilientLLM("writer", [("a", lambda: Streamer(["Hel", "lo ", "world"]))]).invoke("p")
    assert got == ["Hel", "lo ", "world"] and out.content == "Hello world"


def test_stream_failure_mid_way_resets_sink_then_fails_over():
    got = []
    set_token_sink("writer", got.append)
    llm = ResilientLLM("writer", [("a", lambda: Streamer(["Hel", "lo"], fail_after=1)),
                                  ("b", lambda: Streamer(["Hi ", "there"]))])
    out = llm.invoke("p")
    assert got == ["Hel", None, "Hi ", "there"]        # partial text discarded before the retry
    assert out.content == "Hi there"


def test_no_reset_signal_when_failure_precedes_any_text():
    got = []
    set_token_sink("writer", got.append)
    llm = ResilientLLM("writer", [("a", lambda: Streamer(["x"], fail_after=0)), ("b", lambda: Streamer(["ok"]))])
    assert llm.invoke("p").content == "ok" and got == ["ok"]


def test_sink_only_applies_to_its_role_and_to_models_that_stream():
    got = []
    set_token_sink("writer", got.append)
    assert ResilientLLM("critic", [("a", lambda: Good("c"))]).invoke("p").content == "c"   # other role
    assert ResilientLLM("writer", [("a", lambda: Good("w"))]).invoke("p").content == "w"   # no .stream
    assert got == []
    set_token_sink("writer", None)
    assert ResilientLLM("writer", [("a", lambda: Good("z"))]).invoke("p").content == "z"


# --- chaos ------------------------------------------------------------------------
def _trace(model, n=60):
    out = []
    for _ in range(n):
        try:
            out.append(("ok", model.invoke("p").content))
        except Exception as e:  # noqa: BLE001
            out.append((type(e).__name__, str(e)))
    return out


def test_chaos_is_deterministic_for_a_seed_and_differs_across_seeds():
    a = _trace(ChaosModel(Good("fine"), 0.5, seed=7))
    b = _trace(ChaosModel(Good("fine"), 0.5, seed=7))
    c = _trace(ChaosModel(Good("fine"), 0.5, seed=8))
    assert a == b and a != c


def test_chaos_probability_bounds_and_all_fault_kinds_appear():
    assert all(r == ("ok", "fine") for r in _trace(ChaosModel(Good("fine"), 0.0, seed=1)))
    seen = _trace(ChaosModel(Good("fine"), 1.0, seed=1), n=300)
    assert all(r != ("ok", "fine") for r in seen)
    text = " ".join(f"{k} {v}" for k, v in seen)
    for marker in ("TimeoutError", "429", "503", "401", "{{broken json"):
        assert marker in text


def test_chaos_stream_faults_and_passthrough():
    assert [c.content for c in ChaosModel(Streamer(["a", "b"]), 0.0, seed=1).stream("p")] == ["a", "b"]
    assert [c.content for c in ChaosModel(Good("one"), 0.0, seed=1).stream("p")] == ["one"]   # invoke fallback
    raised, broken = 0, 0
    for seed in range(60):
        try:
            chunks = list(ChaosModel(Good("fine"), 1.0, seed=seed).stream("p"))
            assert [c.content for c in chunks] == ["{{broken json"]
            broken += 1
        except (RuntimeError, TimeoutError):
            raised += 1
    assert raised > 0 and broken > 0 and raised + broken == 60   # p=1 never returns the real answer


def test_resilient_llm_heals_a_chaotic_model():
    llm = ResilientLLM("writer", [("a", lambda: ChaosModel(Good("done"), 0.5, seed=11)),
                                  ("b", lambda: Good("done"))])
    for _ in range(20):
        assert llm.invoke("p").content in ("done", "{{broken json")  # broken JSON is invoke_json's job


# --- runtime integration ---------------------------------------------------------------
@pytest.fixture
def rt(tmp_path, monkeypatch):
    vault = KeyVault(tmp_path, use_keyring=False)
    vault.set("k", "openai", "sk-key-for-openai-0000", monthly_cap_usd=0.001)
    vault.set("g", "gemini", "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE1")
    prof = Profile()
    prof.roles["writer"] = [Binding("openai", "gpt-4o-mini", "k"), Binding("gemini", "gemini-2.5-flash", "g")]
    monkeypatch.setattr(runtime, "build_chat_model",
                        lambda *a, **k: Good("letter", {"input_tokens": 1000, "output_tokens": 1000}))
    return Runtime(vault, prof)


def test_runtime_records_usage_and_budget_cap_fails_over_to_next_key(rt):
    log = start_usage_log()
    for _ in range(3):
        assert rt.chat("writer").invoke("p").content == "letter"
    entries = drain_usage()
    assert [e["alias"] for e in entries] == ["k", "k", "g"]    # 3rd call blocked by k's $0.001 cap
    assert entries[0]["cost_usd"] == pytest.approx(0.00075) and entries[0]["input_tokens"] == 1000
    assert entries[2]["role"] == "writer" and entries[2]["label"] == "gemini:gemini-2.5-flash"
    assert log == []                                            # drained entries leave the log empty


def test_runtime_ledger_failure_never_crashes_a_call(rt, monkeypatch):
    monkeypatch.setattr("cost.SpendLedger.add", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert rt.chat("writer").invoke("p").content == "letter"


def test_failover_disabled_keeps_only_primary(rt):
    assert len(rt.chat("writer").candidates) == 2
    rt.failover_enabled = False
    llm = rt.chat("writer")
    assert [label for label, _ in llm.candidates] == ["openai:gpt-4o-mini"]


def test_chaos_wraps_models_only_when_enabled(rt):
    assert not isinstance(rt.chat("writer").candidates[0][1](), ChaosModel)
    rt.chaos, rt.chaos_seed = 0.3, 5
    assert isinstance(rt.chat("writer").candidates[0][1](), ChaosModel)


def test_missing_keys_reports_locked_vault(tmp_path):
    v = KeyVault(tmp_path, use_keyring=False, passphrase="long enough pass")
    v.lock()
    assert "locked" in Runtime(v, Profile()).missing_keys()[0].lower()


# --- profile settings -----------------------------------------------------------------------
def test_settings_default_validate_and_roundtrip(tmp_path):
    p = Profile()
    assert p.settings == DEFAULT_SETTINGS and p.settings is not DEFAULT_SETTINGS
    p.settings.update(alpha=0.7, max_resumes=3, top_k=20)
    p.save(tmp_path / "p.json")
    loaded = Profile.load(tmp_path / "p.json")
    assert loaded.settings == {"alpha": 0.7, "max_resumes": 3, "top_k": 20}
    assert Runtime(KeyVault(tmp_path, use_keyring=False), loaded).settings["alpha"] == 0.7


@pytest.mark.parametrize("bad", [{"alpha": 1.5}, {"alpha": -0.1}, {"alpha": "0.4"}, {"alpha": True},
                                 {"max_resumes": 0}, {"max_resumes": 51}, {"max_resumes": 2.5},
                                 {"top_k": True}, {"top_k": 0}, {"nope": 1}])
def test_settings_validation_rejects(bad):
    p = Profile()
    p.settings.update(bad)
    with pytest.raises(ValueError):
        p.validate()


def test_old_flat_profile_json_still_loads_with_default_settings():
    legacy = json.dumps({"writer": [{"provider": "anthropic", "model": "claude-sonnet-5-5",
                                     "key_alias": "mine", "temperature": 0.0}]})
    prof = Profile.from_json(legacy)
    assert prof.roles["writer"][0].key_alias == "mine" and prof.settings == DEFAULT_SETTINGS
    assert "judge" in prof.roles                       # missing roles get defaults


def test_new_json_shape_with_partial_settings_merges_defaults():
    text = json.dumps({"roles": {}, "settings": {"alpha": 0.9, "junk": 1}})
    prof = Profile.from_json(text)
    assert prof.settings == {"alpha": 0.9, "max_resumes": 10, "top_k": 10}
    assert "roles" in json.loads(prof.to_json()) and "settings" in json.loads(prof.to_json())


def test_corrupt_profile_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "p.json").write_text("[1,2,3]", encoding="utf-8")
    assert Profile.load(tmp_path / "p.json").settings == DEFAULT_SETTINGS


# --- judge role & presets -----------------------------------------------------------------------
def test_judge_role_defaults_and_presets():
    assert ROLES["judge"]["kind"] == "chat" and ROLES["judge"]["label"] == "Judge"
    d = Profile().roles["judge"]
    assert (d[0].provider, d[0].model) == ("anthropic", "claude-opus-5-5")
    assert (d[1].provider, d[1].model) == ("openai", "gpt-4o")
    for name in PRESETS:
        prof = preset_profile(name)
        assert prof.roles["judge"] and prof.settings == DEFAULT_SETTINGS
    assert all(b.provider == "ollama" for b in preset_profile("Fully local (Ollama)").roles["judge"])
    cheap = preset_profile("Cheap (Gemini Flash only)")
    assert cheap.roles["judge"][0].model != cheap.roles["writer"][0].model   # judge differs from writer
