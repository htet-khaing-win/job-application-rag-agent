"""Headless GUI smoke tests with streamlit.testing.v1.AppTest (no browser, no network, no real keys)."""
import pathlib

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import cost  # noqa: E402
import runlog  # noqa: E402
import runtime  # noqa: E402
import vault as vault_mod  # noqa: E402

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    """Keep the GUI away from the real home directory and the OS keychain."""
    import keyring
    import streamlit as st

    def no_keyring(*a, **k):
        raise RuntimeError("keyring disabled in tests")

    monkeypatch.setattr(keyring, "get_password", no_keyring)
    monkeypatch.setattr(keyring, "set_password", no_keyring)
    monkeypatch.setenv("JOBAGENT_HOME", str(tmp_path))
    for mod in (vault_mod, runtime, runlog, cost):
        if hasattr(mod, "APP_DIR"):
            monkeypatch.setattr(mod, "APP_DIR", tmp_path)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _texts(elements):
    return [getattr(e, "value", "") or "" for e in elements]


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _result(**over):
    base = {
        "cover_letter": "I built python services. I led Kubernetes migrations.",
        "unverified_skills": ["kubernetes"], "verified_skills": ["python"],
        "verified_skills_detailed": {"with_evidence": [{"skill": "python", "evidence": "Built python services"}],
                                     "no_evidence": [{"skill": "kubernetes"}]},
        "letter_verified": True, "refinement_count": 0, "vector_relevance_score": 88.0,
        "healing_log": ["[writer] recovered on b (attempt 2)"], "node_timings": [{"node": "Copywriter", "ms": 12.5}],
        "cost_usd": 0.0123, "run_id": "abc123def456", "is_fallback": False,
    }
    base.update(over)
    return {"company": "Acme", "result": base}


def test_renders_all_tabs_without_exceptions():
    at = _run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Generate", "Resumes", "Agents & Models", "API Keys", "Runs", "Security"]


def test_missing_keys_block_generation_and_show_cost_estimate():
    at = _run()
    assert any("Missing keys" in w.value for w in at.warning)
    gen = next(b for b in at.button if b.label == "Generate cover letter")
    assert gen.disabled
    assert [m.label for m in at.metric][:2] == ["Estimated cost", "Estimated tokens"]


def test_chaos_toggle_lives_in_sidebar_and_defaults_off():
    at = _run()
    assert [t.key for t in at.sidebar.toggle] == ["chaos_on"]
    assert at.sidebar.toggle[0].value is False
    assert not any("Chaos mode is ON" in w.value for w in at.warning)
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("Chaos mode is ON" in w.value for w in at.warning)


def test_connection_lights_appear_after_test_all():
    at = _run()
    assert not any("🔴" in e for e in _texts(at.caption))
    next(b for b in at.button if b.label == "Test all connections").click().run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any(lab.startswith("🔴") for lab in labels)            # no keys -> every agent reports failure, not a crash
    assert not any(lab.startswith("⚪") for lab in labels if "Judge" in lab or "Copywriter" in lab)


def test_single_agent_test_button_updates_that_agent_only():
    at = _run()
    at.button(key="test-writer").click().run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    writer = next(lab for lab in labels if "Copywriter" in lab)
    critic = next(lab for lab in labels if "Hiring Manager" in lab)
    assert writer.startswith("🔴") and critic.startswith("⚪")


def test_profile_compare_shows_only_differences():
    at = _run()
    assert not at.exception
    # default vs the 'Balanced' preset are identical; switching to another preset must list differences
    at.selectbox(key="cmp").set_value("Fully local (Ollama)").run()
    assert not at.exception
    assert any("ollama" in m.value for m in at.markdown)


def test_result_panel_renders_editable_draft_skills_gap_and_timing():
    at = _run(last_result=_result(), draft_edit="I built python services. I led Kubernetes migrations.")
    assert not at.exception
    area = next(t for t in at.text_area if t.key == "draft_edit")
    assert "Kubernetes" in area.value
    md = "\n".join(_texts(at.markdown))
    assert "✅ verified" in md and "⚠️ no resume evidence" in md and "Copywriter" in md
    assert [m.label for m in at.metric if m.label == "Cost (USD)"]


def test_recheck_grounding_flags_unsupported_claims_in_edited_draft():
    at = _run(last_result=_result(), draft_edit="I built python services.")
    next(b for b in at.button if b.key == "recheck").click().run()
    assert any("No ungrounded" in s.value for s in at.success)

    at.text_area(key="draft_edit").set_value("I have deep Kubernetes expertise. I built python services.").run()
    next(b for b in at.button if b.key == "recheck").click().run()
    assert any("claim skills with no resume evidence" in w.value for w in at.warning)
    assert any("Kubernetes expertise" in c.value for c in at.code)


def test_unsafe_resume_text_cannot_inject_markdown_into_tables():
    evil = "[click me](http://evil.example) | **x** <script>"
    res = _result(verified_skills_detailed={"with_evidence": [{"skill": "python", "evidence": evil}], "no_evidence": []})
    at = _run(last_result=res, draft_edit="I built python services.")
    assert not at.exception
    md = "\n".join(_texts(at.markdown))
    assert "[click me](http://evil.example)" not in md and "<script>" not in md


def test_letter_history_shows_diff_against_previous_version_for_same_company():
    hist = [{"company": "Acme", "letter": "Line one.\nLine two NEW.", "run_id": "r2", "ts": "t2"},
            {"company": "Acme", "letter": "Line one.\nLine two OLD.", "run_id": "r1", "ts": "t1"}]
    at = _run(history=hist)
    assert not at.exception
    diffs = [c.value for c in at.code if c.language == "diff"]
    assert diffs and "-Line two OLD." in diffs[0] and "+Line two NEW." in diffs[0]


def test_locked_vault_shows_only_unlock_form(monkeypatch):
    class LockedVault:
        is_locked = True
        uses_passphrase = True

        def __init__(self, *a, **k):
            self.unlocked = False

        def unlock(self, pw):
            raise ValueError("secret detail that must not be echoed")

    monkeypatch.setattr(vault_mod, "KeyVault", LockedVault)
    at = _run()
    assert not at.exception
    assert not at.tabs                                         # nothing else renders while locked
    assert any("Vault locked" in t.value for t in at.title)
    assert [t.proto.type for t in at.text_input] == [1]   # TextInput.Type.PASSWORD
    at.text_input[0].set_value("wrong-passphrase")
    next(b for b in at.button if b.label == "Unlock").click().run()
    assert any("Could not unlock" in e.value for e in at.error)
    assert not any("secret detail" in e.value for e in at.error)


def test_security_tab_has_passphrase_cap_and_settings_controls():
    at = _run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Test all connections" in labels
    sub = _texts(at.subheader)
    for expected in ("Vault passphrase", "Key age, last use and rotation", "Retrieval settings"):
        assert expected in sub


def test_saving_settings_persists_to_profile(tmp_path):
    at = _run()
    sliders = [s for s in at.slider if "alpha" in s.label.lower()]
    assert sliders
    sliders[0].set_value(0.7)
    next(b for b in at.button if b.label == "Save settings").click().run()
    assert not at.exception
    saved = (tmp_path / "profile.json").read_text(encoding="utf-8")
    assert '"alpha": 0.7' in saved


def test_runs_tab_reports_percentiles_without_leaking_letter_text(monkeypatch):
    runs = [{"run_id": f"r{i}", "ok": True, "duration_ms": 100.0 * (i + 1), "cost_usd": 0.01, "tokens_in": 10,
             "tokens_out": 5, "healing_events": 0, "letter_verified": True, "cover_letter": "SECRET LETTER TEXT",
             "node_timings": [{"node": "Copywriter", "ms": 10.0 * (i + 1)}]} for i in range(5)]
    monkeypatch.setattr(runlog, "read_runs", lambda *a, **k: runs)
    at = _run()
    assert not at.exception
    md = "\n".join(_texts(at.markdown))
    assert "Copywriter" in md and "p50 ms" in md and "r4" in md
    assert "SECRET LETTER TEXT" not in md
    assert "SECRET LETTER TEXT" not in "".join(str(j.value) for j in at.json)
