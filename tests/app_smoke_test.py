"""Headless GUI tests with streamlit.testing.v1.AppTest (no browser, no network, no real keys)."""
import pathlib
import tomllib

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import cost  # noqa: E402
import pipeline  # noqa: E402
import runlog  # noqa: E402
import runtime  # noqa: E402
import vault as vault_mod  # noqa: E402
from history import PAGE_SIZE, HistoryStore  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")

CHAT, HISTORY = "💬 Chat", "📚 History"
RESUMES, AGENTS = "📄 Resumes", "🤖 Agents & Models"
KEYS, RUNS, SECURITY = "🔑 API Keys", "📊 Runs", "🛡️ Security"
ALL_PAGES = [CHAT, HISTORY, RESUMES, AGENTS, KEYS, RUNS, SECURITY]

JD = ("Machine Learning Engineer. Requirements: Python, PyTorch and Kubernetes experience. Responsibilities: build "
      "and operate model-serving backends, collaborate with product and mentor junior engineers on the team.")


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


def _run(page=None, **state):
    at = AppTest.from_file(APP, default_timeout=60)
    if page:
        at.session_state["nav"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def _go(at, page):
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception
    return at


def _click(at, key):
    """Click a button whose handler calls st.rerun(), then settle (AppTest otherwise keeps the interrupted tree)."""
    next(b for b in at.button if b.key == key).click().run()
    at.run()
    assert not at.exception
    return at


def _entry(**over):
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
    return {"company": "Acme", "jd": JD, "result": base}


def _with_result(**over):
    entry = _entry(**over)
    msgs = [{"role": "assistant", "kind": "text", "content": "hi"},
            {"role": "user", "kind": "jd", "content": JD},
            {"role": "assistant", "kind": "result", "content": None, "entry": entry}]
    return {"messages": msgs, "last_result": entry, "draft_edit": entry["result"]["cover_letter"]}


def _no_missing_keys(monkeypatch):
    monkeypatch.setattr(runtime.Runtime, "missing_keys", lambda self: [])


def _fake_pipeline(monkeypatch, **over):
    async def fake_run(rt, jd, company, **kw):
        if kw.get("token_sink"):
            kw["token_sink"]("Dear ")
            kw["token_sink"]("team")
        if kw.get("on_event"):
            kw["on_event"]({"node": "Copywriter", "ms": 5})
        return _entry(**over)["result"]

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)


# ---------------------------------------------------------------- layout / theme
def test_every_page_renders_and_navigation_lists_all_pages():
    at = _run()
    assert not at.exception
    assert list(at.sidebar.radio(key="nav").options) == ALL_PAGES
    for page in ALL_PAGES:
        _go(at, page)


def test_light_colourful_theme_is_configured():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["theme"]["base"] == "light" and cfg["theme"]["primaryColor"].startswith("#")
    assert cfg["server"]["address"] == "127.0.0.1" and cfg["server"]["enableXsrfProtection"] is True


def test_chat_greets_and_invites_a_job_description():
    at = _run()
    assert any("cover-letter studio" in m.value for m in at.markdown)
    assert at.chat_input and "job description" in at.chat_input[0].proto.placeholder.lower()


def test_missing_keys_warn_and_disable_the_chat_box():
    at = _run()
    assert any("Missing keys" in w.value for w in at.warning)
    assert at.chat_input[0].proto.disabled


def test_chaos_toggle_lives_in_sidebar_and_defaults_off():
    at = _run()
    assert [t.key for t in at.sidebar.toggle] == ["chaos_on"]
    assert at.sidebar.toggle[0].value is False
    assert not any("Chaos mode is ON" in w.value for w in at.warning)
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("Chaos mode is ON" in w.value for w in at.warning)


# ---------------------------------------------------------------- chat flow
def test_chat_flow_jd_then_company_produces_a_letter(monkeypatch):
    _no_missing_keys(monkeypatch)
    _fake_pipeline(monkeypatch)
    at = _run()
    at.chat_input[0].set_value(JD).run()
    assert not at.exception
    assert any("Which company" in m.value for m in at.markdown)
    assert "company" in at.chat_input[0].proto.placeholder.lower()
    at.chat_input[0].set_value("Acme").run()
    assert not at.exception
    assert any("Here's your cover letter" in m.value for m in at.markdown)
    assert any("I built python services" in t.value for t in at.text)
    assert "job description" in at.chat_input[0].proto.placeholder.lower()          # ready for the next one


def test_too_short_job_description_is_politely_rejected(monkeypatch):
    _no_missing_keys(monkeypatch)
    at = _run()
    at.chat_input[0].set_value("hiring dev").run()
    assert any("a bit short" in m.value for m in at.markdown)
    assert "job description" in at.chat_input[0].proto.placeholder.lower()          # still waiting for the JD


def test_failed_run_shows_a_friendly_error_not_a_traceback(monkeypatch):
    _no_missing_keys(monkeypatch)

    async def boom(*a, **k):
        raise RuntimeError("provider exploded with key sk-LEAKEDKEYabcdefghijklmnop")

    monkeypatch.setattr(pipeline, "run_pipeline", boom)
    at = _run()
    at.chat_input[0].set_value(JD).run()
    at.chat_input[0].set_value("Acme").run()
    assert not at.exception
    blob = "\n".join(_texts(at.markdown) + _texts(at.error))
    assert "didn't work" in blob or "provider exploded" in blob
    assert "sk-LEAKEDKEY" not in blob


# ---------------------------------------------------------------- result bubble
def test_result_renders_chips_editable_draft_and_skills_gap():
    at = _run(**_with_result())
    assert not at.exception
    area = next(t for t in at.text_area if t.key == "draft_edit")
    assert "Kubernetes" in area.value
    md = "\n".join(_texts(at.markdown))
    assert "✅ verified" in md and "⚠️ no resume evidence" in md and "Copywriter" in md
    assert "match 88.0" in md and "grounded" in md


def test_recheck_grounding_flags_unsupported_claims_in_edited_draft():
    at = _run(**{**_with_result(), "draft_edit": "I built python services."})
    next(b for b in at.button if b.key == "recheck").click().run()
    assert any("No ungrounded" in s.value for s in at.success)

    at.text_area(key="draft_edit").set_value("I have deep Kubernetes expertise. I built python services.").run()
    next(b for b in at.button if b.key == "recheck").click().run()
    assert any("claim skills with no resume evidence" in w.value for w in at.warning)
    assert any("Kubernetes expertise" in c.value for c in at.code)


def test_unsafe_resume_text_cannot_inject_markdown_into_tables():
    evil = "[click me](http://evil.example) | **x** <script>"
    state = _with_result(verified_skills_detailed={"with_evidence": [{"skill": "python", "evidence": evil}],
                                                   "no_evidence": []})
    at = _run(**state)
    assert not at.exception
    md = "\n".join(_texts(at.markdown))
    assert "[click me](http://evil.example)" not in md and "<script>" not in md


def test_new_letter_button_resets_the_conversation():
    at = _run(**_with_result())
    next(b for b in at.button if b.key == "new-letter").click().run()
    assert at.session_state["chat_stage"] == "await_jd" and len(at.session_state["messages"]) == 1
    at.run()                                      # AppTest shows the pre-rerun tree for a click that calls st.rerun()
    assert not at.exception
    assert not any("Here's your cover letter" in m.value for m in at.markdown)
    assert at.session_state["chat_stage"] == "await_jd"


# ---------------------------------------------------------------- saving to history
def test_satisfied_button_saves_the_edited_draft_once():
    at = _run(**_with_result())
    at.text_area(key="draft_edit").set_value("My edited and final letter.").run()
    next(b for b in at.button if b.key == "save-hist").click().run()
    assert not at.exception
    store = HistoryStore()
    entries = store.list()
    assert len(entries) == 1 and entries[0].letter == "My edited and final letter."
    assert entries[0].company == "Acme" and entries[0].meta["verified"] is True
    assert any("Saved to history" in s.value for s in at.success)
    assert not any(b.key == "save-hist" for b in at.button)                     # cannot double-save by clicking again
    assert store.count() == 1


def test_job_title_guess_handles_one_long_pasted_paragraph():
    import re
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    ns = {"re": re, "sanitize_text": lambda t, n=0: t}
    exec(src[src.index("def _guess_title"):src.index("def _pretty_date")], ns)   # noqa: S102 - our own source
    guess = ns["_guess_title"]
    assert guess(JD) == "Machine Learning Engineer"
    assert guess("Senior AI Engineer\nBuild things") == "Senior AI Engineer"
    assert guess("") == "Untitled role"


def test_nothing_is_saved_until_the_user_says_they_are_satisfied():
    _run(**_with_result())
    assert HistoryStore().count() == 0


# ---------------------------------------------------------------- history page
def _seed(n, company="Acme"):
    store = HistoryStore()
    for i in range(n):
        store.add(f"{company} {i:02d}", f"Role {i}", f"JD text {i}", f"Letter body number {i}", {"verified": True})
    return store


def test_empty_history_explains_how_to_save():
    at = _run(HISTORY)
    assert not at.exception
    assert any("Nothing saved yet" in i.value for i in at.info)


def test_history_shows_ten_per_page_newest_first_and_pages_through():
    _seed(25)
    at = _run(HISTORY)
    assert not at.exception
    assert len([e for e in at.expander if e.label == "Open letter"]) == PAGE_SIZE
    cards = [m.value for m in at.markdown if m.value.startswith("<div class='card-title'>")]
    assert "Acme 24" in cards[0] and "Acme 15" in cards[-1]
    assert next(b for b in at.button if b.key == "hist-newer").disabled
    _click(at, "hist-older")
    assert any("Acme 14" in m.value for m in at.markdown if m.value.startswith("<div class='card-title'>"))
    assert len([e for e in at.expander if e.label == "Open letter"]) == PAGE_SIZE
    _click(at, "hist-older")
    assert len([e for e in at.expander if e.label == "Open letter"]) == 5
    assert next(b for b in at.button if b.key == "hist-older").disabled


def test_delete_needs_confirmation_and_then_erases_permanently(tmp_path):
    store = HistoryStore()
    keep = store.add("Keep Co", "Dev", "jd keep", "KEEP-MARKER letter text")
    gone = store.add("Gone Co", "Dev", "jd gone", "ERASE-MARKER-7f3a letter text")
    at = _run(HISTORY)
    at.button(key=f"hdel-{gone}").click().run()
    assert any("permanently erases" in w.value for w in at.warning)
    assert store.count() == 2                                                   # nothing deleted by the first click
    at.button(key=f"hno-{gone}").click().run()
    assert store.count() == 2
    at.button(key=f"hdel-{gone}").click().run()
    at.button(key=f"hyes-{gone}").click().run()
    assert not at.exception
    assert store.get(gone) is None and store.get(keep) is not None
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("history.db*"))
    assert b"ERASE-MARKER-7f3a" not in raw and b"KEEP-MARKER" in raw


def test_erase_all_requires_typing_delete():
    _seed(3)
    at = _run(HISTORY)
    wipe = next(b for b in at.button if b.key == "wipe-all")
    assert wipe.disabled
    at.text_input(key="wipe-word").set_value("delete").run()
    assert next(b for b in at.button if b.key == "wipe-all").disabled          # case-sensitive on purpose
    at.text_input(key="wipe-word").set_value("DELETE").run()
    next(b for b in at.button if b.key == "wipe-all").click().run()
    assert not at.exception and HistoryStore().count() == 0


def test_history_page_deleting_the_last_item_on_a_page_moves_back():
    store = _seed(11)
    at = _run(HISTORY, hist_page=1)
    oldest = store.list(limit=1, offset=10)[0].id
    at.button(key=f"hdel-{oldest}").click().run()
    at.button(key=f"hyes-{oldest}").click().run()
    assert not at.exception
    assert at.session_state["hist_page"] == 0 and len([e for e in at.expander if e.label == "Open letter"]) == PAGE_SIZE


def test_history_diff_against_previous_letter_for_same_company():
    store = HistoryStore()
    store.add("Acme", "ML", "jd", "Line one.\nLine two OLD.")
    store.add("Acme", "ML", "jd", "Line one.\nLine two NEW.")
    at = _run(HISTORY)
    diffs = [c.value for c in at.code if c.language == "diff"]
    assert diffs and "-Line two OLD." in diffs[0] and "+Line two NEW." in diffs[0]


def test_history_cannot_inject_html_through_company_or_letter():
    HistoryStore().add("<img src=x onerror=alert(1)>", "<b>Role</b>", "jd <script>x</script>",
                       "letter <script>alert(1)</script>")
    at = _run(HISTORY)
    assert not at.exception
    md = "\n".join(_texts(at.markdown))
    assert "<img src=x" not in md and "<script>" not in md and "&lt;img src=x" in md


# ---------------------------------------------------------------- other pages
def test_connection_lights_appear_after_test_all():
    at = _run(AGENTS)
    assert not any("🔴" in e for e in _texts(at.caption))
    next(b for b in at.button if b.label == "Test all connections").click().run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any(lab.startswith("🔴") for lab in labels)            # no keys -> every agent reports failure, not a crash
    assert not any(lab.startswith("⚪") for lab in labels if "Judge" in lab or "Copywriter" in lab)


def test_single_agent_test_button_updates_that_agent_only():
    at = _run(AGENTS)
    at.button(key="test-writer").click().run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    writer = next(lab for lab in labels if "Copywriter" in lab)
    critic = next(lab for lab in labels if "Hiring Manager" in lab)
    assert writer.startswith("🔴") and critic.startswith("⚪")


def test_profile_compare_shows_only_differences():
    at = _run(AGENTS)
    assert not at.exception
    at.selectbox(key="cmp").set_value("Fully local (Ollama)").run()
    assert not at.exception
    assert any("ollama" in m.value for m in at.markdown)


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
    assert not at.chat_input and not at.sidebar.radio                       # nothing else renders while locked
    assert any("Vault locked" in t.value for t in at.title)
    assert [t.proto.type for t in at.text_input] == [1]   # TextInput.Type.PASSWORD
    at.text_input[0].set_value("wrong-passphrase")
    next(b for b in at.button if b.label == "Unlock").click().run()
    assert any("Could not unlock" in e.value for e in at.error)
    assert not any("secret detail" in e.value for e in at.error)


def test_security_page_has_passphrase_cap_and_settings_controls():
    at = _run(SECURITY)
    assert not at.exception
    sub = _texts(at.subheader)
    for expected in ("Vault passphrase", "Key age, last use and rotation", "Retrieval settings"):
        assert expected in sub


def test_saving_settings_persists_to_profile(tmp_path):
    at = _run(SECURITY)
    sliders = [s for s in at.slider if "alpha" in s.label.lower()]
    assert sliders
    sliders[0].set_value(0.7)
    next(b for b in at.button if b.label == "Save settings").click().run()
    assert not at.exception
    saved = (tmp_path / "profile.json").read_text(encoding="utf-8")
    assert '"alpha": 0.7' in saved


def test_runs_page_reports_percentiles_without_leaking_letter_text(monkeypatch):
    runs = [{"run_id": f"r{i}", "ok": True, "duration_ms": 100.0 * (i + 1), "cost_usd": 0.01, "tokens_in": 10,
             "tokens_out": 5, "healing_events": 0, "letter_verified": True, "cover_letter": "SECRET LETTER TEXT",
             "node_timings": [{"node": "Copywriter", "ms": 10.0 * (i + 1)}]} for i in range(5)]
    monkeypatch.setattr(runlog, "read_runs", lambda *a, **k: runs)
    at = _run(RUNS)
    assert not at.exception
    md = "\n".join(_texts(at.markdown))
    assert "Copywriter" in md and "p50 ms" in md and "r4" in md
    assert "SECRET LETTER TEXT" not in md
    assert "SECRET LETTER TEXT" not in "".join(str(j.value) for j in at.json)
