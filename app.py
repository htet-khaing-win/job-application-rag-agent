"""Streamlit GUI (chat-style).  Run:  streamlit run app.py   (binds to 127.0.0.1 only; see .streamlit/config.toml)

Home is a chat: paste a job description, name the company, and the agents reply with a cover letter you can
edit, re-check, export and (when you're happy) save to a local history of up to 10 letters per page that you
can delete permanently. Everything else (resumes, agents, keys, runs, security) lives behind the sidebar.

Optional modules (cost, runlog, vault passphrase API, PDF export) are probed defensively so the
app still renders if one is absent in a given build.
"""
import asyncio
import difflib
import html
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from export import letter_to_docx, letter_to_pdf
from history import PAGE_SIZE, HistoryError, HistoryStore
from pipeline import run_pipeline
from providers import PROVIDERS, providers_for
from runtime import PRESETS, ROLES, Binding, Profile, Runtime, check_connection, preset_profile, use_runtime
from security import MAX_JD_CHARS, install_log_redaction, redact_secrets, safe_filename, sanitize_text, validate_upload
from subagents import find_ungrounded_sentences
from vault import KeyVault, VaultError

try:  # cost tracking / budgets
    import cost as costmod
except ImportError:  # pragma: no cover
    costmod = None  # type: ignore[assignment]
try:  # structured run log
    import runlog
except ImportError:  # pragma: no cover
    runlog = None  # type: ignore[assignment]

st.set_page_config(page_title="Cover Letter Studio", page_icon="✨", layout="wide")
install_log_redaction()

LIGHT = {"ok": "🟢", "fail": "🔴", "none": "⚪"}
SETTING_DEFAULTS = {"alpha": 0.4, "max_resumes": 10, "top_k": 10}
_PRIVATE_KEY_HINTS = ("letter", "cover", "text", "prompt", "resume", "secret", "key")
MIN_JD_CHARS = 60

PAGE_CHAT, PAGE_HISTORY = "💬 Chat", "📚 History"
PAGE_RESUMES, PAGE_AGENTS = "📄 Resumes", "🤖 Agents & Models"
PAGE_KEYS, PAGE_RUNS, PAGE_SECURITY = "🔑 API Keys", "📊 Runs", "🛡️ Security"
PAGES = [PAGE_CHAT, PAGE_HISTORY, PAGE_RESUMES, PAGE_AGENTS, PAGE_KEYS, PAGE_RUNS, PAGE_SECURITY]

GREETING = ("Hi! I'm your **cover-letter studio** ✨ Paste the **job description** below and I'll write a letter "
            "that only claims skills your resume can back up. When you're happy with one, save it to your history.")

# Static theme CSS. Nothing user-supplied is ever interpolated into HTML: dynamic text goes through
# html.escape() in _chip(), everything else is rendered with st.text / escaped markdown.
_THEME_CSS = """
<style>
:root { --violet:#7C3AED; --pink:#EC4899; --blue:#3B82F6; --mint:#10B981; --amber:#F59E0B; --ink:#1F1B2E; }
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, #E9DDFF 0%, rgba(233,221,255,0) 60%),
    radial-gradient(900px 480px at 100% 0%, #FFD9EC 0%, rgba(255,217,236,0) 55%),
    radial-gradient(800px 500px at 50% 110%, #D5F5EA 0%, rgba(213,245,234,0) 60%),
    #FAF9FF;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(185deg, #4C1D95 0%, #7C3AED 55%, #DB2777 135%); }
[data-testid="stSidebar"] * { color: #F5F3FF; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  background: rgba(255,255,255,.10); border-radius: 12px; padding: .45rem .8rem; transition: all .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.22); transform: translateX(2px); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: #FFFFFF; box-shadow: 0 6px 16px rgba(0,0,0,.18);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * { color: #4C1D95; font-weight: 700; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.30); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(255,255,255,.14); }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.25); }
.brand { font-size: 1.45rem; font-weight: 800; letter-spacing: .2px; margin: .2rem 0 .1rem 0; }
.brand-sub { font-size: .8rem; opacity: .85; margin-bottom: .8rem; }
.hero {
  border-radius: 20px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; color: #fff;
  background: linear-gradient(120deg, #7C3AED 0%, #EC4899 60%, #F59E0B 120%);
  box-shadow: 0 12px 30px rgba(124,58,237,.25);
}
.hero h1 { color:#fff; font-size: 1.6rem; margin: 0; padding: 0; }
.hero p { margin: .25rem 0 0 0; opacity: .95; }
[data-testid="stChatMessage"] {
  background: #FFFFFF; border: 1px solid #EEE9FF; border-radius: 20px; padding: .9rem 1.1rem;
  box-shadow: 0 8px 24px rgba(76,29,149,.08); margin-bottom: .6rem;
}
[data-testid="stChatMessage"]:has(.user-marker) {
  background: linear-gradient(135deg, #7C3AED 0%, #EC4899 100%); border: none;
  box-shadow: 0 10px 26px rgba(124,58,237,.30);
}
[data-testid="stChatMessage"]:has(.user-marker) * { color: #FFFFFF !important; }
[data-testid="stChatMessage"]:has(.user-marker) [data-testid="stExpander"] { background: rgba(255,255,255,.14); border-radius: 12px; }
[data-testid="stChatInput"] { border-radius: 18px; box-shadow: 0 10px 30px rgba(124,58,237,.18); border: 2px solid #C4B5FD; }
.stButton > button, .stDownloadButton > button {
  border-radius: 12px; border: 1px solid #DDD6FE; transition: all .15s ease; font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 14px rgba(124,58,237,.18); border-color: #7C3AED; }
.stButton > button[kind="secondary"] { background: #FFFFFF; color: #4C1D95; }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #7C3AED, #EC4899); color: #fff; border: none;
}
[data-testid="stExpander"] { border-radius: 14px; border: 1px solid #E9E3FF; background: rgba(255,255,255,.7); }
[data-testid="stMetric"] { background: #fff; border-radius: 16px; padding: .6rem .9rem; border: 1px solid #EEE9FF; box-shadow: 0 4px 14px rgba(76,29,149,.06); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
.chip { display:inline-block; padding: .16rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 700; margin: 0 .3rem .25rem 0; }
.chip-violet { background:#EDE9FE; color:#5B21B6; } .chip-green { background:#D1FAE5; color:#047857; }
.chip-pink { background:#FCE7F3; color:#BE185D; }   .chip-blue { background:#DBEAFE; color:#1D4ED8; }
.chip-amber { background:#FEF3C7; color:#B45309; }  .chip-grey { background:#F1F5F9; color:#475569; }
.card-title { font-weight: 800; font-size: 1.05rem; color: #1F1B2E; }
</style>
"""


# ------------------------------------------------------------------ small helpers
def _chip(text: Any, tone: str = "violet") -> str:
    """A coloured pill. The only place dynamic text reaches HTML, so it is always escaped."""
    tone = tone if tone in {"violet", "green", "pink", "blue", "amber", "grey"} else "violet"
    return f'<span class="chip chip-{tone}">{html.escape(str(text))}</span>'


def _scrub(obj: Any) -> Any:
    """Drop anything that could carry letter/resume text or secrets before rendering log data."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if not any(h in str(k).lower() for h in _PRIVATE_KEY_HINTS)}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _cell(v: Any) -> str:
    """Escape a value for a markdown table cell (user/resume text must not inject markup)."""
    text = re.sub(r"\s+", " ", str(v if v is not None else "")).strip()[:240]
    for ch in ("\\", "|", "[", "]", "`", "<", ">", "*", "_"):
        text = text.replace(ch, "\\" + ch)
    return text


def _table(rows: Any, **_ignored: Any) -> None:
    """Render list-of-dicts as a markdown table (avoids st.dataframe's pandas dependency, which is
    fragile in some environments and cannot be made to render untrusted text safely)."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    lines = ["| " + " | ".join(_cell(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(r.get(c, "")) for c in cols) + " |" for r in rows]
    st.markdown("\n".join(lines))


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (q / 100.0) * (len(s) - 1)
    lo, hi = int(pos), min(len(s) - 1, int(pos) + 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - int(pos))


def _node_percentiles(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per: dict[str, list[float]] = {}
    for r in runs:
        for t in (r.get("node_timings") or r.get("timings") or []):
            if isinstance(t, dict) and "node" in t and isinstance(t.get("ms"), (int, float)):
                per.setdefault(str(t["node"]), []).append(float(t["ms"]))
    return [{"Agent": n, "runs": len(v), "p50 ms": round(_percentile(v, 50), 1), "p95 ms": round(_percentile(v, 95), 1)}
            for n, v in sorted(per.items())]


def letter_diff(old: str, new: str) -> str:
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), "previous", "this version", lineterm="", n=1)
    return "\n".join(diff) or "(identical to the previous version)"


def _fmt_bindings(bindings: list[Binding] | None) -> str:
    if not bindings:
        return "—"
    return " → ".join(f"{b.provider}:{b.model}" + (f" [{b.key_alias}]" if b.key_alias else "") for b in bindings)


def _role_usd(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in ("usd", "cost_usd", "total_usd"):
            if isinstance(v.get(k), (int, float)):
                return float(v[k])
    return 0.0


def _mentions(skill: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", text.lower()))


def _guess_title(jd: str) -> str:
    """Best-effort job title: the first short line of the posting (history label only)."""
    for line in sanitize_text(jd, 2000).splitlines():
        line = line.strip(" -•*#\t")
        if len(line) > 90:                       # one long paragraph: use its first clause ("ML Engineer. Requirements: ...")
            line = re.split(r"[.:;|]| - ", line, maxsplit=1)[0].strip(" -•*#\t")
        if 3 <= len(line) <= 90:
            return line
    return "Untitled role"


def _pretty_date(iso: str) -> str:
    try:
        return f"{iso[:10]} {iso[11:16]} UTC"
    except Exception:  # noqa: BLE001
        return iso


@st.cache_resource
def get_vault() -> KeyVault:
    return KeyVault()


st.markdown(_THEME_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------ vault gate
vault = get_vault()
if getattr(vault, "is_locked", False):
    st.title("🔒 Vault locked")
    st.caption("Enter your vault passphrase to continue. Nothing else is available while the vault is locked.")
    with st.form("unlock", clear_on_submit=True):
        _pw = st.text_input("Vault passphrase", type="password", autocomplete="off")
        if st.form_submit_button("Unlock"):
            try:
                vault.unlock(_pw)
                st.rerun()
            except Exception:  # noqa: BLE001 - never echo details on a failed unlock
                st.error("Could not unlock the vault. Check the passphrase.")
    st.stop()

if "profile" not in st.session_state:
    st.session_state.profile = Profile.load()
profile: Profile = st.session_state.profile
st.session_state.setdefault("conn_status", {})
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("messages", [{"role": "assistant", "kind": "text", "content": GREETING}])
st.session_state.setdefault("chat_stage", "await_jd")
st.session_state.setdefault("pending_jd", "")
st.session_state.setdefault("run_request", None)
st.session_state.setdefault("hist_page", 0)
st.session_state.setdefault("confirm_delete", None)
st.session_state.setdefault("timeout_s", 180)


def _settings() -> dict[str, Any]:
    return {**SETTING_DEFAULTS, **dict(getattr(profile, "settings", {}) or {})}


rt = Runtime(vault, profile)

# ------------------------------------------------------------------ sidebar: brand, navigation, run options
with st.sidebar:
    st.markdown('<div class="brand">✨ Cover Letter Studio</div>'
                '<div class="brand-sub">Grounded in your resume · keys stay on this machine</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    with st.expander("⚙️ Run options"):
        st.slider("Timeout (seconds)", 30, 600, 180, 10, key="timeout_s",
                  help="A run is aborted after this long. Streamlit's Stop button (top right while running) also cancels.")
    with st.expander("🧪 Chaos mode (demo)"):
        st.caption("Demo of self-healing: randomly injects provider faults (timeouts, 429s, bad JSON, auth errors) "
                   "into model calls. Leave off for real use; expect slower, sometimes degraded runs. "
                   "Watch the self-healing log after a run.")
        _chaos_on = st.toggle("Enable chaos", key="chaos_on")
        _chaos_p = st.slider("Fault probability", 0.0, 0.9, 0.3, 0.05, key="chaos_p", disabled=not _chaos_on)
try:
    rt.chaos = float(_chaos_p) if _chaos_on else 0.0  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    pass


# ------------------------------------------------------------------ cost / budget
def _resolve_alias(b: Binding) -> str:
    if b.key_alias:
        return b.key_alias
    return next((k["alias"] for k in vault.list() if k["provider"] == b.provider), "")


def _estimate(jd_chars: int) -> dict[str, Any] | None:
    if costmod is None or not hasattr(costmod, "estimate_run_cost"):
        return None
    try:
        return costmod.estimate_run_cost(profile, jd_chars)
    except Exception:  # noqa: BLE001
        return None


def _spend_rows() -> list[dict[str, Any]]:
    if costmod is None or not hasattr(costmod, "SpendLedger"):
        return []
    try:
        ledger = costmod.SpendLedger()
        return [{"Key": r["alias"], "Provider": r["provider"],
                 "Spent this month (USD)": round(float(ledger.month_total(r["alias"])), 4),
                 "Monthly cap (USD)": r.get("monthly_cap_usd") or "none"} for r in vault.list()]
    except Exception:  # noqa: BLE001
        return []


def _budget_problems(est: dict[str, Any] | None) -> list[str]:
    if not est or costmod is None:
        return []
    try:
        ledger = costmod.SpendLedger()
        caps = {r["alias"]: r.get("monthly_cap_usd") for r in vault.list()}
        extra: dict[str, float] = {}
        for role, v in (est.get("per_role") or {}).items():
            b = (profile.roles.get(role) or [None])[0]
            if b is None or not PROVIDERS[b.provider].needs_key:
                continue
            alias = _resolve_alias(b)
            extra[alias] = extra.get(alias, 0.0) + _role_usd(v)
        out = []
        for alias, add in extra.items():
            cap = caps.get(alias)
            if cap:
                spent = float(ledger.month_total(alias))
                if spent + add > float(cap):
                    out.append(f"Key '{alias}': ${spent:.2f} spent + ~${add:.2f} for this run would exceed its "
                               f"${float(cap):.2f} monthly cap.")
        return out
    except Exception:  # noqa: BLE001
        return []


# =================================================================== CHAT
def _push(role: str, kind: str, content: Any = None, **extra: Any) -> None:
    st.session_state.messages.append({"role": role, "kind": kind, "content": content, **extra})


def _reset_chat() -> None:
    st.session_state.messages = [{"role": "assistant", "kind": "text", "content": GREETING}]
    st.session_state.chat_stage = "await_jd"
    st.session_state.pending_jd = ""
    st.session_state.last_result = None
    st.session_state.pop("draft_edit", None)


def _history_store() -> HistoryStore:
    return HistoryStore()


def _save_to_history(entry: dict[str, Any], draft: str) -> None:
    result = entry["result"]
    meta = {"retrieval_score": float(result.get("vector_relevance_score") or 0),
            "refinements": int(result.get("refinement_count", 0)),
            "verified": bool(result.get("letter_verified")),
            "cost_usd": float(result.get("cost_usd") or 0),
            "run_id": str(result.get("run_id", ""))[:32]}
    entry["saved_id"] = _history_store().add(entry["company"], _guess_title(entry["jd"]), entry["jd"], draft, meta)


def render_result(entry: dict[str, Any], interactive: bool) -> None:
    """The assistant's reply to a finished run. Only the newest reply is interactive."""
    result, company = entry["result"], entry["company"]
    if result.get("is_fallback"):
        st.warning(result.get("final_response", "I couldn't complete that request."))
    else:
        letter = result.get("cover_letter", "")
        chips = [_chip(f"🎯 match {result.get('vector_relevance_score') or 0}", "violet"),
                 _chip("✅ grounded" if result.get("letter_verified") else "⚠️ not verified",
                       "green" if result.get("letter_verified") else "amber"),
                 _chip(f"✏️ {result.get('refinement_count', 0)} refinement(s)", "blue"),
                 _chip(f"💲 {float(result.get('cost_usd') or 0):.4f}", "pink")]
        st.markdown("".join(chips), unsafe_allow_html=True)
        if not interactive:
            with st.expander(f"Earlier letter · {company}"):
                st.text(letter)
            return
        st.markdown(f"**Here's your cover letter for {_cell(company)}** 🎉")
        draft = st.session_state.get("draft_edit") or letter
        with st.container(border=True):
            st.text(draft)

        a1, a2, a3 = st.columns([1.3, 1, 1])
        saved_id = entry.get("saved_id")
        if saved_id:
            a1.success(f"Saved to history (#{saved_id})")
        elif a1.button("👍 I'm satisfied: save to history", key="save-hist", type="primary"):
            try:
                _save_to_history(entry, draft)
                st.toast("Saved to your history 📚")
                st.rerun()
            except (HistoryError, ValueError) as e:
                st.error(redact_secrets(e))
        if a2.button("🔁 Try again", key="try-again"):
            st.session_state.run_request = (entry["jd"], company)
            st.rerun()
        if a3.button("🆕 New letter", key="new-letter"):
            _reset_chat()
            st.rerun()

        with st.expander("✏️ Edit the draft and re-check grounding"):
            st.caption("Edit freely, then re-check that your edits don't claim skills your resume doesn't support.")
            st.text_area("Draft", key="draft_edit", height=340, label_visibility="collapsed")
            draft = st.session_state.get("draft_edit") or letter
            unverified = [s for s in result.get("unverified_skills", []) if len(s) > 2]
            if st.button("Re-check grounding", key="recheck"):
                bad = find_ungrounded_sentences(draft, unverified)
                if bad:
                    st.warning(f"{len(bad)} sentence(s) claim skills with no resume evidence:")
                    for sentence in bad:
                        st.code(sentence, language=None)
                else:
                    st.success("No ungrounded skill claims found in this draft.")

        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ .txt", draft, file_name="cover_letter.txt", key="dl-txt")
        try:
            d2.download_button("⬇️ .docx", letter_to_docx(draft, company), file_name="cover_letter.docx", key="dl-docx")
        except ImportError:
            d2.caption("Install python-docx for .docx export")
        try:
            d3.download_button("⬇️ .pdf", letter_to_pdf(draft, company), file_name="cover_letter.pdf",
                               mime="application/pdf", key="dl-pdf")
        except ImportError:
            d3.caption("Install fpdf2 for .pdf export")

        with st.expander("🧩 Skills-gap panel", expanded=False):
            detail = result.get("verified_skills_detailed") or {}
            rows = [{"Skill": d.get("skill"), "Status": "✅ verified", "Evidence": d.get("evidence", ""),
                     "In draft": "yes" if _mentions(str(d.get("skill", "")), draft) else "no"}
                    for d in detail.get("with_evidence", [])]
            rows += [{"Skill": d.get("skill"), "Status": "⚠️ no resume evidence", "Evidence": "—",
                      "In draft": "yes" if _mentions(str(d.get("skill", "")), draft) else "no"}
                     for d in detail.get("no_evidence", [])]
            if not rows:
                rows = [{"Skill": s, "Status": "✅ verified", "Evidence": "", "In draft": "—"}
                        for s in result.get("verified_skills", [])]
                rows += [{"Skill": s, "Status": "⚠️ no resume evidence", "Evidence": "—", "In draft": "—"}
                         for s in result.get("unverified_skills", [])]
            if rows:
                _table(rows, hide_index=True)
            else:
                st.info("No skills were analysed for this run.")

    if not interactive:
        return
    if result.get("run_id"):
        st.caption(f"Run id: {result['run_id']}")
    timings = result.get("node_timings", [])
    if timings:
        with st.expander("⏱️ Per-agent timing"):
            _table(timings, hide_index=True)
    usage = result.get("usage")
    if usage:
        with st.expander("🔢 Token usage"):
            try:
                _table(usage if isinstance(usage, list) else [usage], hide_index=True)
            except Exception:  # noqa: BLE001
                st.json(_scrub(usage), expanded=False)
    events = result.get("healing_log", [])
    with st.expander(f"🩹 Self-healing log ({len(events)} events)", expanded=bool(events)):
        for ev in events or ["No recovery actions were needed."]:
            st.text(ev)


def _execute_run(jd: str, company: str) -> None:
    """Run the pipeline inside an assistant bubble, then append the reply and rerun."""
    est = _estimate(len(jd))
    problems = _budget_problems(est)
    if problems:
        _push("assistant", "text", "I can't run that yet: " + " ".join(_cell(p) for p in problems))
        st.session_state.chat_stage = "await_jd"
        st.rerun()
    note = ""
    if est:
        note = (f"Estimated cost **${float(est.get('total_usd', 0)):.4f}** "
                f"(~{int(est.get('total_tokens', 0)):,} tokens).")
    with st.chat_message("assistant", avatar="✨"):
        if note:
            st.markdown(note)
        draft_buf: list[str] = []
        progress_lines: list[str] = []
        with st.status("Agents are working…", expanded=True) as status:
            live_box = st.empty()
            progress_box = st.empty()

            def sink(text: str | None) -> None:
                if text is None:  # a retry/failover restarted the stream
                    draft_buf.clear()
                else:
                    draft_buf.append(text)
                # plain statements, not a conditional expression: Streamlit "magic" would st.write() a bare expression
                if draft_buf:
                    live_box.text("Writing…\n" + "".join(draft_buf))
                else:
                    live_box.empty()

            def on_event(ev: dict[str, Any]) -> None:
                progress_lines.append(f"✓ {ev.get('node', '?')} ({ev.get('ms', '?')} ms)")
                progress_box.text("\n".join(progress_lines))

            result: dict[str, Any] | None = None
            error: str | None = None
            try:
                result = asyncio.run(run_pipeline(rt, jd, company, timeout_s=float(st.session_state.timeout_s),
                                                  token_sink=sink, on_event=on_event))
                status.update(label="Done", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="Something went wrong", state="error")
                if costmod is not None and isinstance(e, getattr(costmod, "BudgetExceeded", ())):
                    error = "Budget cap reached: " + redact_secrets(e)
                elif isinstance(e, TimeoutError):
                    error = f"The run timed out after {st.session_state.timeout_s}s. Raise the timeout in the sidebar or use faster models."
                else:
                    error = redact_secrets(e)
            live_box.empty()
    if result is not None:
        entry = {"company": company, "jd": jd, "result": result}
        st.session_state.last_result = entry
        if not result.get("is_fallback") and result.get("cover_letter"):
            st.session_state["draft_edit"] = result["cover_letter"]
        _push("assistant", "result", entry=entry)
    else:
        _push("assistant", "text", "😕 " + _cell(error or "That didn't work."))
    st.session_state.chat_stage = "await_jd"
    st.rerun()


def page_chat() -> None:
    st.markdown('<div class="hero"><h1>✨ Let\'s write your cover letter</h1>'
                '<p>Paste a job description, tell me the company, and I\'ll draft a letter grounded in your resume.</p></div>',
                unsafe_allow_html=True)
    missing = rt.missing_keys()
    if missing:
        st.warning("Missing keys for: " + ", ".join(missing) + ". Add them in **🔑 API Keys** or re-assign agents in **🤖 Agents & Models**.")
    statuses = st.session_state.conn_status
    if statuses:
        st.markdown("".join(_chip(f"{LIGHT['ok' if s['ok'] else 'fail']} {ROLES[r]['label']}", "grey")
                            for r, s in statuses.items() if r in ROLES), unsafe_allow_html=True)
    if getattr(rt, "chaos", 0.0):
        st.warning(f"Chaos mode is ON ({rt.chaos:.0%} fault probability): faults are injected on purpose.")  # type: ignore[attr-defined]

    msgs = st.session_state.messages
    last_result_idx = max((i for i, m in enumerate(msgs) if m["kind"] == "result"), default=-1)
    for i, m in enumerate(msgs):
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "✨"):
            if m["role"] == "user":
                st.markdown('<span class="user-marker"></span>', unsafe_allow_html=True)
            if m["kind"] == "text":
                if m["role"] == "user":
                    st.text(m["content"])
                else:
                    st.markdown(m["content"])
            elif m["kind"] == "jd":
                st.markdown(f"📋 **Job description** ({len(m['content']):,} characters)")
                with st.expander("Show what I pasted"):
                    st.text(m["content"][:3000] + ("…" if len(m["content"]) > 3000 else ""))
            elif m["kind"] == "result":
                render_result(m["entry"], interactive=(i == last_result_idx))

    req = st.session_state.run_request
    if req:
        st.session_state.run_request = None
        _push("user", "text", f"Try again for {req[1]}")
        _execute_run(*req)

    stage = st.session_state.chat_stage
    placeholder = ("Paste the job description here…" if stage == "await_jd"
                   else "Which company is this for?")
    text = st.chat_input(placeholder, max_chars=MAX_JD_CHARS if stage == "await_jd" else 120, disabled=bool(missing))
    if not text:
        return
    if stage == "await_jd":
        jd = sanitize_text(text)
        if len(jd) < MIN_JD_CHARS:
            _push("user", "text", text[:500])
            _push("assistant", "text", "That looks a bit short for a job posting. Could you paste the full description "
                                       "(responsibilities and requirements)?")
        else:
            _push("user", "jd", jd)
            st.session_state.pending_jd = jd
            st.session_state.chat_stage = "await_company"
            _push("assistant", "text", "Got it! 🙌 **Which company is this for?**")
        st.rerun()
    else:
        company = sanitize_text(text, 120).strip()
        if not company:
            st.rerun()
        _push("user", "text", company)
        _execute_run(st.session_state.pending_jd, company)


# =================================================================== HISTORY
def _prev_same_company(store: HistoryStore, e: Any) -> Any:
    """Most recent older saved letter for the same company (to show what changed)."""
    for other in store.list(limit=PAGE_SIZE * 5, offset=0):
        if other.id < e.id and other.company.lower() == e.company.lower():
            return other
    return None


def page_history() -> None:
    st.markdown('<div class="hero"><h1>📚 Saved cover letters</h1>'
                '<p>The letters you were happy with, stored only on this computer. Deleting one erases it permanently.</p></div>',
                unsafe_allow_html=True)
    store = _history_store()
    try:
        total = store.count()
    except HistoryError as e:
        st.error("Could not open the history database: " + redact_secrets(e))
        return
    pages = max(1, math.ceil(total / PAGE_SIZE))
    st.session_state.hist_page = min(max(0, st.session_state.hist_page), pages - 1)
    page_no = st.session_state.hist_page

    if total == 0:
        st.info("Nothing saved yet. After a letter is generated, press **👍 I'm satisfied: save to history** "
                "and it will appear here.")
        return
    first, last = page_no * PAGE_SIZE + 1, min(total, (page_no + 1) * PAGE_SIZE)
    st.markdown(_chip(f"{total} saved", "violet") + _chip(f"showing {first}–{last}", "blue")
                + _chip(f"page {page_no + 1} of {pages}", "grey"), unsafe_allow_html=True)

    entries = store.list(limit=PAGE_SIZE, offset=page_no * PAGE_SIZE)
    for e in entries:
        with st.container(border=True):
            h1, h2 = st.columns([5, 2])
            h1.markdown(f"<div class='card-title'>{html.escape(e.company)}</div>", unsafe_allow_html=True)
            h1.caption(f"{e.job_title} · saved {_pretty_date(e.created_at)}")
            chips = ""
            if e.meta.get("verified"):
                chips += _chip("✅ grounded", "green")
            if "retrieval_score" in e.meta:
                chips += _chip(f"🎯 match {e.meta['retrieval_score']:.0f}", "violet")
            h2.markdown(chips, unsafe_allow_html=True)

            with st.expander("Open letter"):
                st.code(e.letter, language=None, wrap_lines=True)   # read-only, with a built-in copy button
                prev = _prev_same_company(store, e)
                if prev is not None:
                    st.caption(f"Changes vs the previous saved letter for {_cell(e.company)}")
                    st.code(letter_diff(prev.letter, e.letter), language="diff")
                with st.expander("Job description it answered"):
                    st.text(e.jd[:4000] + ("…" if len(e.jd) > 4000 else ""))
                d1, d2, d3 = st.columns(3)
                d1.download_button("⬇️ .txt", e.letter, file_name=f"cover_letter_{e.id}.txt", key=f"htxt-{e.id}")
                try:
                    d2.download_button("⬇️ .docx", letter_to_docx(e.letter, e.company),
                                       file_name=f"cover_letter_{e.id}.docx", key=f"hdocx-{e.id}")
                except ImportError:
                    d2.caption("Install python-docx for .docx export")
                try:
                    d3.download_button("⬇️ .pdf", letter_to_pdf(e.letter, e.company),
                                       file_name=f"cover_letter_{e.id}.pdf", mime="application/pdf", key=f"hpdf-{e.id}")
                except ImportError:
                    d3.caption("Install fpdf2 for .pdf export")

            if st.session_state.confirm_delete == e.id:
                st.warning("This permanently erases the letter and its job description from this computer. "
                           "It cannot be undone.")
                c1, c2, _ = st.columns([1.4, 1, 3])
                if c1.button("Yes, erase forever", key=f"hyes-{e.id}", type="primary"):
                    store.delete(e.id)
                    st.session_state.confirm_delete = None
                    st.toast("Letter erased permanently 🗑️")
                    st.rerun()
                if c2.button("Cancel", key=f"hno-{e.id}"):
                    st.session_state.confirm_delete = None
                    st.rerun()
            elif st.button("🗑️ Delete permanently", key=f"hdel-{e.id}"):
                st.session_state.confirm_delete = e.id
                st.rerun()

    n1, n2, n3 = st.columns([1, 2, 1])
    if n1.button("◀ Newer", disabled=page_no == 0, key="hist-newer"):
        st.session_state.hist_page = page_no - 1
        st.session_state.confirm_delete = None
        st.rerun()
    n2.markdown(f"<div style='text-align:center'>{_chip(f'page {page_no + 1} / {pages}', 'grey')}</div>",
                unsafe_allow_html=True)
    if n3.button("Older ▶", disabled=page_no >= pages - 1, key="hist-older"):
        st.session_state.hist_page = page_no + 1
        st.session_state.confirm_delete = None
        st.rerun()

    with st.expander("⚠️ Danger zone: erase everything"):
        st.caption("Type DELETE to permanently erase all saved letters.")
        word = st.text_input("Confirmation", key="wipe-word", label_visibility="collapsed", placeholder="DELETE")
        if st.button("Erase all saved letters", key="wipe-all", disabled=word != "DELETE"):
            removed = store.delete_all()
            st.session_state.hist_page = 0
            st.session_state.confirm_delete = None
            st.toast(f"Erased {removed} letter(s) permanently")
            st.rerun()


# =================================================================== RESUMES
def page_resumes() -> None:
    st.subheader("📄 Your resumes")
    st.write("Resumes are PII-redacted before they are embedded and stored.")
    up = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], accept_multiple_files=False)
    if up is not None and st.button("Ingest resume", type="primary"):
        data = up.getvalue()
        ok, msg = validate_upload(up.name, data)
        if not ok:
            st.error(msg)
        else:
            try:
                from database import ingest_resume_to_pinecone
                with tempfile.TemporaryDirectory() as td, use_runtime(rt):
                    path = Path(td) / safe_filename(up.name)
                    path.write_bytes(data)
                    with st.spinner("Redacting, chunking, embedding…"):
                        ingest_resume_to_pinecone(str(path))
                st.success(f"Ingested {path.name}")
            except Exception as e:  # noqa: BLE001
                st.error(redact_secrets(e))
    if st.button("Refresh list"):
        pass
    try:
        from database import delete_resume_from_pinecone, list_stored_resumes
        with use_runtime(rt):
            names = list_stored_resumes()
        for n in names:
            col1, col2 = st.columns([6, 1])
            col1.write(n)
            if col2.button("Delete", key=f"del-{n}"):
                with use_runtime(rt):
                    delete_resume_from_pinecone(n)
                st.rerun()
        if not names:
            st.info("No resumes stored yet.")
    except Exception as e:  # noqa: BLE001
        st.info("Connect a vector-store key to list resumes. (" + redact_secrets(e)[:120] + ")")


# =================================================================== AGENTS & MODELS
def page_agents() -> None:
    st.subheader("🤖 Agents & models")
    st.write("Give each agent its own provider, model and API key. The first entry is primary; "
             "the second is the automatic failover used by the self-healing layer.")

    if st.button("Test all connections"):
        with st.spinner("Probing every agent's primary model…"):
            for _role in ROLES:
                if _role in profile.roles:
                    st.session_state.conn_status[_role] = check_connection(rt, _role)

    pc1, pc2 = st.columns([3, 1])
    preset = pc1.selectbox("Preset", list(PRESETS), format_func=lambda n: f"{n} — {PRESETS[n]['desc']}")
    if pc2.button("Apply preset"):
        new_prof = preset_profile(preset)
        if hasattr(profile, "settings"):
            new_prof.settings = dict(profile.settings)  # presets change models, not tuning settings
        st.session_state.profile = new_prof
        st.rerun()

    with st.expander("Compare profiles"):
        _opts = ["Saved on disk (profile.json)"] + list(PRESETS)
        _other_name = st.selectbox("Compare the current configuration with", _opts, key="cmp")
        _other = Profile.load() if _other_name == _opts[0] else preset_profile(_other_name)
        diff_rows = []
        for _role, _meta in ROLES.items():
            a, b = _fmt_bindings(profile.roles.get(_role)), _fmt_bindings(_other.roles.get(_role))
            if a != b:
                diff_rows.append({"Agent": _meta["label"], "Current": a, "Other": b})
        _cur_s, _oth_s = _settings(), {**SETTING_DEFAULTS, **dict(getattr(_other, "settings", {}) or {})}
        for _k in SETTING_DEFAULTS:
            if _cur_s[_k] != _oth_s[_k]:
                diff_rows.append({"Agent": f"setting: {_k}", "Current": str(_cur_s[_k]), "Other": str(_oth_s[_k])})
        if diff_rows:
            _table(diff_rows, hide_index=True)
        else:
            st.success("The two profiles are identical.")

    key_rows = vault.list()
    key_aliases = [""] + [k["alias"] for k in key_rows]
    new_roles: dict[str, list[Binding]] = {}
    for role, meta in ROLES.items():
        kind = meta["kind"]
        cur = profile.roles.get(role)
        if cur is None:
            continue
        status = st.session_state.conn_status.get(role)
        light = LIGHT["none"] if status is None else LIGHT["ok" if status["ok"] else "fail"]
        with st.expander(f"{light} {meta['label']} — {meta['desc']}"):
            bindings = []
            for slot, label in enumerate(["Primary", "Failover"]):
                if slot >= len(cur) and kind in ("embedding", "search", "vector"):
                    break  # no failover for stateful/dimension-bound roles
                b = cur[slot] if slot < len(cur) else None
                st.markdown(f"**{label}**")
                c1, c2, c3 = st.columns(3)
                provs = providers_for(kind)
                prov = c1.selectbox("Provider", provs, index=provs.index(b.provider) if b and b.provider in provs else 0,
                                    key=f"{role}-{slot}-p")
                models = PROVIDERS[prov].models.get(kind, [])
                model = c2.text_input("Model", value=b.model if b and b.provider == prov else (models[0] if models else ""),
                                      key=f"{role}-{slot}-m", help="Suggested: " + ", ".join(models))
                if PROVIDERS[prov].needs_key:
                    opts = [a for a in key_aliases if not a or next((k["provider"] for k in key_rows if k["alias"] == a), prov) == prov]
                    alias = c3.selectbox("API key", opts, index=opts.index(b.key_alias) if b and b.key_alias in opts else 0,
                                         key=f"{role}-{slot}-k", format_func=lambda a: a or "(first key for provider)")
                else:
                    alias = ""
                    c3.caption("No key needed")
                bindings.append(Binding(prov, model.strip(), alias))
            new_roles[role] = bindings
            if status is not None:
                st.caption(f"{light} {status['detail']} ({status['ms']} ms)")
            if st.button("Test connection", key=f"test-{role}"):
                with st.spinner("Testing…"):
                    st.session_state.conn_status[role] = check_connection(rt, role)
                st.rerun()
    if st.button("Save agent configuration", type="primary"):
        try:
            candidate = Profile(new_roles)
            if hasattr(profile, "settings"):
                candidate.settings = dict(profile.settings)
            candidate.save()
            st.session_state.profile = candidate
            st.success("Saved.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))


# =================================================================== API KEYS
def page_keys() -> None:
    st.subheader("🔑 API keys")
    st.write("Keys are encrypted with a master key held in your OS keychain (or your passphrase, if you set one). "
             "Raw keys are never shown again.")
    rows = vault.list()
    due = [r["alias"] for r in rows if r.get("rotate_due")]
    if due:
        st.warning("Rotation due for: " + ", ".join(due) + ". Generate a new key at the provider, add it here, then remove the old one.")
    if rows:
        for r in rows:
            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
            c1.write(f"**{r['alias']}**" + (" ⏰" if r.get("rotate_due") else ""))
            c2.write(PROVIDERS[r["provider"]].label if r["provider"] in PROVIDERS else r["provider"])
            c3.code(r["masked"])
            c4.caption(f"{r['age_days']} days old" if r.get("age_days") is not None else "")
            if c5.button("Remove", key=f"rm-{r['alias']}"):
                vault.delete(r["alias"])
                st.rerun()
    else:
        st.info("No keys stored yet.")
    st.divider()
    with st.form("add-key", clear_on_submit=True):
        prov = st.selectbox("Provider", [n for n, p in PROVIDERS.items() if p.needs_key],
                            format_func=lambda n: PROVIDERS[n].label)
        alias = st.text_input("Alias", placeholder="e.g. personal-openai")
        secret = st.text_input("API key", type="password", autocomplete="off")
        cap = st.number_input("Monthly spend cap in USD (0 = no cap)", min_value=0.0, value=0.0, step=1.0)
        persist = st.checkbox("Remember (encrypted on disk). Untick for this session only.", value=True)
        if st.form_submit_button("Save key"):
            try:
                kwargs: dict[str, Any] = {"persist": persist}
                if cap > 0:
                    kwargs["monthly_cap_usd"] = float(cap)
                try:
                    vault.set(alias or prov, prov, secret, **kwargs)
                except TypeError:  # vault build without cap support
                    vault.set(alias or prov, prov, secret, persist=persist)
                st.success("Key stored.")
                st.rerun()
            except VaultError as e:
                st.error(str(e))


# =================================================================== RUNS
def page_runs() -> None:
    st.subheader("📊 Runs")
    if runlog is None:
        st.info("Run logging is not available in this build.")
        return
    runs = runlog.read_runs()
    if not runs:
        st.info("No runs recorded yet. Generate a letter and it will appear here (run ids, timings and cost only; "
                "letter text is never logged).")
        return
    st.subheader(f"Latency and cost across {len(runs)} run(s)")
    pct = _node_percentiles(runs)
    if pct:
        _table(pct, hide_index=True)
    try:
        summary = _scrub(runlog.summarize_runs(runs))
        with st.expander("Run summary"):
            st.json(summary, expanded=False)
    except Exception as e:  # noqa: BLE001
        st.caption("Summary unavailable: " + redact_secrets(e)[:120])
    cols = ("run_id", "ts", "ok", "fallback", "error_type", "duration_ms", "cost_usd", "tokens_in",
            "tokens_out", "healing_events", "letter_verified")
    recent = [{c: r[c] for c in cols if c in r} for r in runs[-50:]][::-1]
    st.subheader("Recent runs")
    _table(recent, hide_index=True)


# =================================================================== SECURITY
def page_security() -> None:
    st.subheader("🛡️ Security")
    st.markdown("""
- **Keys**: Fernet-encrypted vault; master key in OS keychain (file fallback `0600`), or derived from your passphrase. Not in `.env`, logs, or profile JSON.
- **Logs & errors**: secret-pattern redaction on all log records and UI errors.
- **Inputs**: Unicode normalisation, control/bidi-character stripping, length caps, prompt-injection fencing (JD and web research are treated as data).
- **Uploads**: extension + magic-byte + size checks; filenames sanitised against path traversal; PII redacted before storage.
- **History**: saved letters live in a local SQLite file only; deleting one overwrites and vacuums it so it is gone from the file.
- **Network**: UI binds to `127.0.0.1`, XSRF protection on, telemetry off.
- **Resilience**: retry → provider failover → circuit breaker → graceful degradation; ungrounded skill claims can never ship.
""")
    rows = vault.list()

    st.subheader("Vault passphrase")
    if not hasattr(vault, "enable_passphrase"):
        st.info("This build of the vault has no passphrase support.")
    else:
        pass_on = bool(getattr(vault, "uses_passphrase", False))
        st.caption("Passphrase protection: **ON**" if pass_on else
                   "Passphrase protection: **OFF** (master key is held in your OS keychain). "
                   "With a passphrase, anyone with your user account but not the passphrase cannot read the keys.")
        if not pass_on:
            with st.form("pp-enable", clear_on_submit=True):
                _p1 = st.text_input("New passphrase (12+ characters)", type="password", autocomplete="off")
                _p2 = st.text_input("Repeat passphrase", type="password", autocomplete="off")
                if st.form_submit_button("Enable passphrase"):
                    if _p1 != _p2:
                        st.error("Passphrases do not match.")
                    elif len(_p1) < 12:
                        st.error("Use at least 12 characters.")
                    else:
                        try:
                            vault.enable_passphrase(_p1)
                            st.success("Passphrase enabled.")
                        except VaultError as e:
                            st.error(str(e))
        else:
            with st.form("pp-disable", clear_on_submit=True):
                _cur = st.text_input("Current passphrase", type="password", autocomplete="off")
                if st.form_submit_button("Disable passphrase"):
                    try:
                        vault.disable_passphrase(_cur)
                        st.success("Passphrase removed; the keychain protects the vault again.")
                    except Exception:  # noqa: BLE001
                        st.error("Could not disable: wrong passphrase.")
        if st.button("🔒 Lock vault now", key="lock-now", disabled=not pass_on):
            vault.lock()
            st.rerun()

    st.subheader("Key age, last use and rotation")
    if rows:
        _table([{"Key": r["alias"], "Provider": r["provider"], "Created": r.get("created_at", ""),
                 "Last used": r.get("last_used") or "never", "Age (days)": r.get("age_days", ""),
                 "Rotation": "⏰ due" if r.get("rotate_due") else "ok",
                 "Monthly cap (USD)": r.get("monthly_cap_usd") or "none"} for r in rows], hide_index=True)
        if hasattr(vault, "set_cap"):
            with st.form("cap-edit"):
                _alias = st.selectbox("Key", [r["alias"] for r in rows])
                _cap = st.number_input("Monthly cap (USD, 0 removes the cap)", min_value=0.0, value=0.0, step=1.0)
                if st.form_submit_button("Save cap"):
                    try:
                        vault.set_cap(_alias, float(_cap) if _cap > 0 else None)
                        st.success("Cap saved.")
                        st.rerun()
                    except VaultError as e:
                        st.error(str(e))
        spend = _spend_rows()
        if spend:
            with st.expander("Spend this month vs caps"):
                _table(spend, hide_index=True)
    else:
        st.info("No keys stored yet.")

    st.subheader("Retrieval settings")
    cur_s = _settings()
    with st.form("settings-edit"):
        s1, s2, s3 = st.columns(3)
        _alpha = s1.slider("Hybrid alpha (1 = dense only, 0 = keyword only)", 0.0, 1.0, float(cur_s["alpha"]), 0.05)
        _max_res = s2.number_input("Max resumes searched", min_value=1, max_value=50, value=int(cur_s["max_resumes"]))
        _top_k = s3.number_input("Top-k chunks per resume", min_value=1, max_value=50, value=int(cur_s["top_k"]))
        if st.form_submit_button("Save settings"):
            if not hasattr(profile, "settings"):
                st.error("This build does not support persisted settings.")
            else:
                try:
                    profile.settings = {**profile.settings, "alpha": float(_alpha),
                                        "max_resumes": int(_max_res), "top_k": int(_top_k)}
                    profile.save()
                    st.success("Settings saved.")
                except ValueError as e:
                    st.error(str(e))


# =================================================================== router
_PAGE_RENDERERS = {
    PAGE_CHAT: page_chat, PAGE_HISTORY: page_history, PAGE_RESUMES: page_resumes, PAGE_AGENTS: page_agents,
    PAGE_KEYS: page_keys, PAGE_RUNS: page_runs, PAGE_SECURITY: page_security,
}
_PAGE_RENDERERS[page]()   # a call statement; its (None) result is never passed to st.write
