"""Streamlit GUI.  Run:  streamlit run app.py   (binds to 127.0.0.1 only; see .streamlit/config.toml)

Optional modules (cost, runlog, vault passphrase API, PDF export) are probed defensively so the
app still renders if one is absent in a given build.
"""
import asyncio
import datetime as dt
import difflib
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from export import letter_to_docx, letter_to_pdf
from pipeline import run_pipeline
from providers import PROVIDERS, providers_for
from runtime import PRESETS, ROLES, Binding, Profile, Runtime, check_connection, preset_profile, use_runtime
from security import MAX_JD_CHARS, install_log_redaction, redact_secrets, safe_filename, validate_upload
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

st.set_page_config(page_title="Job Application Assistant", page_icon="✉️", layout="wide")
install_log_redaction()

LIGHT = {"ok": "🟢", "fail": "🔴", "none": "⚪"}
SETTING_DEFAULTS = {"alpha": 0.4, "max_resumes": 10, "top_k": 10}
_PRIVATE_KEY_HINTS = ("letter", "cover", "text", "prompt", "resume", "secret", "key")


# ------------------------------------------------------------------ small helpers
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


@st.cache_resource
def get_vault() -> KeyVault:
    return KeyVault()


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
st.session_state.setdefault("history", [])
st.session_state.setdefault("conn_status", {})
st.session_state.setdefault("last_result", None)


def _settings() -> dict[str, Any]:
    return {**SETTING_DEFAULTS, **dict(getattr(profile, "settings", {}) or {})}


rt = Runtime(vault, profile)

# ------------------------------------------------------------------ sidebar: chaos demo
with st.sidebar.expander("🧪 Chaos mode (demo)"):
    st.caption("Demo of self-healing: randomly injects provider faults (timeouts, 429s, bad JSON, auth errors) "
               "into model calls. Leave off for real use; expect slower, sometimes degraded runs. "
               "Watch the self-healing log after a run.")
    _chaos_on = st.toggle("Enable chaos", key="chaos_on")
    _chaos_p = st.slider("Fault probability", 0.0, 0.9, 0.3, 0.05, key="chaos_p", disabled=not _chaos_on)
try:
    rt.chaos = float(_chaos_p) if _chaos_on else 0.0  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    pass

st.title("✉️ Job Application Assistant")
st.caption("Self-healing multi-agent pipeline · keys are encrypted locally and never written to .env")

tab_run, tab_resumes, tab_agents, tab_keys, tab_runs, tab_security = st.tabs(
    ["Generate", "Resumes", "Agents & Models", "API Keys", "Runs", "Security"])


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


# ---------------------------------------------------------------- Generate
def render_result(entry: dict[str, Any]) -> None:
    result, company = entry["result"], entry["company"]
    if result.get("is_fallback"):
        st.warning(result.get("final_response", "Request could not be completed."))
    else:
        st.subheader("Cover letter")
        st.caption("Edit freely, then re-check that the edits don't claim skills your resume doesn't support.")
        draft = st.text_area("Draft", key="draft_edit", height=380, label_visibility="collapsed")
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
        d1.download_button("Download .txt", draft, file_name="cover_letter.txt", key="dl-txt")
        try:
            d2.download_button("Download .docx", letter_to_docx(draft, company), file_name="cover_letter.docx",
                               key="dl-docx")
        except ImportError:
            d2.caption("Install python-docx for .docx export")
        try:
            d3.download_button("Download .pdf", letter_to_pdf(draft, company), file_name="cover_letter.pdf",
                               mime="application/pdf", key="dl-pdf")
        except ImportError:
            d3.caption("Install fpdf2 for .pdf export")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval score", result.get("vector_relevance_score") or 0)
        c2.metric("Refinements", result.get("refinement_count", 0))
        c3.metric("Grounding verified", "yes" if result.get("letter_verified") else "no")
        c4.metric("Cost (USD)", f"{float(result.get('cost_usd') or 0):.4f}")
        if result.get("run_id"):
            st.caption(f"Run id: {result['run_id']}")

        with st.expander("Skills-gap panel", expanded=True):
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

    timings = result.get("node_timings", [])
    if timings:
        with st.expander("Per-agent timing"):
            _table(timings, hide_index=True)
    usage = result.get("usage")
    if usage:
        with st.expander("Token usage"):
            try:
                _table(usage if isinstance(usage, list) else [usage], hide_index=True)
            except Exception:  # noqa: BLE001
                st.json(_scrub(usage), expanded=False)
    events = result.get("healing_log", [])
    with st.expander(f"Self-healing log ({len(events)} events)", expanded=bool(events)):
        for ev in events or ["No recovery actions were needed."]:
            st.text(ev)


with tab_run:
    missing = rt.missing_keys()
    if missing:
        st.warning("Missing keys for: " + ", ".join(missing) + ". Add them in **API Keys** or re-assign agents in **Agents & Models**.")
    statuses = st.session_state.conn_status
    if statuses:
        st.caption("Connections: " + "  ".join(
            f"{LIGHT['ok' if s['ok'] else 'fail']} {ROLES[r]['label']}" for r, s in statuses.items() if r in ROLES))
    if getattr(rt, "chaos", 0.0):
        st.warning(f"Chaos mode is ON ({rt.chaos:.0%} fault probability): faults are injected on purpose.")  # type: ignore[attr-defined]

    company = st.text_input("Company name", max_chars=120)
    jd = st.text_area("Job description", height=280, max_chars=MAX_JD_CHARS)
    timeout_s = st.slider("Timeout (seconds)", 30, 600, 180, 10,
                          help="The run is aborted after this long. Streamlit's Stop button (top right while running) also cancels.")

    est = _estimate(len(jd))
    problems = _budget_problems(est)
    if est:
        e1, e2 = st.columns(2)
        e1.metric("Estimated cost", f"${float(est.get('total_usd', 0)):.4f}")
        e2.metric("Estimated tokens", f"{int(est.get('total_tokens', 0)):,}")
        with st.expander("Estimate by agent"):
            _table([{"Agent": ROLES[r]["label"] if r in ROLES else r, "Estimated USD": round(_role_usd(v), 5)}
                          for r, v in (est.get("per_role") or {}).items()], hide_index=True)
            st.caption("Rough pre-run estimate from prompt sizes and the built-in price table; actual spend is logged after the run.")
    spend = _spend_rows()
    if spend:
        with st.expander("Spend this month vs caps"):
            _table(spend, hide_index=True)
    for p in problems:
        st.error("Budget cap would be exceeded. " + p)

    if st.button("Generate cover letter", type="primary",
                 disabled=bool(missing) or bool(problems) or not (jd.strip() and company.strip())):
        st.session_state.last_result = None
        draft_buf: list[str] = []
        progress_lines: list[str] = []
        with st.status("Agents working…", expanded=True) as status:
            live_box = st.empty()
            progress_box = st.empty()

            def sink(text: str | None) -> None:
                if text is None:  # a retry/failover restarted the stream
                    draft_buf.clear()
                else:
                    draft_buf.append(text)
                live_box.text("Live draft:\n" + "".join(draft_buf)) if draft_buf else live_box.empty()

            def on_event(ev: dict[str, Any]) -> None:
                progress_lines.append(f"✓ {ev.get('node', '?')} ({ev.get('ms', '?')} ms)")
                progress_box.text("\n".join(progress_lines))

            result: dict[str, Any] | None = None
            try:
                result = asyncio.run(run_pipeline(rt, jd, company, timeout_s=float(timeout_s),
                                                  token_sink=sink, on_event=on_event))
                status.update(label="Done", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="Failed", state="error")
                if costmod is not None and isinstance(e, getattr(costmod, "BudgetExceeded", ())):
                    st.error("Budget cap reached: " + redact_secrets(e))
                elif isinstance(e, TimeoutError):
                    st.error(f"The run timed out after {timeout_s}s. Raise the timeout or switch to faster models.")
                else:
                    st.error(redact_secrets(e))
            live_box.empty()
        if result:
            st.session_state.last_result = {"company": company, "result": result}
            if not result.get("is_fallback") and result.get("cover_letter"):
                st.session_state["draft_edit"] = result["cover_letter"]
                st.session_state.history = ([{"company": company, "letter": result["cover_letter"],
                                              "run_id": result.get("run_id", ""),
                                              "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}]
                                            + st.session_state.history)[:10]

    if st.session_state.last_result:
        render_result(st.session_state.last_result)

    if st.session_state.history:
        with st.expander(f"Recent letters ({len(st.session_state.history)}, this session only)"):
            hist = st.session_state.history
            for i, h in enumerate(hist):
                st.markdown(f"**{h['company']}** · {h.get('ts', '')} · `{h.get('run_id', '')}`")
                prev = next((o for o in hist[i + 1:] if o["company"] == h["company"]), None)
                if prev is not None:
                    st.caption("Changes vs the previous letter for this company")
                    st.code(letter_diff(prev["letter"], h["letter"]), language="diff")
                else:
                    st.text(h["letter"][:600] + ("…" if len(h["letter"]) > 600 else ""))

# ----------------------------------------------------------------- Resumes
with tab_resumes:
    st.write("Resumes are PII-redacted before they are embedded and stored.")
    up = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], accept_multiple_files=False)
    if up is not None and st.button("Ingest resume"):
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

# ---------------------------------------------------------- Agents & Models
with tab_agents:
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

# ---------------------------------------------------------------- API Keys
with tab_keys:
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

# -------------------------------------------------------------------- Runs
with tab_runs:
    if runlog is None:
        st.info("Run logging is not available in this build.")
    else:
        runs = runlog.read_runs()
        if not runs:
            st.info("No runs recorded yet. Generate a letter and it will appear here (run ids, timings and cost only; "
                    "letter text is never logged).")
        else:
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

# ---------------------------------------------------------------- Security
with tab_security:
    st.markdown("""
- **Keys**: Fernet-encrypted vault; master key in OS keychain (file fallback `0600`), or derived from your passphrase. Not in `.env`, logs, or profile JSON.
- **Logs & errors**: secret-pattern redaction on all log records and UI errors.
- **Inputs**: Unicode normalisation, control/bidi-character stripping, length caps, prompt-injection fencing (JD and web research are treated as data).
- **Uploads**: extension + magic-byte + size checks; filenames sanitised against path traversal; PII redacted before storage.
- **Network**: UI binds to `127.0.0.1`, XSRF protection on, telemetry off.
- **Resilience**: retry → provider failover → circuit breaker → graceful degradation; ungrounded skill claims can never ship.
""")

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
