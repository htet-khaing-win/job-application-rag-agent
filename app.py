"""Streamlit GUI.  Run:  streamlit run app.py   (binds to 127.0.0.1 only; see .streamlit/config.toml)"""
import asyncio
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import run_pipeline
from providers import PROVIDERS, providers_for
from runtime import ROLES, Binding, Profile, Runtime, use_runtime
from security import (MAX_JD_CHARS, install_log_redaction, redact_secrets, safe_filename,
                      validate_upload)
from vault import KeyVault, VaultError

st.set_page_config(page_title="Job Application Assistant", page_icon="✉️", layout="wide")
install_log_redaction()


@st.cache_resource
def get_vault() -> KeyVault:
    return KeyVault()


vault = get_vault()
if "profile" not in st.session_state:
    st.session_state.profile = Profile.load()
profile: Profile = st.session_state.profile
rt = Runtime(vault, profile)

st.title("✉️ Job Application Assistant")
st.caption("Self-healing multi-agent pipeline · keys are encrypted locally and never written to .env")

tab_run, tab_resumes, tab_agents, tab_keys, tab_security = st.tabs(
    ["Generate", "Resumes", "Agents & Models", "API Keys", "Security"])

# ---------------------------------------------------------------- Generate
with tab_run:
    missing = rt.missing_keys()
    if missing:
        st.warning("Missing keys for: " + ", ".join(missing) + ". Add them in **API Keys** or re-assign agents in **Agents & Models**.")
    company = st.text_input("Company name", max_chars=120)
    jd = st.text_area("Job description", height=280, max_chars=MAX_JD_CHARS)
    if st.button("Generate cover letter", type="primary", disabled=bool(missing) or not (jd.strip() and company.strip())):
        with st.status("Agents working…", expanded=True) as status:
            try:
                result = asyncio.run(run_pipeline(rt, jd, company))
                status.update(label="Done", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="Failed", state="error")
                st.error(redact_secrets(e))
                result = None
        if result:
            if result.get("is_fallback"):
                st.warning(result.get("final_response", "Request could not be completed."))
            else:
                st.subheader("Cover letter")
                st.text_area("Copy your letter", result["cover_letter"], height=380, label_visibility="collapsed")
                st.download_button("Download .txt", result["cover_letter"], file_name="cover_letter.txt")
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval score", result.get("vector_relevance_score") or 0)
                c2.metric("Refinements", result.get("refinement_count", 0))
                c3.metric("Grounding verified", "yes" if result.get("letter_verified") else "no")
                with st.expander("Verified / unverified skills"):
                    st.write("**Verified:**", ", ".join(result.get("verified_skills", [])) or "—")
                    st.write("**To address:**", ", ".join(result.get("unverified_skills", [])) or "—")
            events = result.get("healing_log", [])
            with st.expander(f"Self-healing log ({len(events)} events)", expanded=bool(events)):
                for ev in events or ["No recovery actions were needed."]:
                    st.text(ev)

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
    key_aliases = [""] + [k["alias"] for k in vault.list()]
    new_roles = {}
    for role, meta in ROLES.items():
        kind = meta["kind"]
        with st.expander(f"{meta['label']} — {meta['desc']}"):
            cur = profile.roles[role]
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
                    opts = [a for a in key_aliases if not a or next((k["provider"] for k in vault.list() if k["alias"] == a), prov) == prov]
                    alias = c3.selectbox("API key", opts, index=opts.index(b.key_alias) if b and b.key_alias in opts else 0,
                                         key=f"{role}-{slot}-k", format_func=lambda a: a or "(first key for provider)")
                else:
                    alias = ""
                    c3.caption("No key needed")
                bindings.append(Binding(prov, model.strip(), alias))
            new_roles[role] = bindings
    if st.button("Save agent configuration", type="primary"):
        try:
            candidate = Profile(new_roles)
            candidate.save()
            st.session_state.profile = candidate
            st.success("Saved.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- API Keys
with tab_keys:
    st.write("Keys are encrypted with a master key held in your OS keychain. Raw keys are never shown again.")
    rows = vault.list()
    if rows:
        for r in rows:
            c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
            c1.write(f"**{r['alias']}**")
            c2.write(PROVIDERS[r["provider"]].label if r["provider"] in PROVIDERS else r["provider"])
            c3.code(r["masked"])
            if c4.button("Remove", key=f"rm-{r['alias']}"):
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
        persist = st.checkbox("Remember (encrypted on disk). Untick for this session only.", value=True)
        if st.form_submit_button("Save key"):
            try:
                vault.set(alias or prov, prov, secret, persist=persist)
                st.success("Key stored.")
                st.rerun()
            except VaultError as e:
                st.error(str(e))

# ---------------------------------------------------------------- Security
with tab_security:
    st.markdown("""
- **Keys**: Fernet-encrypted vault; master key in OS keychain (file fallback `0600`). Not in `.env`, logs, or profile JSON.
- **Logs & errors**: secret-pattern redaction on all log records and UI errors.
- **Inputs**: Unicode normalisation, control/bidi-character stripping, length caps, prompt-injection fencing (JD and web research are treated as data).
- **Uploads**: extension + magic-byte + size checks; filenames sanitised against path traversal; PII redacted before storage.
- **Network**: UI binds to `127.0.0.1`, XSRF protection on, telemetry off.
- **Resilience**: retry → provider failover → circuit breaker → graceful degradation; ungrounded skill claims can never ship.
""")
