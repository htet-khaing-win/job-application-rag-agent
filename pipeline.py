"""Single entry point used by both the CLI and the GUI."""
from __future__ import annotations

from typing import Any, Dict, Optional

from graph import build_graph
from resilience import start_heal_log
from runtime import Runtime, use_runtime
from security import MAX_COMPANY_CHARS, install_log_redaction, looks_like_injection, sanitize_text


def initial_state(job_description: str, company: str) -> Dict[str, Any]:
    return {
        "job_description": sanitize_text(job_description),
        "company_name": sanitize_text(company, MAX_COMPANY_CHARS),
    }


async def run_pipeline(rt: Runtime, job_description: str, company: str,
                       generator_llm: Optional[Any] = None, critic_llm: Optional[Any] = None) -> Dict[str, Any]:
    """Validate input, run the self-healing graph under `rt`, return the final state dict."""
    install_log_redaction()
    state = initial_state(job_description, company)
    if not state["job_description"]:
        raise ValueError("Job description is empty.")
    if not state["company_name"]:
        raise ValueError("Company name is required.")
    missing = rt.missing_keys()
    if missing and generator_llm is None:
        raise ValueError("Missing API keys for: " + ", ".join(missing))

    injection = looks_like_injection(state["job_description"])
    app = build_graph(generator_llm, critic_llm)
    with use_runtime(rt):
        start_heal_log()
        result = await app.ainvoke(state, config={
            "run_name": "job_application_run",
            "tags": ["langgraph", "async", "cover-letter", "self-healing"],
            "metadata": {"has_company": True},
        })
    result = dict(result)
    if injection:
        result.setdefault("healing_log", []).append("[security] possible prompt-injection text detected in JD; treated as data")
    return result
