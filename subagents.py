"""Specialised sub-agents, one per pipeline node, plus the self-healing Supervisor.

Each SubAgent owns exactly one role in the runtime profile, so the user can bind a
different provider / model / API key to every agent. LLMs are resolved lazily at call
time, which means keys can be rotated in the GUI without rebuilding the graph.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from database import retrieve_resumes_node
from node import (critique_letter_node, fallback_handler_node, generate_summary_node,
                  ingest_jd_node, join_context_node, refine_letter_node,
                  research_company_node, write_cover_letter_node)
from resilience import drain_heal_log, record_heal
from runtime import get_runtime
from security import redact_secrets
from state import GraphState

Update = Dict[str, Any]
Degrade = Callable[[GraphState, BaseException], Update]


class SubAgent:
    role: str = ""
    name: str = ""
    # node-level retries for deterministic failures (LLM calls already retry/failover internally)
    node_retries: int = 0

    def __init__(self, llm_override: Any = None):
        self._override = llm_override

    @property
    def llm(self) -> Any:
        return self._override if self._override is not None else get_runtime().chat(self.role)

    def run(self, state: GraphState) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        """Return a safe state update when the agent cannot recover; None = unrecoverable."""
        return None


class JDAnalystAgent(SubAgent):
    role, name = "jd_analyst", "JD Analyst"

    def run(self, state: GraphState) -> Update:
        return ingest_jd_node(state, self.llm)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        from security import sanitize_text
        text = sanitize_text(state.job_description)
        low = text.lower()
        if len(text) > 200 and any(k in low for k in ("requirements", "responsibilities", "qualifications", "experience")):
            record_heal("[jd_analyst] LLM unavailable: used keyword heuristic validation")
            return {"cleaned_jd": text, "is_valid_jd": True}
        return {"is_valid_jd": False, "error_type": "invalid_input",
                "error_message": "Could not validate the job description (model unavailable)."}


class ResearchAgent(SubAgent):
    role, name = "researcher", "Researcher"
    node_retries = 1

    async def run(self, state: GraphState) -> Update:
        return await research_company_node(state)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        return {"company_research": f"Unable to fetch current information about {state.company_name}.",
                "company_research_success": False}


class RetrieverAgent(SubAgent):
    role, name = "vector_store", "Retriever"
    node_retries = 2

    async def run(self, state: GraphState) -> Update:
        return await retrieve_resumes_node(state)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        return {"retrieved_chunks": [], "vector_relevance_score": 0.0,
                "error_type": "retrieval_failed",
                "error_message": "Resume retrieval failed: " + redact_secrets(exc)[:200]}


class AnalystAgent(SubAgent):
    role, name = "analyst", "Analyst"

    def run(self, state: GraphState) -> Update:
        return generate_summary_node(state)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        return {"error_type": "generation_failed", "error_message": "Skill analysis failed."}


class WriterAgent(SubAgent):
    role, name = "writer", "Copywriter"

    def run(self, state: GraphState) -> Update:
        return write_cover_letter_node(state, self.llm)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        return {"error_type": "generation_failed",
                "error_message": "Every configured writer model failed: " + redact_secrets(exc)[:300]}


class CriticAgent(SubAgent):
    role, name = "critic", "Hiring Manager"

    def run(self, state: GraphState) -> Update:
        return critique_letter_node(state, self.llm)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        record_heal("[critic] unavailable: skipping critique, shipping verified draft")
        return {"refinement_edits": [], "needs_refinement": False,
                "critique_feedback": "Critique skipped (model unavailable)."}


class EditorAgent(SubAgent):
    role, name = "editor", "Editor"

    def run(self, state: GraphState) -> Update:
        return refine_letter_node(state, self.llm)

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        record_heal("[editor] unavailable: keeping un-refined letter")
        return {"cover_letter": state.cover_letter, "needs_refinement": False,
                "refinement_count": state.refinement_count + 1}


# --- Fact checker -----------------------------------------------------------
_HEDGES = ("learn", "eager", "transferable", "build on", "building on", "grow", "develop",
           "ramp", "quickly acquire", "prepared to", "willing to", "excited to")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def find_ungrounded_sentences(letter: str, unverified: List[str]) -> List[str]:
    """Sentences that claim an unverified skill without a learning/transfer hedge."""
    bad = []
    for sent in _SENT_SPLIT.split(letter):
        low = sent.lower()
        for skill in unverified:
            if re.search(rf"(?<!\w){re.escape(skill.lower())}(?!\w)", low):
                if not any(h in low for h in _HEDGES):
                    bad.append(sent)
                break
    return bad


class FactCheckerAgent(SubAgent):
    """Deterministic grounding check with a bounded LLM repair loop.

    Converges by construction: after MAX_REPAIRS failed LLM repairs, offending sentences
    are removed deterministically, so an ungrounded claim can never ship.
    """
    role, name = "verifier", "Fact Checker"
    MAX_REPAIRS = 2

    def run(self, state: GraphState) -> Update:
        letter = state.cover_letter
        unverified = [s for s in state.unverified_skills if len(s) > 2]
        bad = find_ungrounded_sentences(letter, unverified)
        attempts = 0
        while bad and attempts < self.MAX_REPAIRS:
            attempts += 1
            record_heal(f"[verifier] {len(bad)} ungrounded sentence(s); repair attempt {attempts}")
            prompt = (
                "Rewrite ONLY the listed sentences so they no longer claim experience with the "
                "unverified skills; instead say the candidate is prepared to learn them via a "
                "verified skill. Keep everything else identical. Output the full letter only.\n\n"
                f"UNVERIFIED SKILLS: {', '.join(unverified)}\n"
                "SENTENCES TO FIX:\n" + "\n".join(f"- {b}" for b in bad) +
                f"\n\nLETTER:\n{letter}"
            )
            try:
                candidate = self.llm.invoke(prompt).content.strip()
            except Exception as e:  # noqa: BLE001
                record_heal(f"[verifier] repair LLM failed: {redact_secrets(e)[:160]}")
                break
            if candidate:
                letter = candidate
            bad = find_ungrounded_sentences(letter, unverified)
        if bad:
            record_heal(f"[verifier] removed {len(bad)} ungrounded sentence(s) deterministically")
            for b in bad:
                letter = letter.replace(b, "", 1)
            letter = re.sub(r"[ \t]{2,}", " ", letter).strip()
        return {"cover_letter": letter, "letter_verified": True, "verification_attempts": attempts}

    def degrade(self, state: GraphState, exc: BaseException) -> Optional[Update]:
        bad = find_ungrounded_sentences(state.cover_letter, state.unverified_skills)
        letter = state.cover_letter
        for b in bad:
            letter = letter.replace(b, "", 1)
        record_heal("[verifier] crashed: applied deterministic sentence removal")
        return {"cover_letter": letter.strip(), "letter_verified": True}


class ContextJoinAgent(SubAgent):
    role, name = "analyst", "Context Join"

    def run(self, state: GraphState) -> Update:
        return join_context_node(state)


class FallbackAgent(SubAgent):
    role, name = "analyst", "Fallback"

    def run(self, state: GraphState) -> Update:
        return fallback_handler_node(state)


# --- Supervisor -------------------------------------------------------------
class Supervisor:
    """Wraps agents as graph nodes: retries, degrade-on-failure, and heal-event logging."""

    def __init__(self, backoff_s: float = 0.5):
        self.backoff_s = backoff_s

    def guard(self, agent: SubAgent) -> Callable[[GraphState], Awaitable[Update]]:
        async def node(state: GraphState) -> Update:
            t0 = time.perf_counter()
            update = await _run(state)
            update["node_timings"] = [{"node": agent.name,
                                       "ms": round((time.perf_counter() - t0) * 1000, 1)}]
            return update

        async def _run(state: GraphState) -> Update:
            last: Optional[BaseException] = None
            for attempt in range(agent.node_retries + 1):
                try:
                    out = agent.run(state)
                    if inspect.isawaitable(out):
                        out = await out
                    return _with_log(out)
                except Exception as e:  # noqa: BLE001
                    last = e
                    record_heal(f"[{agent.name}] failed (attempt {attempt + 1}): "
                                f"{type(e).__name__}: {redact_secrets(e)[:200]}")
                    if attempt < agent.node_retries:
                        await asyncio.sleep(self.backoff_s * (2 ** attempt))
            assert last is not None
            fallback = agent.degrade(state, last)
            if fallback is None:
                record_heal(f"[{agent.name}] unrecoverable")
                raise last
            record_heal(f"[{agent.name}] degraded gracefully")
            return _with_log(fallback)

        node.__name__ = agent.name.replace(" ", "_").lower()
        return node


def _with_log(update: Update) -> Update:
    update = dict(update or {})
    events = drain_heal_log()
    if events:
        update["healing_log"] = events
    return update
