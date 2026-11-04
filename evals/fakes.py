"""Deterministic fake chat models and graph backend stubs, shared by the chaos / ablation harnesses
and by tests. Nothing here talks to a network."""
from __future__ import annotations

import contextlib
import json
import random
from typing import Any, Callable, Iterator, List, Optional, Tuple

SECRET = "sk-PLANTEDSECRETKEYabcdefghijkl0123"

JD = ("Requirements: Python and Kubernetes experience. Responsibilities: build and operate backend services, "
      "collaborate with product and design, and mentor juniors. Qualifications: BSc in computer science or "
      "equivalent practical experience shipping production software.")
RESUME = "Built python services handling 5000 requests per second. Mentored 3 junior engineers. BSc Computer Science."
GOOD_LETTER = "I built python services handling 5000 requests per second."


class Msg:
    def __init__(self, content: str):
        self.content = content


class ScriptedModel:
    """Routes on prompt content like the real pipeline's prompts. `letter` is what the writer returns."""

    def __init__(self, letter: str = GOOD_LETTER, critique: Optional[str] = None, model_id: str = "fake-model"):
        self.letter = letter
        self.critique = critique if critique is not None else json.dumps({"edits": [], "ready_to_submit": True})
        self.model_id = model_id
        self.calls = 0

    def invoke(self, prompt: Any, *a: Any, **k: Any) -> Msg:
        self.calls += 1
        p = str(prompt)
        if "not valid JSON" in p:
            return Msg(json.dumps({"valid": True, "cleaned_jd": JD}))
        if "processing a job description" in p:
            return Msg(json.dumps({"valid": True, "cleaned_jd": JD}))
        if "Write a factual cover letter" in p:
            return Msg(self.letter)
        if "Rewrite ONLY the listed sentences" in p:
            return Msg("I built python services. I am prepared to learn Kubernetes via my Python background.")
        if "Hiring Manager" in p:
            return Msg(self.critique)
        return Msg("ok")


FaultFn = Callable[[int, str], Optional[Tuple[str, Any]]]


class FaultyModel:
    """Wraps a model; `fault(call_index, prompt)` returns None (pass through),
    ("raise", exc) or ("return", Msg)."""

    def __init__(self, inner: Any, fault: FaultFn, model_id: str = "faulty-model"):
        self.inner, self.fault, self.model_id, self.calls = inner, fault, model_id, 0

    def invoke(self, prompt: Any, *a: Any, **k: Any) -> Any:
        i = self.calls
        self.calls += 1
        f = self.fault(i, str(prompt))
        if f is None:
            return self.inner.invoke(prompt, *a, **k)
        kind, payload = f
        if kind == "raise":
            raise payload
        return payload


def always(exc: BaseException) -> FaultFn:
    return lambda i, p: ("raise", exc)


def first_n(n: int, exc: BaseException) -> FaultFn:
    return lambda i, p: ("raise", exc) if i < n else None


def with_probability(p_fail: float, exc: BaseException, seed: int) -> FaultFn:
    rng = random.Random(seed)
    return lambda i, p: ("raise", exc) if rng.random() < p_fail else None


def malformed_json_once() -> FaultFn:
    """Return broken JSON the first time a JSON-expecting prompt is seen (the repair prompt passes)."""
    seen = {"n": 0}

    def fault(i: int, prompt: str) -> Optional[Tuple[str, Any]]:
        wants_json = "processing a job description" in prompt or "Hiring Manager" in prompt
        if wants_json and "not valid JSON" not in prompt and seen["n"] < 2:
            seen["n"] += 1
            return ("return", Msg("{broken json"))
        return None

    return fault


@contextlib.contextmanager
def stub_backends(keywords: Tuple[str, ...] = ("python", "kubernetes"),
                  chunks: Optional[list] = None, research_text: str = "Acme builds things") -> Iterator[None]:
    """Replace Pinecone retrieval, Tavily research and TF-IDF keyword extraction with fixed fakes."""
    from unittest import mock

    import node
    import subagents

    chunk_list = chunks if chunks is not None else [
        {"text": RESUME, "source": "r.pdf", "score": 0.9, "chunk_id": "r.pdf#0"}]

    async def retrieve(state: Any) -> dict:
        return {"retrieved_chunks": chunk_list, "vector_relevance_score": 90.0}

    async def research(state: Any) -> dict:
        return {"company_research": research_text, "company_research_success": True}

    patches: List[Any] = [mock.patch.object(node, "extract_jd_keywords", lambda jd, top_k=20: list(keywords))]
    for name, fn in (("retrieve_resumes_node", retrieve), ("research_company_node", research)):
        if hasattr(subagents, name):
            patches.append(mock.patch.object(subagents, name, fn))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield
