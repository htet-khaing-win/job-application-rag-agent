"""Gold-set tooling.

IMPORTANT: this module builds the *sheet* a human labels and turns the finished sheet into
`gold_set.json`. The labelling itself (is this chunk evidence for this JD? does the resume support
this skill?) is human judgement and cannot be automated honestly - an LLM-labelled gold set would
only measure agreement with that LLM. Aim for 30-50 queries, ideally two annotators.

  python -m evals label make-sheet --jds evals/jds.json --top-n 10          (live; vault keys)
  ... a human sets every "relevant" (true/false) and "supported" (true/false) ...
  python -m evals label build-gold --sheets sheet_a.json [sheet_b.json]     (>=2 sheets -> kappa)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from evals.metrics import cohens_kappa

SHEET_VERSION = 1
MIN_QUERIES = 30

INSTRUCTIONS = ("For every chunk set \"relevant\" to true if it is direct evidence for a requirement of the JD, "
                "else false. For every skill set \"supported\" to true only if the whole resume states it. "
                "Do not leave null. Do not edit chunk ids or text.")

_STOP = frozenset(
    "the and for with are you our your will have has from this that their they them who what which into over such "
    "than then also can may per via able work working team role candidate experience skills required requirements "
    "responsibilities strong good excellent ability knowledge years year including using use build develop manage "
    "support".split())


class LabelError(ValueError):
    pass


Fetcher = Callable[[str, str, int], List[Dict[str, str]]]


def candidate_skills(jd: str, limit: int = 15) -> List[str]:
    """Candidate skill tokens for the annotator to confirm (frequency over simple tokens)."""
    toks = [t.lower().strip(".-") for t in re.findall(r"[A-Za-z][A-Za-z0-9+#.\-]{2,}", jd)]
    counts = Counter(t for t in toks if t not in _STOP)
    return [t for t, _ in counts.most_common(limit)]


def make_sheet(jds: Sequence[Mapping[str, str]], namespaces: Sequence[str], fetch: Fetcher, top_n: int = 10,
               skills_fn: Callable[[str], List[str]] = candidate_skills) -> Dict[str, Any]:
    """jds: [{id, jd}]. fetch(jd, namespace, n) -> [{chunk_id, text}] (top-n candidates, any order)."""
    items = []
    for j in jds:
        for ns in namespaces:
            chunks = fetch(j["jd"], ns, top_n)
            items.append({
                "query_id": str(j["id"]), "jd": j["jd"], "namespace": ns,
                "chunks": [{"chunk_id": c["chunk_id"], "text": c["text"], "relevant": None} for c in chunks],
                "skills": [{"skill": s, "supported": None} for s in skills_fn(j["jd"])],
            })
    return {"version": SHEET_VERSION, "instructions": INSTRUCTIONS, "annotator": "", "items": items}


def live_fetcher(rt: Any) -> Fetcher:
    from database import get_embeddings, get_pinecone_index
    from runtime import activate_runtime
    activate_runtime(rt)
    emb, index = get_embeddings(), get_pinecone_index()

    def fetch(jd: str, namespace: str, n: int) -> List[Dict[str, str]]:
        res = index.query(vector=emb.embed_query(jd), top_k=n, include_metadata=True, namespace=namespace)
        return [{"chunk_id": m["id"], "text": m["metadata"].get("text", "")} for m in res["matches"]]

    return fetch


def validate_sheet(sheet: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). Errors block build-gold."""
    errors: List[str] = []
    warnings: List[str] = []
    if sheet.get("version") != SHEET_VERSION:
        errors.append(f"unsupported sheet version {sheet.get('version')!r}")
    seen = set()
    for it in sheet.get("items", []):
        key = (it.get("query_id"), it.get("namespace"))
        if key in seen:
            errors.append(f"duplicate item {key}")
        seen.add(key)
        ids = [c["chunk_id"] for c in it.get("chunks", [])]
        if len(ids) != len(set(ids)):
            errors.append(f"{key}: duplicate chunk ids")
        for c in it.get("chunks", []):
            if not isinstance(c.get("relevant"), bool):
                errors.append(f"{key}: chunk {c.get('chunk_id')} is unlabelled (relevant must be true/false)")
        for s in it.get("skills", []):
            if not isinstance(s.get("supported"), bool):
                errors.append(f"{key}: skill {s.get('skill')!r} is unlabelled (supported must be true/false)")
    if not sheet.get("items"):
        errors.append("sheet has no items")
    return errors, warnings


def _index(sheet: Mapping[str, Any]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(it["query_id"], it["namespace"]): it for it in sheet.get("items", [])}


def agreement(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Inter-annotator agreement on chunk relevance and skill support between two labelled sheets."""
    ia, ib = _index(a), _index(b)
    shared = sorted(set(ia) & set(ib))
    chunk_a: List[bool] = []
    chunk_b: List[bool] = []
    skill_a: List[bool] = []
    skill_b: List[bool] = []
    disagreements: List[Dict[str, Any]] = []
    for key in shared:
        ca = {c["chunk_id"]: c["relevant"] for c in ia[key]["chunks"]}
        cb = {c["chunk_id"]: c["relevant"] for c in ib[key]["chunks"]}
        for cid in sorted(set(ca) & set(cb)):
            chunk_a.append(ca[cid]); chunk_b.append(cb[cid])
            if ca[cid] != cb[cid]:
                disagreements.append({"query": key[0], "namespace": key[1], "type": "chunk", "id": cid})
        sa = {s["skill"]: s["supported"] for s in ia[key]["skills"]}
        sb = {s["skill"]: s["supported"] for s in ib[key]["skills"]}
        for sk in sorted(set(sa) & set(sb)):
            skill_a.append(sa[sk]); skill_b.append(sb[sk])
            if sa[sk] != sb[sk]:
                disagreements.append({"query": key[0], "namespace": key[1], "type": "skill", "id": sk})

    def stats(x: List[bool], y: List[bool]) -> Dict[str, Any]:
        n = len(x)
        return {"n": n, "agreement": round(sum(1 for p, q in zip(x, y) if p == q) / n, 4) if n else None,
                "kappa": round(cohens_kappa(x, y), 4) if n else None}

    return {"shared_items": len(shared), "chunk_relevance": stats(chunk_a, chunk_b),
            "skill_support": stats(skill_a, skill_b), "disagreements": disagreements}


def build_gold(sheets: Sequence[Mapping[str, Any]], adjudicate: str = "error",
               min_queries: int = MIN_QUERIES) -> Dict[str, Any]:
    """Turn finished sheets into a gold set.

    adjudicate: how to resolve annotator disagreements when two sheets are given -
      'error' (refuse; resolve them by hand), 'first' (annotator 1 wins), 'intersection' (relevant only if both agree).
    """
    if not sheets:
        raise LabelError("no sheets given")
    for i, s in enumerate(sheets):
        errs, _ = validate_sheet(s)
        if errs:
            raise LabelError(f"sheet {i + 1} is not fully labelled: " + "; ".join(errs[:8])
                             + (f" (+{len(errs) - 8} more)" if len(errs) > 8 else ""))
    agree = agreement(sheets[0], sheets[1]) if len(sheets) > 1 else None
    if agree and agree["disagreements"] and adjudicate == "error":
        raise LabelError(f"{len(agree['disagreements'])} annotator disagreements; adjudicate them or pass "
                         "adjudicate='first' / 'intersection'. First few: " + json.dumps(agree["disagreements"][:5]))
    second = _index(sheets[1]) if len(sheets) > 1 else {}
    queries, dropped, warnings = [], [], []
    for key, it in _index(sheets[0]).items():
        other = second.get(key)
        o_chunks = {c["chunk_id"]: c["relevant"] for c in other["chunks"]} if other else {}
        o_skills = {s["skill"]: s["supported"] for s in other["skills"]} if other else {}

        def pick(mine: bool, theirs: Optional[bool]) -> bool:
            if theirs is None or adjudicate == "first":
                return mine
            return mine and theirs  # intersection (also the agreed value under 'error')

        relevant = [c["chunk_id"] for c in it["chunks"] if pick(c["relevant"], o_chunks.get(c["chunk_id"]))]
        supported = [s["skill"] for s in it["skills"] if pick(s["supported"], o_skills.get(s["skill"]))]
        unsupported = [s["skill"] for s in it["skills"] if s["skill"] not in supported]
        if not relevant:
            dropped.append(f"{key[0]}::{key[1]}")
            continue
        queries.append({"id": f"{key[0]}::{key[1]}", "jd": it["jd"], "namespace": key[1],
                        "relevant_chunk_ids": relevant, "supported_skills": supported,
                        "unsupported_skills": unsupported})
    if not queries:
        raise LabelError("no query has a relevant chunk; nothing to evaluate")
    if dropped:
        warnings.append(f"dropped {len(dropped)} item(s) with no relevant chunk (recall undefined): {dropped[:5]}")
    if len(queries) < min_queries:
        warnings.append(f"only {len(queries)} labelled queries (< {min_queries}); confidence intervals will be wide")
    return {"version": SHEET_VERSION, "queries": queries,
            "meta": {"n_annotators": len(sheets), "adjudicate": adjudicate, "agreement": agree,
                     "dropped_no_relevant": dropped, "warnings": warnings}}


def load(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(obj: Mapping[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
