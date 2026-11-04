"""Span-based retrieval metrics.

Gold labels from `evals.goldgen` mark *evidence spans* (resume lines) instead of chunk ids, so they
survive re-chunking and re-indexing. A retrieved chunk is relevant if it contains a span; a span is
"covered" if any chunk in the top-k contains it.

  recall@k    = covered spans / all spans
  precision@k = relevant chunks in top-k / k
  mrr         = 1 / rank of the first relevant chunk
  ndcg@k      = binary-gain nDCG with the ideal list taken as min(#spans, k) relevant chunks (an upper
                bound, because several spans can live in one chunk, so it is slightly conservative)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Sequence

_TOKEN = re.compile(r"[a-z0-9+#]+")
MATCH_THRESHOLD = 0.8


def tokens(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def span_in_chunk(span: str, chunk_text: str, threshold: float = MATCH_THRESHOLD) -> bool:
    """True if at least `threshold` of the span's tokens occur in the chunk.

    Order-insensitive and whitespace/redaction tolerant: PII redaction or a different PDF extractor
    changes spacing, not the words.
    """
    st = tokens(span)
    if len(st) < 3:
        return False
    ct = set(tokens(chunk_text))
    return sum(t in ct for t in st) / len(st) >= threshold


def query_metrics(chunks: Sequence[Mapping[str, str]], spans: Sequence[str], k: int) -> Dict[str, float]:
    top = list(chunks)[:k]
    rel = [any(span_in_chunk(s, c.get("text", "")) for s in spans) for c in top]
    covered = sum(any(span_in_chunk(s, c.get("text", "")) for c in top) for s in spans)
    first = next((i for i, r in enumerate(rel, 1) if r), 0)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(spans), k)))
    return {
        f"recall@{k}": covered / len(spans) if spans else 0.0,
        f"precision@{k}": sum(rel) / k if k else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
    }


def evaluate_run_spans(runs: Mapping[str, Sequence[Mapping[str, str]]], gold_queries: Sequence[Mapping[str, Any]],
                       k: int = 5) -> Dict[str, List[float]]:
    """Per-query metric lists over the scorable queries (those with at least one span)."""
    out: Dict[str, List[float]] = {f"recall@{k}": [], f"precision@{k}": [], "mrr": [], f"ndcg@{k}": []}
    for q in gold_queries:
        spans = q.get("relevant_spans") or []
        if not spans:
            continue
        m = query_metrics(runs.get(q["id"], []), spans, k)
        for key in out:
            out[key].append(m[key])
    return out
