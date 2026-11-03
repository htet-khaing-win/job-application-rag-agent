"""Dependency-free retrieval metrics and statistics for evaluating the pipeline."""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(relevant) & set(ranked[:k])) / k


def mrr(ranked: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, doc in enumerate(ranked, 1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean(values), lo, hi


def paired_permutation_pvalue(a: Sequence[float], b: Sequence[float], n_perm: int = 5000,
                              seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on mean difference (b - a)."""
    if len(a) != len(b):
        raise ValueError("paired samples must be equal length")
    diffs = [y - x for x, y in zip(a, b)]
    if not diffs or all(d == 0 for d in diffs):
        return 1.0
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        if abs(mean([d if rng.random() < 0.5 else -d for d in diffs])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def evaluate_run(runs: Dict[str, List[str]], gold: Dict[str, List[str]], k: int = 5) -> Dict[str, List[float]]:
    """Per-query metric lists. runs/gold: {query_id: [chunk_ids]}."""
    out: Dict[str, List[float]] = {f"recall@{k}": [], f"precision@{k}": [], "mrr": [], f"ndcg@{k}": []}
    for qid, relevant in gold.items():
        ranked = runs.get(qid, [])
        out[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
        out[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
        out["mrr"].append(mrr(ranked, relevant))
        out[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
    return out


def compare(base: Dict[str, List[float]], other: Dict[str, List[float]]) -> List[Dict[str, float]]:
    rows = []
    for metric in base:
        m, lo, hi = bootstrap_ci(other[metric])
        rows.append({
            "metric": metric,
            "baseline": round(mean(base[metric]), 4),
            "candidate": round(m, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "delta": round(m - mean(base[metric]), 4),
            "p_value": round(paired_permutation_pvalue(base[metric], other[metric]), 4),
        })
    return rows
