"""Dependency-free retrieval metrics and statistics for evaluating the pipeline.

Pure Python on purpose: the eval harness must run where numpy/scipy are absent or broken.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

# ---------------------------------------------------------------------------- ranking metrics


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


# ---------------------------------------------------------------------------- descriptive stats


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: Sequence[float], ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


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


def aggregate_repeats(runs: Sequence[Mapping[str, float]], alpha: float = 0.05,
                      seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Aggregate several repeats of the same experiment.

    runs: [{metric: value}, ...] one dict per repeat. Returns {metric: {mean, std, ci_lo, ci_hi, n}}.
    The CI is a bootstrap over the repeats (honest about small n: it will be wide).
    """
    out: Dict[str, Dict[str, float]] = {}
    metrics = sorted({m for r in runs for m in r})
    for m in metrics:
        vals = [float(r[m]) for r in runs if m in r]
        mu, lo, hi = bootstrap_ci(vals, alpha=alpha, seed=seed)
        out[m] = {"mean": round(mu, 6), "std": round(std(vals), 6),
                  "ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "n": len(vals)}
    return out


# ---------------------------------------------------------------------------- paired tests


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


def _average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank test for paired samples (b vs a).

    Zero differences are dropped (Wilcoxon's convention). For n <= 20 non-zero differences the
    p-value is exact: the null distribution of W+ is built by dynamic programming over the
    (doubled, so integral) mid-ranks, which is exact even with ties. Beyond that, a normal
    approximation with tie and continuity corrections is used.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must be equal length")
    diffs = [y - x for x, y in zip(a, b) if y != x]
    n = len(diffs)
    if n == 0:
        return 1.0
    ranks = _average_ranks([abs(d) for d in diffs])
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean_w = n * (n + 1) / 4.0
    if n <= 20:
        doubled = [int(round(2 * r)) for r in ranks]
        dist: Dict[int, int] = {0: 1}
        for r2 in doubled:
            new: Dict[int, int] = dict(dist)
            for s, c in dist.items():
                new[s + r2] = new.get(s + r2, 0) + c
            dist = new
        total = 2 ** n
        obs = abs(2 * w_plus - 2 * mean_w)
        extreme = sum(c for s, c in dist.items() if abs(s - 2 * mean_w) >= obs - 1e-9)
        return min(1.0, extreme / total)
    ties = Counter(abs(d) for d in diffs)
    tie_corr = sum(t ** 3 - t for t in ties.values()) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var <= 0:
        return 1.0
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2)))


# ---------------------------------------------------------------------------- effect sizes


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired effect size: mean(b - a) / sd(b - a). +/-inf if all differences are identical."""
    if len(a) != len(b):
        raise ValueError("paired samples must be equal length")
    diffs = [y - x for x, y in zip(a, b)]
    if len(diffs) < 2:
        return 0.0
    sd = std(diffs)
    m = mean(diffs)
    if sd == 0:
        return 0.0 if m == 0 else math.copysign(math.inf, m)
    return m / sd


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Ordinal effect size in [-1, 1]: P(b > a) - P(b < a) over all cross pairs."""
    if not a or not b:
        return 0.0
    gt = sum(1 for x in a for y in b if y > x)
    lt = sum(1 for x in a for y in b if y < x)
    return (gt - lt) / (len(a) * len(b))


def holm_correction(pvalues: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def cohens_kappa(labels1: Sequence[Hashable], labels2: Sequence[Hashable]) -> float:
    """Cohen's kappa for two raters over the same items. 1.0 when both are constant and equal."""
    if len(labels1) != len(labels2):
        raise ValueError("label lists must be equal length")
    n = len(labels1)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(labels1, labels2) if x == y) / n
    c1, c2 = Counter(labels1), Counter(labels2)
    pe = sum((c1[k] / n) * (c2[k] / n) for k in set(c1) | set(c2))
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------- run comparison


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


def compare(base: Dict[str, List[float]], other: Dict[str, List[float]],
            alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Compare two systems metric-by-metric on the same queries.

    Each row carries: baseline/candidate means, bootstrap CI of the candidate mean, delta,
    permutation p (`p_value`), Wilcoxon p (`p_wilcoxon`), effect sizes (`dz`, `cliffs`),
    Holm-adjusted Wilcoxon p across all metrics (`p_holm`) and `significant` (p_holm < alpha).
    """
    rows: List[Dict[str, Any]] = []
    for metric in base:
        m, lo, hi = bootstrap_ci(other[metric])
        dz = cohens_dz(base[metric], other[metric])
        rows.append({
            "metric": metric,
            "baseline": round(mean(base[metric]), 4),
            "candidate": round(m, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "delta": round(m - mean(base[metric]), 4),
            "p_value": round(paired_permutation_pvalue(base[metric], other[metric]), 4),
            "p_wilcoxon": round(wilcoxon_signed_rank_pvalue(base[metric], other[metric]), 4),
            "dz": round(dz, 4) if math.isfinite(dz) else dz,
            "cliffs": round(cliffs_delta(base[metric], other[metric]), 4),
        })
    adj = holm_correction([r["p_wilcoxon"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = round(p, 4)
        r["significant"] = p < alpha
    return rows


def flatten_rows(rows: Iterable[Mapping[str, Any]], prefix: str = "retrieval") -> Dict[str, Dict[str, Any]]:
    """Rows from compare() -> flat {name: {value, direction}} for regression tracking."""
    return {f"{prefix}.{r['metric']}": {"value": r["candidate"], "direction": "higher"} for r in rows}
