"""Hybrid-weight (alpha) sweep with cross-validated selection (`python -m evals sweep`).

alpha is the dense weight: score = alpha * dense + (1 - alpha) * sparse (alpha=1 is dense-only).
Picking the best alpha on the same queries you report is optimistic, so selection is done k-fold:
for each fold the alpha is chosen on the training queries and scored on the held-out ones. The gap
between the in-sample best and the cross-validated score (`optimism`) is reported.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from evals.metrics import bootstrap_ci, mean, ndcg_at_k, precision_at_k, recall_at_k, mrr

Retriever = Callable[[Mapping[str, Any], float, int], List[str]]

CAVEAT = ("Best-alpha-in-sample is optimistically biased (selected and scored on the same queries); "
          "trust cv_ndcg. With fewer than ~30 queries the choice of alpha is mostly noise.")


def kfold_indices(n: int, k: int, seed: int = 0) -> List[List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = max(2, min(k, n)) if n >= 2 else 1
    return [idx[i::k] for i in range(k)]


def sweep_alpha(gold: Mapping[str, Any], alphas: Sequence[float], k: int, retriever: Retriever,
                folds: int = 5, seed: int = 0) -> Dict[str, Any]:
    """gold: {"queries": [{id, jd, namespace, relevant_chunk_ids}]}."""
    queries = list(gold["queries"])
    n = len(queries)
    if n == 0:
        raise ValueError("gold set has no queries")
    per_alpha: Dict[float, Dict[str, List[float]]] = {}
    for a in alphas:
        cols: Dict[str, List[float]] = {"ndcg": [], "recall": [], "precision": [], "mrr": []}
        for q in queries:
            ranked = retriever(q, a, k)
            rel = q["relevant_chunk_ids"]
            cols["ndcg"].append(ndcg_at_k(ranked, rel, k))
            cols["recall"].append(recall_at_k(ranked, rel, k))
            cols["precision"].append(precision_at_k(ranked, rel, k))
            cols["mrr"].append(mrr(ranked, rel))
        per_alpha[a] = cols

    table = []
    for a in alphas:
        m, lo, hi = bootstrap_ci(per_alpha[a]["ndcg"], seed=seed)
        table.append({"alpha": a, f"ndcg@{k}": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      f"recall@{k}": round(mean(per_alpha[a]["recall"]), 4),
                      f"precision@{k}": round(mean(per_alpha[a]["precision"]), 4),
                      "mrr": round(mean(per_alpha[a]["mrr"]), 4)})
    best_in = max(alphas, key=lambda a: (mean(per_alpha[a]["ndcg"]), -abs(a - 0.5)))

    held: List[Optional[float]] = [None] * n
    chosen: List[float] = []
    if n >= 4:
        for test in kfold_indices(n, folds, seed):
            test_set = set(test)
            train = [i for i in range(n) if i not in test_set]
            pick = max(alphas, key=lambda a: (mean([per_alpha[a]["ndcg"][i] for i in train]), -abs(a - 0.5)))
            chosen.append(pick)
            for i in test:
                held[i] = per_alpha[pick]["ndcg"][i]
    scored = [h for h in held if h is not None]
    cv_m, cv_lo, cv_hi = bootstrap_ci(scored, seed=seed) if scored else (0.0, 0.0, 0.0)
    in_sample = mean(per_alpha[best_in]["ndcg"])
    return {
        "k": k, "n_queries": n, "table": table,
        "best_alpha_in_sample": best_in, "in_sample_ndcg": round(in_sample, 4),
        "cv_ndcg": round(cv_m, 4) if scored else None,
        "cv_ci": [round(cv_lo, 4), round(cv_hi, 4)] if scored else None,
        "optimism": round(in_sample - cv_m, 4) if scored else None,
        "cv_selected_alphas": chosen,
        "caveat": CAVEAT if n >= 30 else CAVEAT + f" (n={n}: below 30; treat as exploratory.)",
    }


def live_retriever(rt: Any) -> Retriever:
    """Weighted hybrid retrieval against Pinecone (needs vault keys).

    Pinecone has no `alpha` query argument; the documented approach is to scale the query vectors,
    which is what this does (dense * alpha, sparse * (1 - alpha)).
    """
    from database import get_bm25, get_embeddings, get_pinecone_index
    from runtime import activate_runtime
    activate_runtime(rt)
    emb, bm25, index = get_embeddings(), get_bm25(), get_pinecone_index()
    cache: Dict[str, Any] = {}

    def retrieve(q: Mapping[str, Any], alpha: float, k: int) -> List[str]:
        key = q["id"]
        if key not in cache:
            cache[key] = (emb.embed_query(q["jd"]), bm25.encode_queries(q["jd"]))
        dense, sparse = cache[key]
        res = index.query(vector=[v * alpha for v in dense],
                          sparse_vector={"indices": sparse["indices"],
                                         "values": [v * (1 - alpha) for v in sparse["values"]]},
                          top_k=k, include_metadata=False, namespace=q["namespace"])
        return [m["id"] for m in res["matches"]]

    return retrieve


def format_table(result: Mapping[str, Any]) -> str:
    k = result["k"]
    rows = [f"{'alpha':>6}{'ndcg@%d' % k:>10}{'95% CI':>20}{'recall':>9}{'prec':>8}{'mrr':>8}"]
    for r in result["table"]:
        rows.append(f"{r['alpha']:>6}{r[f'ndcg@{k}']:>10}{'[%s, %s]' % (r['ci_lo'], r['ci_hi']):>20}"
                    f"{r[f'recall@{k}']:>9}{r[f'precision@{k}']:>8}{r['mrr']:>8}")
    rows += ["", f"best alpha (in-sample): {result['best_alpha_in_sample']}  ndcg={result['in_sample_ndcg']}"]
    if result["cv_ndcg"] is not None:
        rows.append(f"cross-validated ndcg:   {result['cv_ndcg']} {result['cv_ci']}  optimism={result['optimism']}")
        rows.append(f"alphas chosen per fold: {result['cv_selected_alphas']}")
    rows += ["", "NOTE: " + result["caveat"]]
    return "\n".join(rows)
