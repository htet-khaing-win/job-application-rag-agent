"""Statistics checked against hand-computed values (pure-Python implementations, so they must be right)."""
import math

import pytest

from evals.metrics import (aggregate_repeats, bootstrap_ci, cliffs_delta, cohens_dz, cohens_kappa, compare,
                           evaluate_run, flatten_rows, holm_correction, paired_permutation_pvalue, percentile,
                           wilcoxon_signed_rank_pvalue)


# ----------------------------------------------------------------- Wilcoxon signed-rank
def test_wilcoxon_exact_all_positive_n5():
    # W+ = 15 is the single most extreme outcome of 32 sign patterns; two-sided p = 2/32.
    assert wilcoxon_signed_rank_pvalue([0] * 5, [1, 2, 3, 4, 5]) == pytest.approx(0.0625)


def test_wilcoxon_exact_one_negative_n5():
    # diffs 1,2,3,4,-5 -> W+ = 10. Subset-sum counts of {1..5}: sums>=10 -> 10, sums<=5 -> 10, p = 20/32.
    assert wilcoxon_signed_rank_pvalue([0] * 5, [1, 2, 3, 4, -5]) == pytest.approx(0.625)


def test_wilcoxon_exact_with_ties_uses_midranks():
    # |d| = 1,1,2 -> ranks 1.5,1.5,3; all positive; doubled ranks 3,3,6; extreme = {all +, all -} -> 2/8.
    assert wilcoxon_signed_rank_pvalue([0, 0, 0], [1, 1, 2]) == pytest.approx(0.25)


def test_wilcoxon_drops_zero_differences():
    assert wilcoxon_signed_rank_pvalue([1, 2, 3], [1, 2, 3]) == 1.0
    # zeros are dropped: same as the n=5 all-positive case
    assert wilcoxon_signed_rank_pvalue([0] * 7, [1, 2, 3, 4, 5, 0, 0]) == pytest.approx(0.0625)


def test_wilcoxon_symmetric_in_sign_of_effect():
    a, b = [0] * 6, [1, 2, 3, 4, 5, 6]
    assert wilcoxon_signed_rank_pvalue(a, b) == pytest.approx(wilcoxon_signed_rank_pvalue(b, a))


def test_wilcoxon_normal_approximation_for_large_n():
    n = 30
    assert wilcoxon_signed_rank_pvalue([0] * n, list(range(1, n + 1))) < 1e-4  # every diff positive
    alternating = [i if i % 2 else -i for i in range(1, n + 1)]
    assert wilcoxon_signed_rank_pvalue([0] * n, alternating) > 0.5  # balanced signs -> no evidence


def test_wilcoxon_agrees_with_permutation_in_direction():
    a = [0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.1, 0.35]
    strong = [x + 0.4 for x in a]
    assert wilcoxon_signed_rank_pvalue(a, strong) < 0.05 and paired_permutation_pvalue(a, strong) < 0.05


def test_paired_tests_reject_unequal_lengths():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank_pvalue([1], [1, 2])
    with pytest.raises(ValueError):
        cohens_dz([1], [1, 2])


# ----------------------------------------------------------------- effect sizes, Holm, kappa
def test_cohens_dz_known_value():
    # diffs 1,2,3,4: mean 2.5, sample sd = sqrt(5/3)
    assert cohens_dz([0] * 4, [1, 2, 3, 4]) == pytest.approx(2.5 / math.sqrt(5 / 3))
    assert cohens_dz([1, 2, 3], [1, 2, 3]) == 0.0
    assert cohens_dz([0, 0, 0], [2, 2, 2]) == math.inf


def test_cliffs_delta_extremes():
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == 1.0
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == -1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_holm_known_values_and_order():
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5]) == [0.5]
    adj = holm_correction([0.2, 0.9, 0.001])
    assert all(a >= p for a, p in zip(adj, [0.2, 0.9, 0.001])) and all(a <= 1 for a in adj)


def test_cohens_kappa_known_table():
    # rows rater1, cols rater2: [[20, 5], [10, 15]] -> po = .7, pe = .5, kappa = .4
    r1 = [1] * 25 + [0] * 25
    r2 = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohens_kappa(r1, r2) == pytest.approx(0.4)


def test_cohens_kappa_perfect_chance_and_degenerate():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert cohens_kappa([1, 1], [1, 1]) == 1.0  # both constant and equal
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 0])


# ----------------------------------------------------------------- descriptive stats
def test_percentile_linear_interpolation():
    xs = [1, 2, 3, 4, 5]
    assert percentile(xs, 0) == 1 and percentile(xs, 50) == 3 and percentile(xs, 100) == 5
    assert percentile(xs, 95) == pytest.approx(4.8)
    assert percentile([], 50) == 0.0 and percentile([7], 95) == 7


def test_aggregate_repeats_mean_std_ci():
    agg = aggregate_repeats([{"ndcg": 0.5}, {"ndcg": 0.7}, {"ndcg": 0.9}])["ndcg"]
    assert agg["mean"] == pytest.approx(0.7) and agg["std"] == pytest.approx(0.2) and agg["n"] == 3
    assert 0.5 <= agg["ci_lo"] <= 0.7 <= agg["ci_hi"] <= 0.9


def test_aggregate_repeats_handles_missing_metric_in_a_repeat():
    agg = aggregate_repeats([{"a": 1.0, "b": 2.0}, {"a": 3.0}])
    assert agg["a"]["n"] == 2 and agg["b"]["n"] == 1 and agg["b"]["std"] == 0.0


def test_bootstrap_ci_is_seeded_and_brackets_mean():
    vals = [0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]
    m, lo, hi = bootstrap_ci(vals)
    assert lo <= m <= hi and bootstrap_ci(vals) == (m, lo, hi)
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------- compare()
def _gold(n):
    return {f"q{i}": [f"c{i}"] for i in range(n)}


def test_compare_flags_real_improvement_as_significant_after_holm():
    gold = _gold(12)
    bad = evaluate_run({q: ["x"] for q in gold}, gold, k=3)
    good = evaluate_run({q: [c[0]] for q, c in gold.items()}, gold, k=3)
    rows = compare(bad, good)
    mrr_row = next(r for r in rows if r["metric"] == "mrr")
    assert mrr_row["delta"] == 1.0 and mrr_row["significant"] is True
    assert mrr_row["p_wilcoxon"] < 0.001 and mrr_row["cliffs"] == 1.0
    assert all(r["p_holm"] >= r["p_wilcoxon"] for r in rows)
    assert math.isinf(mrr_row["dz"]) or mrr_row["dz"] > 1


def test_compare_identical_systems_not_significant():
    gold = _gold(8)
    run = evaluate_run({q: [c[0]] for q, c in gold.items()}, gold, k=3)
    rows = compare(run, run)
    assert all(r["delta"] == 0 and r["p_value"] == 1.0 and r["p_wilcoxon"] == 1.0 and not r["significant"]
               for r in rows)


def test_compare_small_noisy_difference_is_not_significant():
    gold = _gold(6)
    base = evaluate_run({q: [c[0]] for q, c in list(gold.items())} | {"q0": ["x"]}, gold, k=3)
    cand = evaluate_run({q: [c[0]] for q, c in list(gold.items())} | {"q1": ["x"]}, gold, k=3)
    assert not any(r["significant"] for r in compare(base, cand))


def test_flatten_rows_marks_direction_higher():
    gold = _gold(3)
    run = evaluate_run({q: [c[0]] for q, c in gold.items()}, gold, k=3)
    flat = flatten_rows(compare(run, run))
    assert flat["retrieval.mrr"] == {"value": 1.0, "direction": "higher"}
