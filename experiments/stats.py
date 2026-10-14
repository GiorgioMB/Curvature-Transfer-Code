#!/usr/bin/env python3
"""Non-inferiority tests for the rewiring case study (Appendix A).

Design of gnn_experiments.py: for every pipeline, repetition r uses the seed
``seed + EVAL_SEED_OFFSET + r`` both for model initialization and for a fresh
shuffled K-fold partition (``KFold(shuffle=True, random_state=seed)``). Given
the dataset, the partition depends only on the seed, so repetition r of a
surrogate pipeline and repetition r of its exact counterpart see the same K
train/test splits, and the evaluation is *repeated random K-fold CV*.

Test (default, ``corrected``): the corrected repeated k-fold cv test of
Bouckaert & Frank (2004), which applies the variance correction of Nadeau &
Bengio (2003) to the r*K paired fold-level differences d_{r,k}:

    t = (mean(d) + delta) / sqrt((1/(rK) + n_test/n_train) * var(d)),
    n_test/n_train = 1/(K-1),   df = rK - 1,

with H0: the surrogate is worse than the exact method by at least delta. The
reported one-sided lower confidence bound uses the same corrected variance, so
"non-inferior at level alpha" <=> lower bound > -delta. p-values are Holm-
corrected across all (dataset, architecture, pair) comparisons.

``--test paired`` is the uncorrected paired t-test on fold-level differences.
It is too liberal here, because training sets overlap across folds and
repetitions; it is kept for reference only.

Pairing requires the "runs" field (repetition index + per-fold scores)
written by the current gnn_experiments.py. Older result files store only the
per-repetition means, in completion order; they cannot be paired, and this
script refuses to test them rather than produce an invalid p-value.

Caveat: hyperparameters are selected by Optuna on CV folds of the same data
(not nested CV), so the inference is conditional on the selected
configuration of each pipeline.

Usage: python stats.py out/<run>/gnn_results.json --delta 0.01 [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy import stats

PAIRS = [("BORF (Bounds)", "BORF (OR)"), ("SDRF (Bounds)", "SDRF (OR)"),
         ("BORF (Sharpened)", "BORF (OR)"), ("BORF (JL)", "BORF (OR)")]
LOWER_IS_BETTER_DATASETS = {"ZINC", "QM9"}  # MAE; all other metrics are maximized


def fold_differences(sur_entry, ref_entry, higher_better=True):
    """Paired fold-level differences (positive = surrogate better), matched on
    (repetition index, fold index). Returns (d, K, r_used)."""
    if "runs" not in sur_entry or "runs" not in ref_entry:
        raise ValueError("result entries lack per-repetition fold scores ('runs'); "
                         "re-run with the current gnn_experiments.py")
    sr = {r["rep"]: r for r in sur_entry["runs"]}
    rr = {r["rep"]: r for r in ref_entry["runs"]}
    common = sorted(set(sr) & set(rr))
    d = []
    K = None
    for rep in common:
        a, b = sr[rep], rr[rep]
        if "seed" in a and "seed" in b and a["seed"] != b["seed"]:
            raise ValueError(f"repetition {rep} used different seeds in the two pipelines")
        fa, fb = a["folds"], b["folds"]
        if len(fa) != len(fb):
            raise ValueError(f"repetition {rep}: different numbers of folds")
        K = len(fa) if K is None else K
        for x, y in zip(fa, fb):
            d.append((x - y) if higher_better else (y - x))
    return np.asarray(d, float), K, len(common)


def noninferiority(d, K, delta, alpha=0.05, test="corrected"):
    """One-sided non-inferiority test on fold-level differences d (length r*K).
    Returns (p, mean difference, one-sided (1-alpha) lower confidence bound)."""
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    var = d.var(ddof=1)
    if test == "corrected":
        if K is None or K < 2:
            raise ValueError("the corrected test needs K >= 2 folds")
        se = math.sqrt((1.0 / n + 1.0 / (K - 1)) * var)
    elif test == "paired":
        se = math.sqrt(var / n)
    else:
        raise ValueError(test)
    df = n - 1
    m = float(d.mean())
    if se == 0:
        return (0.0 if m + delta > 0 else 1.0), m, m
    t = (m + delta) / se
    p = float(1 - stats.t.cdf(t, df))
    lcb = m - float(stats.t.ppf(1 - alpha, df)) * se
    return p, m, lcb


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(pvals) - r) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--delta", type=float, required=True,
                    help="non-inferiority margin in metric units, fixed before the experiments")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--test", choices=["corrected", "paired"], default="corrected")
    a = ap.parse_args()
    res = json.load(open(a.results))
    rows = []
    for key, pipes in res.items():
        hb = key.split("_")[0] not in LOWER_IS_BETTER_DATASETS
        for sur, ref in PAIRS:
            if sur not in pipes or ref not in pipes:
                continue
            try:
                d, K, r = fold_differences(pipes[sur], pipes[ref], hb)
            except ValueError as err:
                print(f"[skip] {key}: {sur} vs {ref}: {err}")
                continue
            p, m, lcb = noninferiority(d, K, a.delta, a.alpha, a.test)
            rows.append((key, sur, ref, r, K, m, lcb, p))
    if not rows:
        print("no testable pairs"); return
    adj = holm([x[-1] for x in rows])
    print(f"test={a.test}  delta={a.delta}  alpha={a.alpha}  (positive difference = surrogate better)")
    for (key, sur, ref, r, K, m, lcb, p), pa in zip(rows, adj):
        verdict = "non-inferior" if pa < a.alpha else "not shown"
        print(f"{key:28s} {sur:17s} vs {ref:10s} r={r:3d} K={K} diff={m:+.4f} "
              f"LCB{int(100 * (1 - a.alpha))}={lcb:+.4f} p={p:.4f} p_holm={pa:.4f} {verdict}")


if __name__ == "__main__":
    main()
