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
corrected across all (dataset, architecture, pair) comparisons. The pairs are
the surrogates S-BORF, S+-BORF, and S-SDRF and the uncertified controls JL-BORF
and SDRF with BF (``NONINF``). The exact methods BORF and OR-SDRF are compared
with the original graph by the two-sided version of the same test, with Holm
correction across their own comparisons (``TWO_SIDED``).

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

# (surrogate or uncertified control, exact counterpart, label, LaTeX label, short key)
NONINF = [("BORF (Bounds)", "BORF (OR)", "S-BORF vs BORF", "S-BORF vs BORF", "SBORF"),
          ("BORF (Sharpened)", "BORF (OR)", "S+-BORF vs BORF", "S$^{+}$-BORF vs BORF", "SplusBORF"),
          ("SDRF (Bounds)", "SDRF (OR)", "S-SDRF vs OR-SDRF", "S-SDRF vs OR-SDRF", "SSDRF"),
          ("BORF (JL)", "BORF (OR)", "JL-BORF vs BORF", "JL-BORF vs BORF", "JLBORF"),
          ("SDRF (BF)", "SDRF (OR)", "SDRF (BF) vs OR-SDRF", "SDRF (BF) vs OR-SDRF", "SDRFBF")]
# exact methods against the original graph: two-sided version of the same test
TWO_SIDED = [("BORF (OR)", "Original", "BORF vs Original", "BORF vs Original", "BORFOriginal"),
             ("SDRF (OR)", "Original", "OR-SDRF vs Original", "OR-SDRF vs Original", "ORSDRFOriginal")]
PAIRS = [(s, r) for s, r, *_ in NONINF]
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


def two_sided(d, K, test="corrected"):
    """Two-sided test of a zero mean fold-level difference, with the variance of
    :func:`noninferiority`. Returns (p, mean difference)."""
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    var = d.var(ddof=1)
    if test == "corrected":
        if K is None or K < 2:
            raise ValueError("the corrected test needs K >= 2 folds")
        se = math.sqrt((1.0 / n + 1.0 / (K - 1)) * var)
    elif test == "paired":
        se = math.sqrt(var / n)
    else:
        raise ValueError(test)
    m = float(d.mean())
    if se == 0:
        return (0.0 if m != 0 else 1.0), m
    return float(2 * (1 - stats.t.cdf(abs(m) / se, n - 1))), m


def case_study_tests(res, datasets, archs, delta=None, alpha=0.05, test="corrected"):
    """Every test of Appendix A.

    ``datasets`` is a list of (key, name, metric) with metric "mae" for the data
    sets where lower is better. Non-inferiority tests (``NONINF``, needs
    ``delta``) and two-sided tests (``TWO_SIDED``) are Holm-corrected within
    their own family. Returns one dict per test: dataset, arch, kind
    ("noninferiority" or "two-sided"), label, tex, key, surrogate, reference,
    r (repetitions), K, mean, lcb (NaN for two-sided tests), p, p_holm,
    significant.
    """
    fams = {"noninferiority": [], "two-sided": []}
    for ds, name, metric in datasets:
        hb = metric != "mae"
        for arch in archs:
            entry = res.get(f"{ds}_{arch}", {})
            for kind, pairs in (("noninferiority", NONINF), ("two-sided", TWO_SIDED)):
                if kind == "noninferiority" and delta is None:
                    continue
                for sur, ref, lab, tex, key in pairs:
                    if sur not in entry or ref not in entry:
                        continue
                    try:
                        d, K, r = fold_differences(entry[sur], entry[ref], hb)
                    except ValueError:
                        continue
                    if kind == "noninferiority":
                        p, m, lcb = noninferiority(d, K, float(delta), alpha, test)
                    else:
                        (p, m), lcb = two_sided(d, K, test), float("nan")
                    if math.isfinite(p):
                        fams[kind].append(dict(dataset=name, arch=arch, kind=kind, label=lab, tex=tex, key=key,
                                               surrogate=sur, reference=ref, r=r, K=K, mean=m, lcb=lcb, p=p))
    out = []
    for kind in ("noninferiority", "two-sided"):
        rows = fams[kind]
        if rows:
            for row, pa in zip(rows, holm([x["p"] for x in rows])):
                row["p_holm"] = float(pa)
                row["significant"] = bool(pa < alpha)
                out.append(row)
    return out


def verdict(row):
    if row["kind"] == "noninferiority":
        return "non-inferior" if row["significant"] else "not shown"
    return "differs" if row["significant"] else "no difference shown"


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
    keys = sorted(res)
    dsets = list(dict.fromkeys(k.rsplit("_", 1)[0] for k in keys))
    archs = list(dict.fromkeys(k.rsplit("_", 1)[1] for k in keys if "_" in k))
    datasets = [(ds, ds, "mae" if ds in LOWER_IS_BETTER_DATASETS else "max") for ds in dsets]
    rows = case_study_tests(res, datasets, archs, a.delta, a.alpha, a.test)
    if not rows:
        print("no testable pairs"); return
    print(f"test={a.test}  delta={a.delta}  alpha={a.alpha}  (positive difference = surrogate better)")
    for kind in ("noninferiority", "two-sided"):
        sel = [x for x in rows if x["kind"] == kind]
        if not sel:
            continue
        print(f"\n{kind} tests (Holm across {len(sel)} comparisons)")
        for x in sel:
            lcb = f"LCB{int(100 * (1 - a.alpha))}={x['lcb']:+.4f} " if kind == "noninferiority" else ""
            print(f"{x['dataset'] + '_' + x['arch']:28s} {x['surrogate']:17s} vs {x['reference']:10s} r={x['r']:3d} "
                  f"K={x['K']} diff={x['mean']:+.4f} {lcb}p={x['p']:.4f} p_holm={x['p_holm']:.4f} {verdict(x)}")


if __name__ == "__main__":
    main()
