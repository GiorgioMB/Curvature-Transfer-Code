#!/usr/bin/env python3
"""Edgewise comparison of certified OR bounds against exact OR (Section 7.1).

For every graph of the suite (or a uniform edge sample for the dense HRGs,
as in the preliminary tables) this computes:

Table 3 (non-lazy c_OR-0): mean gaps of the lower bounds JL, TLW, and
  L^square (Theorem 4.8) and of the upper bounds JL, TLW dual, and the
  envelope at zero idleness; share of edges where ours is strictly tighter /
  looser than the best prior bound; and the number of edges on which L^square
  is below JL or TLW (the "never weaker" claims).
Table 4 (lazy c_OR): lifted lower bounds, the prior upper certificate
  min(TLW dual, zero-cost bound), the envelope, and the share of edges whose
  sign the prior / matching interval certifies.
Table 5 (BF interval and sharpened interval, new): mean gaps of the BF lift
  Lambda(c_BF) and of the combined lift Lambda(max(L^square, c_BF)); share of
  edges where the combined lift beats Lambda(L^square); sign certification and
  Spearman correlation of the BF interval and of the sharpened interval.
Ranking (Table "ranking"): Spearman rho and recall of the 10% most negatively
  curved edges for BF, the JL mean (ORC-A of Tian et al.), and the midpoints
  of the prior, matching, BF, and sharpened intervals.
Extras for the pending statements of the manuscript: share of edges with
  B^BF >= 0 (Proposition 5.1(iii)); gain of Theorem 5.3 over the preliminary
  modulus (Remark 5.5); which term of Corollary 6.2 is active; violations of
  any certified bound (must be 0).

Usage
  python compare_bounds.py                    # full suite incl. real networks
  python compare_bounds.py --only "Karate;Jazz" --all-edges
  python compare_bounds.py --edgelist my.csv --name Mine
Outputs bound_comparison.csv and bound_comparison_rows.tex in --out.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from curvature_transfer import CurvatureEngine, Graph  # noqa: E402
from curvature_transfer import bounds as B  # noqa: E402
import suite  # noqa: E402

TOL = 1e-9


def topk_recall(exact, proxy, frac=0.1, seed=0):
    k = max(1, int(math.ceil(frac * len(exact))))
    rng = np.random.default_rng(seed)
    te = set(np.lexsort((rng.random(len(exact)), exact))[:k])
    tp = set(np.lexsort((rng.random(len(proxy)), proxy))[:k])
    return len(te & tp) / k


def _rho(a, b):
    return float(spearmanr(a, b).correlation) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def evaluate(name, n, edges, max_edges=None, seed=0, n_jobs=None, varpi="paper"):
    n, edges = suite.largest_component(n, edges)
    g = Graph.from_edges(edges, n)
    eng = CurvatureEngine(g, n_jobs=n_jobs, varpi=varpi)
    M = g.num_edges
    idx = np.arange(M)
    if max_edges and M > max_edges:
        idx = np.sort(np.random.default_rng(seed).choice(M, max_edges, replace=False))
    t0 = time.perf_counter()
    st = eng.stats()
    b = eng.bounds()
    t_bounds = time.perf_counter() - t0
    t0 = time.perf_counter()
    ex = eng.exact(("c_OR", "c_OR0", "c_LLY"), edges=idx)
    t_exact = time.perf_counter() - t0
    sel = {k: v[idx] for k, v in {**st, **b}.items()}
    ex0, ex1, lly = ex["c_OR0"], ex["c_OR"], ex["c_LLY"]
    di, dj = sel["deg_i"], sel["deg_j"]

    lo = {"JL": sel["jl_lower"], "TLW": sel["tlw_lower"], "ours": sel["matching_lower"]}
    up = {"JL": sel["jl_upper"], "TLW": ex["tlw_dual_nonlazy"], "ours": sel["envelope_upper_nonlazy"]}
    prior_up = np.minimum(ex["tlw_dual_lazy"], sel["zero_cost_upper"])
    Llo = {"JL": sel["jl_lower_lazy"], "ours": sel["matching_lower_lazy"],
           "BF": sel["bf_lower"], "comb": sel["sharpened_lower"]}
    Lup = {"prior": prior_up, "ours": sel["envelope_upper"], "BF": sel["bf_upper"]}
    up_match = np.minimum(Lup["ours"], Lup["prior"])
    up_sharp = np.minimum(up_match, Lup["BF"])

    out = dict(graph=name, E=M, sampled=len(idx), t_bounds=t_bounds, t_exact=t_exact)
    viol = 0
    for k, v in lo.items():
        out[f"gap_lo_{k}"] = float(np.mean(ex0 - v)); viol += int((v > ex0 + 1e-7).sum())
    for k, v in up.items():
        out[f"gap_up_{k}"] = float(np.mean(v - ex0)); viol += int((v < ex0 - 1e-7).sum())
    for k, v in Llo.items():
        out[f"gap_Llo_{k}"] = float(np.mean(ex1 - v)); viol += int((v > ex1 + 1e-7).sum())
    for k, v in Lup.items():
        out[f"gap_Lup_{k}"] = float(np.mean(v - ex1)); viol += int((v < ex1 - 1e-7).sum())
    viol += int((sel["c_BF"] > lly + 1e-7).sum())  # Lemma 3.4 (Topping et al.)
    out["violations"] = viol

    best_lo = np.maximum(lo["JL"], lo["TLW"])
    best_up = np.minimum(up["JL"], up["TLW"])
    out["lo_tighter"] = float(np.mean(lo["ours"] > best_lo + TOL))
    out["lo_looser"] = float(np.mean(lo["ours"] < best_lo - TOL))
    out["n_ours_below_JL"] = int((lo["ours"] < lo["JL"] - TOL).sum())
    out["n_ours_below_TLW"] = int((lo["ours"] < lo["TLW"] - TOL).sum())
    out["up_tighter"] = float(np.mean(up["ours"] < best_up - TOL))
    out["up_looser"] = float(np.mean(up["ours"] > best_up + TOL))
    out["lo_relgain"] = float(1 - out["gap_lo_ours"] / max(min(out["gap_lo_JL"], out["gap_lo_TLW"]), 1e-12))
    out["Llo_tighter"] = float(np.mean(Llo["ours"] > Llo["JL"] + TOL))
    out["Lup_tighter"] = float(np.mean(Lup["ours"] < Lup["prior"] - TOL))
    out["Lup_looser"] = float(np.mean(Lup["ours"] > Lup["prior"] + TOL))

    def cert(lo_, up_):
        return float(np.mean((lo_ > TOL) | (up_ < -TOL)))

    out["cert_prior"] = cert(Llo["JL"], Lup["prior"])
    out["cert_ours"] = cert(Llo["ours"], up_match)          # matching interval
    out["cert_bf"] = cert(Llo["BF"], Lup["BF"])             # BF interval
    out["cert_sharp"] = cert(Llo["comb"], up_sharp)         # sharpened interval
    out["comb_tighter"] = float(np.mean(Llo["comb"] > Llo["ours"] + TOL))
    out["comb_gain_over_bf"] = float(np.mean(Llo["comb"] > Llo["BF"] + TOL))

    # Remark 5.5: Theorem 5.3 vs the preliminary modulus
    pre = sel["bf_lower_preliminary"]
    ok = ~np.isnan(pre)
    out["thm53_strict_gain"] = float(np.mean(Llo["BF"][ok] > pre[ok] + TOL)) if ok.any() else float("nan")
    out["gap_Llo_preliminary"] = float(np.mean(ex1[ok] - pre[ok])) if ok.any() else float("nan")
    # Proposition 5.1(iii): share of edges with B^BF >= 0, and BF <= c_OR-0 there
    m2 = np.minimum(di, dj) >= 2
    Bd = sel["bf_deficit_B"][m2]
    out["frac_Bdef_nonneg"] = float(np.mean(Bd >= -TOL)) if m2.any() else float("nan")
    out["frac_BF_gt_OR0"] = float(np.mean(sel["c_BF"][m2] > ex0[m2] + TOL)) if m2.any() else float("nan")
    # Corollary 6.2: which term is active at theta = c_OR
    _, t_s0, t_match = B.psi_or_to_bf(ex1, di, dj, sel["C4"], sel["matching"], return_parts=True)
    out["cor62_s0_tighter"] = float(np.mean(t_s0 < t_match - TOL))
    out["cor62_match_tighter"] = float(np.mean(t_match < t_s0 - TOL))
    out["gap_cor62_s0"] = float(np.mean(t_s0 - sel["c_BF"]))
    out["gap_cor62_match"] = float(np.mean(t_match - sel["c_BF"]))
    out["gap_cor62_min"] = float(np.mean(np.minimum(t_s0, t_match) - sel["c_BF"]))

    proxies = {
        "BF": sel["c_BF"],
        "ORCA": 0.5 * (lo["JL"] + up["JL"]),
        "midprior": 0.5 * (Llo["JL"] + Lup["prior"]),
        "mid": 0.5 * (Llo["ours"] + up_match),
        "midbf": 0.5 * (Llo["BF"] + Lup["BF"]),
        "midsharp": 0.5 * (Llo["comb"] + up_sharp),
    }
    for k, p in proxies.items():
        out["rho_" + k] = _rho(ex1, p)
        out["rec_" + k] = topk_recall(ex1, p, 0.1, seed)
    out["rho_best_other"] = max(out["rho_BF"], out["rho_ORCA"], out["rho_midprior"])
    out["pearson_OR_BF"] = float(np.corrcoef(ex1, sel["c_BF"])[0, 1]) if np.std(sel["c_BF"]) > 0 else float("nan")
    return out


def _bold(vals, fmt, best):
    """Format values and bold every value that ties the best at display precision."""
    txt = [fmt(v) for v in vals]
    nums = [float(t) for t in txt]
    b = (min if best == "min" else max)(nums)
    return [f"\\textbf{{{t}}}" if x == b else t for t, x in zip(txt, nums)]


def latex_rows(o):
    f = lambda x: f"{x:.3f}"
    p = lambda x: f"{100 * x:.1f}"
    r2 = lambda x: f"{x:.2f}"
    lo = _bold([o["gap_lo_JL"], o["gap_lo_TLW"], o["gap_lo_ours"]], f, "min")
    up = _bold([o["gap_up_JL"], o["gap_up_TLW"], o["gap_up_ours"]], f, "min")
    a = (f"{o['graph']} & {o['E']} & {lo[0]} & {lo[1]} & {lo[2]} & {p(o['lo_tighter'])} & "
         f"{up[0]} & {up[1]} & {up[2]} & {p(o['up_tighter'])} / {p(o['up_looser'])} \\\\")
    Llo = _bold([o["gap_Llo_JL"], o["gap_Llo_ours"]], f, "min")
    Lup = _bold([o["gap_Lup_prior"], o["gap_Lup_ours"]], f, "min")
    ce = _bold([o["cert_prior"], o["cert_ours"]], p, "max")
    b = (f"{o['graph']} & {Llo[0]} & {Llo[1]} & {p(o['Llo_tighter'])} & {Lup[0]} & {Lup[1]} & "
         f"{p(o['Lup_tighter'])} & {ce[0]} & {ce[1]} \\\\")
    rho = _bold([o[k] for k in ("rho_BF", "rho_ORCA", "rho_midprior", "rho_mid")], r2, "max")
    rec = _bold([o[k] for k in ("rec_BF", "rec_ORCA", "rec_midprior", "rec_mid")], r2, "max")
    c = f"{o['graph']} & " + " & ".join(rho + rec) + " \\\\"
    gl = _bold([o["gap_Llo_BF"], o["gap_Llo_comb"]], f, "min")
    cs = _bold([o["cert_bf"], o["cert_sharp"]], p, "max")
    rs = _bold([o["rho_midbf"], o["rho_midsharp"]], r2, "max")
    d = (f"{o['graph']} & {gl[0]} & {gl[1]} & {p(o['comb_tighter'])} & {cs[0]} & {cs[1]} & "
         f"{rs[0]} & {rs[1]} \\\\")
    return a, b, c, d


def text_numbers(results):
    """Every aggregate number quoted in the abstract and Section 7, recomputed."""
    R = results
    nm = [o["graph"] for o in R]
    arr = lambda k: np.array([o[k] for o in R], float)

    def rng(k, scale=100, digits=1):
        v = arr(k) * scale
        i, j = int(np.nanargmin(v)), int(np.nanargmax(v))
        return f"{v[i]:.{digits}f} ({nm[i]}) to {v[j]:.{digits}f} ({nm[j]})"

    gain = 100 * arr("lo_relgain")
    best_up_prior = np.minimum(arr("gap_up_JL"), arr("gap_up_TLW"))
    lines = [
        f"graphs: {len(R)}",
        f"certified-bound violations (must be 0): {int(arr('violations').sum())}",
        f"edges where Thm 4.8 < JL (must be 0): {int(arr('n_ours_below_JL').sum())}",
        f"edges where Thm 4.8 < TLW: {int(arr('n_ours_below_TLW').sum())}",
        f"Thm 4.8 strictly tighter than best prior lower bound: {rng('lo_tighter')} % of edges",
        f"mean-gap reduction of Thm 4.8 vs best prior: median {np.median(gain):.1f} %, range {rng('lo_relgain')} %",
        f"graphs with reduction > 50 %: {', '.join(n for n, g in zip(nm, gain) if g > 50)}",
        f"non-lazy envelope has smaller mean gap than JL/TLW upper on {int((arr('gap_up_ours') < best_up_prior).sum())} of {len(R)} graphs",
        f"max excess of the envelope where it is looser: {np.max(np.maximum(arr('gap_up_ours') - best_up_prior, 0)):.3f}",
        f"lazy: prior upper certificate has smaller mean gap on {int((arr('gap_Lup_prior') < arr('gap_Lup_ours')).sum())} of {len(R)} graphs",
        f"lazy envelope strictly tighter than prior: at most {100 * arr('Lup_tighter').max():.1f} % of edges",
        f"matching interval certifies more signs than prior on {int((arr('cert_ours') > arr('cert_prior') + 1e-12).sum())} graphs, "
        f"as many on {int((np.abs(arr('cert_ours') - arr('cert_prior')) <= 1e-12).sum())}, "
        f"> 50 % of edges on {int((arr('cert_ours') > 0.5).sum())}",
        f"sign certified by the matching interval: {rng('cert_ours')} %",
        f"sign certified by the BF interval: {rng('cert_bf')} %",
        f"sign certified by the sharpened interval: {rng('cert_sharp')} %",
        f"matching-interval midpoint has highest or tied-highest Spearman on "
        f"{int(sum(round(o['rho_mid'], 2) >= max(round(o[k], 2) for k in ('rho_BF', 'rho_ORCA', 'rho_midprior')) for o in R))} of {len(R)} graphs",
        f"share of edges with B^BF >= 0 (Prop 5.1(iii)): {rng('frac_Bdef_nonneg')} %",
        f"share of edges with c_BF > c_OR-0: {rng('frac_BF_gt_OR0')} %",
        f"Thm 5.3 strictly beats the preliminary modulus on: {rng('thm53_strict_gain')} % of edges",
        f"combined lift strictly beats Lambda(L^square) on: {rng('comb_tighter')} % of edges",
        f"Cor 6.2: s0 term strictly tighter on {rng('cor62_s0_tighter')} %, matching term on {rng('cor62_match_tighter')} % of edges",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edgelist"); ap.add_argument("--name", default="custom")
    ap.add_argument("--only", default=None, help="semicolon-separated subset of suite names")
    ap.add_argument("--no-real", action="store_true", help="skip the real networks in experiments/data")
    ap.add_argument("--max-edges", type=int, default=None, help="subsample every graph to this many edges")
    ap.add_argument("--all-edges", action="store_true", help="ignore the HRG subsampling of the paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--varpi", choices=["paper", "legacy"], default="paper",
                    help="4-cycle normalization; 'legacy' reproduces the preliminary BF values")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "bounds"))
    args = ap.parse_args()

    if args.edgelist:
        n, e = suite.load_edgelist(args.edgelist)
        jobs = [(args.name, n, e, args.max_edges)]
    else:
        jobs = suite.full_suite(args.seed, include_real=not args.no_real,
                                only=args.only.split(";") if args.only else None)
    os.makedirs(args.out, exist_ok=True)
    results = []
    for name, n, e, me in jobs:
        me = None if args.all_edges else (args.max_edges or me)
        t0 = time.time()
        o = evaluate(name, n, e, me, args.seed, args.jobs, args.varpi)
        results.append(o)
        print(f"{name:26s} E={o['E']:6d} sampled={o['sampled']:6d} viol={o['violations']} "
              f"lo JL/TLW/ours={o['gap_lo_JL']:.3f}/{o['gap_lo_TLW']:.3f}/{o['gap_lo_ours']:.3f} "
              f"BFlift/comb={o['gap_Llo_BF']:.3f}/{o['gap_Llo_comb']:.3f} "
              f"cert prior/match/BF/sharp={100*o['cert_prior']:.1f}/{100*o['cert_ours']:.1f}/"
              f"{100*o['cert_bf']:.1f}/{100*o['cert_sharp']:.1f} ({time.time() - t0:.1f}s)", flush=True)
        with open(os.path.join(args.out, "bound_comparison.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
    with open(os.path.join(args.out, "bound_comparison_rows.tex"), "w") as fh:
        for title, k in [("Table 3 rows (non-lazy)", 0), ("Table 4 rows (lazy + sign certification)", 1),
                         ("Ranking table rows", 2), ("Table 5 rows (BF interval and sharpened interval)", 3)]:
            fh.write(f"% {title}\n" + "\n".join(latex_rows(o)[k] for o in results) + "\n")
    with open(os.path.join(args.out, "text_numbers.txt"), "w") as fh:
        fh.write(text_numbers(results))
    print(text_numbers(results))
    print(f"[compare_bounds] wrote {args.out}")


if __name__ == "__main__":
    main()
