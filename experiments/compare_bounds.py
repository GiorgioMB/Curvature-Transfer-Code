#!/usr/bin/env python3
"""Edgewise comparison of certified OR bounds against exact OR (Section 6.1).

Reads the per-edge arrays of ``edges.py`` (computing and checkpointing them
if they are missing) for every graph of the registry, on every edge, and
computes:

Table 2 (non-lazy c_OR-0): mean gaps of the lower bounds JL, TLW, and
  L^square (Theorem 4.8) and of the upper bounds JL, TLW dual, and the
  envelope at zero idleness; share of edges where ours is strictly tighter /
  looser than the best prior bound; edges on which L^square is below JL/TLW.
Table 3 (lazy c_OR): lifted lower bounds, the prior upper certificate, the
  envelope, and the share of edges whose sign the prior / matching interval
  certifies.
Table 4 (ranking): Spearman rho and recall of the 10% most negatively curved
  edges for BF, the JL mean, and the midpoints of the prior and matching
  intervals.
Table 5 (BF and sharpened intervals): mean gap of the BF lift and of the
  combined lift, share of edges where the combined lift beats
  Lambda(L^square), sign certification and Spearman rho of both intervals.
Extras quoted in the text: share of edges with B^BF >= 0 (Prop 5.1(iii)),
  gain of Theorem 5.3 over the preliminary modulus, the active term of
  Corollary C.3, and violations of any certified bound (must be 0).

The intervals are those of :mod:`curvature_transfer.intervals`, the same
arrays that ``screening.py`` uses.

Each graph's result is written to ``<out>/per_graph/<file_tag>.json`` as soon
as it is computed; the summary files are rewritten after every graph.

Usage
  python compare_bounds.py --results ../results --out ../results/tables/bounds
  python compare_bounds.py --results ../results --out ... --only "Karate;Jazz"
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

from curvature_transfer import _io  # noqa: E402
from curvature_transfer import bounds as B  # noqa: E402
from curvature_transfer import intervals as I  # noqa: E402
import edges as E  # noqa: E402
import suite  # noqa: E402

TOL = I.ROUNDING_TOL  # strict comparisons between bounds ("tighter"); sign certification
#                       lower > TOL or upper < -TOL is the widened interval of intervals.widen
VIOL_TOL = 1e-7     # tolerance for the validity check of a certified bound
RESULT_VERSION = 2  # bump when evaluate() changes, to invalidate per-graph files


def topk_recall(exact, proxy, frac=0.1, seed=0):
    k = max(1, int(math.ceil(frac * len(exact))))
    rng = np.random.default_rng(seed)
    te = set(np.lexsort((rng.random(len(exact)), exact))[:k])
    tp = set(np.lexsort((rng.random(len(proxy)), proxy))[:k])
    return len(te & tp) / k


def _rho(a, b):
    return float(spearmanr(a, b).correlation) if np.std(a) > 0 and np.std(b) > 0 else float("nan")


def _mean(x):
    return float(np.mean(x)) if len(x) else float("nan")


def evaluate(spec, A, seed=0):
    """All per-graph quantities from the per-edge arrays ``A`` of ``spec``."""
    M = int(len(A["c_OR"]))
    ex0, ex1, lly = A["c_OR0"], A["c_OR"], A["c_LLY"]
    di, dj = A["deg_i"], A["deg_j"]
    iv = I.from_arrays(A, A["prior_upper"])

    lo = {"JL": A["jl_lower"], "TLW": A["tlw_lower"], "ours": A["matching_lower"]}
    up = {"JL": A["jl_upper"], "TLW": A["tlw_dual_nonlazy"], "ours": A["envelope_upper_nonlazy"]}
    Llo = {"JL": A["jl_lower_lazy"], "ours": iv["matching"][0], "BF": iv["bf"][0], "comb": iv["sharpened"][0]}
    Lup = {"prior": A["prior_upper"], "ours": A["envelope_upper"], "BF": iv["bf"][1]}
    up_match, up_sharp = iv["matching"][1], iv["sharpened"][1]

    out = dict(graph=spec.name, tag=spec.tag, group=spec.group, E=M, n=int(A["num_nodes"]))
    viol, worst = 0, 0.0

    def chk(excess):
        nonlocal viol, worst
        viol += int((excess > VIOL_TOL).sum())
        worst = max(worst, float(np.max(excess)) if len(excess) else 0.0)

    for k, v in lo.items():
        out[f"gap_lo_{k}"] = _mean(ex0 - v); chk(v - ex0)
    for k, v in up.items():
        out[f"gap_up_{k}"] = _mean(v - ex0); chk(ex0 - v)
    for k, v in Llo.items():
        out[f"gap_Llo_{k}"] = _mean(ex1 - v); chk(v - ex1)
    for k, v in Lup.items():
        out[f"gap_Lup_{k}"] = _mean(v - ex1); chk(ex1 - v)
    chk(ex1 - up_match)                                        # matching interval, upper end
    chk(ex1 - up_sharp)                                        # sharpened interval, upper end
    chk(A["c_BF"] - lly)                                       # Lemma 3.4 (Topping et al.)
    out["violations"] = viol
    out["max_violation"] = worst  # largest (bound - exact) on the wrong side; <= VIOL_TOL means rounding only

    best_lo = np.maximum(lo["JL"], lo["TLW"])
    best_up = np.minimum(up["JL"], up["TLW"])
    out["lo_tighter"] = _mean(lo["ours"] > best_lo + TOL)
    out["lo_looser"] = _mean(lo["ours"] < best_lo - TOL)
    out["n_ours_below_JL"] = int((lo["ours"] < lo["JL"] - TOL).sum())
    out["n_ours_below_TLW"] = int((lo["ours"] < lo["TLW"] - TOL).sum())
    out["up_tighter"] = _mean(up["ours"] < best_up - TOL)
    out["up_looser"] = _mean(up["ours"] > best_up + TOL)
    best_prior_gap = min(out["gap_lo_JL"], out["gap_lo_TLW"])
    # gap reduction is undefined (0/0) when the prior bounds are already exact
    out["lo_relgain"] = float(1 - out["gap_lo_ours"] / best_prior_gap) if best_prior_gap > 1e-12 else float("nan")
    out["Llo_tighter"] = _mean(Llo["ours"] > Llo["JL"] + TOL)
    out["Lup_tighter"] = _mean(Lup["ours"] < Lup["prior"] - TOL)
    out["Lup_looser"] = _mean(Lup["ours"] > Lup["prior"] + TOL)

    def cert(lo_, up_):  # strict sign certification on the widened interval: lower > 0 or upper < 0
        wl, wu = I.widen(lo_, up_)
        return _mean((wl > 0) | (wu < 0))

    out["cert_prior"] = cert(Llo["JL"], Lup["prior"])
    out["cert_ours"] = cert(*iv["matching"])
    out["cert_bf"] = cert(*iv["bf"])
    out["cert_sharp"] = cert(*iv["sharpened"])
    out["comb_tighter"] = _mean(Llo["comb"] > Llo["ours"] + TOL)
    out["comb_gain_over_bf"] = _mean(Llo["comb"] > Llo["BF"] + TOL)

    pre = A["bf_lower_preliminary"]
    ok = ~np.isnan(pre)
    out["thm53_strict_gain"] = _mean(Llo["BF"][ok] > pre[ok] + TOL) if ok.any() else float("nan")
    out["gap_Llo_preliminary"] = _mean(ex1[ok] - pre[ok]) if ok.any() else float("nan")
    m2 = np.minimum(di, dj) >= 2
    out["frac_Bdef_nonneg"] = _mean(A["bf_deficit_B"][m2] >= -TOL) if m2.any() else float("nan")
    out["frac_BF_gt_OR0"] = _mean(A["c_BF"][m2] > ex0[m2] + TOL) if m2.any() else float("nan")
    _, t_s0, t_match = B.psi_or_to_bf(ex1, di, dj, A["C4"], A["matching"], return_parts=True)
    out["cor62_s0_tighter"] = _mean(t_s0 < t_match - TOL)
    out["cor62_match_tighter"] = _mean(t_match < t_s0 - TOL)
    out["gap_cor62_s0"] = _mean(t_s0 - A["c_BF"])
    out["gap_cor62_match"] = _mean(t_match - A["c_BF"])
    out["gap_cor62_min"] = _mean(np.minimum(t_s0, t_match) - A["c_BF"])

    proxies = {
        "BF": A["c_BF"],
        "ORCA": 0.5 * (lo["JL"] + up["JL"]),
        "midprior": 0.5 * (Llo["JL"] + Lup["prior"]),
        "mid": 0.5 * (iv["matching"][0] + iv["matching"][1]),
        "midbf": 0.5 * (iv["bf"][0] + iv["bf"][1]),
        "midsharp": 0.5 * (iv["sharpened"][0] + iv["sharpened"][1]),
    }
    for k, p in proxies.items():
        out["rho_" + k] = _rho(ex1, p)
        out["rec_" + k] = topk_recall(ex1, p, 0.1, seed)
    others = [out["rho_BF"], out["rho_ORCA"], out["rho_midprior"]]
    out["rho_best_other"] = float(np.nanmax(others)) if np.isfinite(others).any() else float("nan")
    out["pearson_OR_BF"] = (float(np.corrcoef(ex1, A["c_BF"])[0, 1])
                            if np.std(A["c_BF"]) > 0 and np.std(ex1) > 0 else float("nan"))
    out["c_OR_constant"] = bool(np.std(ex1) == 0)
    return out


# ------------------------------------------------------------- formatting
def fnum(x, digits=3):
    """Fixed-point number; '--' for NaN; values that round to zero print as
    0.000 (never '-0.000')."""
    if x is None or not math.isfinite(x):
        return "--"
    s = f"{x:.{digits}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def fpct(x):
    return fnum(100 * x, 1) if x is not None and math.isfinite(x) else "--"


def _bold(txt, best):
    """Bold every entry that ties the best at display precision ('--' ignored)."""
    nums = [float(t) if t != "--" else None for t in txt]
    have = [x for x in nums if x is not None]
    if not have:
        return txt
    b = (min if best == "min" else max)(have)
    return [f"\\textbf{{{t}}}" if x is not None and x == b else t for t, x in zip(txt, nums)]


def latex_rows(o):
    f = lambda x: fnum(x, 3)
    r2 = lambda x: fnum(x, 2)
    lo = _bold([f(o["gap_lo_JL"]), f(o["gap_lo_TLW"]), f(o["gap_lo_ours"])], "min")
    up = _bold([f(o["gap_up_JL"]), f(o["gap_up_TLW"]), f(o["gap_up_ours"])], "min")
    a = (f"{o['graph']} & {o['E']} & {lo[0]} & {lo[1]} & {lo[2]} & {fpct(o['lo_tighter'])} & "
         f"{up[0]} & {up[1]} & {up[2]} & {fpct(o['up_tighter'])} / {fpct(o['up_looser'])} \\\\")
    Llo = _bold([f(o["gap_Llo_JL"]), f(o["gap_Llo_ours"])], "min")
    Lup = _bold([f(o["gap_Lup_prior"]), f(o["gap_Lup_ours"])], "min")
    ce = _bold([fpct(o["cert_prior"]), fpct(o["cert_ours"])], "max")
    b = (f"{o['graph']} & {Llo[0]} & {Llo[1]} & {fpct(o['Llo_tighter'])} & {Lup[0]} & {Lup[1]} & "
         f"{fpct(o['Lup_tighter'])} & {ce[0]} & {ce[1]} \\\\")
    rho = _bold([r2(o[k]) for k in ("rho_BF", "rho_ORCA", "rho_midprior", "rho_mid")], "max")
    rec = _bold([r2(o[k]) for k in ("rec_BF", "rec_ORCA", "rec_midprior", "rec_mid")], "max")
    c = f"{o['graph']} & " + " & ".join(rho + rec) + " \\\\"
    gl = _bold([f(o["gap_Llo_BF"]), f(o["gap_Llo_comb"])], "min")
    cs = _bold([fpct(o["cert_bf"]), fpct(o["cert_sharp"])], "max")
    rs = _bold([r2(o["rho_midbf"]), r2(o["rho_midsharp"])], "max")
    d = (f"{o['graph']} & {gl[0]} & {gl[1]} & {fpct(o['comb_tighter'])} & {cs[0]} & {cs[1]} & "
         f"{rs[0]} & {rs[1]} \\\\")
    return a, b, c, d


TABLE_NCOLS = (10, 9, 9, 8)


def grouped_rows(results, k):
    """Body rows of table ``k`` with a header row per graph group."""
    lines = []
    for grp in suite.GROUPS:
        sel = [o for o in results if o["group"] == grp]
        if not sel:
            continue
        if lines:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{TABLE_NCOLS[k]}}}{{l}}{{\\emph{{{suite.GROUP_TITLES[grp]}}}}} \\\\")
        lines += [latex_rows(o)[k] for o in sel]
    return "\n".join(lines) + "\n"


def write_summary(results, out):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)
    _io.atomic_text(buf.getvalue(), os.path.join(out, "bound_comparison.csv"))
    txt = ""
    for title, k in [("Table 2 rows (non-lazy)", 0), ("Table 3 rows (lazy + sign certification)", 1),
                     ("Ranking table rows", 2), ("Table 5 rows (BF interval and sharpened interval)", 3)]:
        txt += f"% {title}\n" + grouped_rows(results, k)
    _io.atomic_text(txt, os.path.join(out, "bound_comparison_rows.tex"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results directory (per-edge arrays live in <results>/edges)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="semicolon-separated names or tags of suite graphs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--varpi", choices=["paper", "legacy"], default="paper",
                    help="4-cycle normalization; 'legacy' reproduces the preliminary BF values")
    a = ap.parse_args()
    specs = suite.specs(a.only.split(";") if a.only else None, quick=a.quick)
    os.makedirs(os.path.join(a.out, "per_graph"), exist_ok=True)
    results = []
    for spec in specs:
        t0 = time.time()
        A = E.compute(spec, a.results, a.seed, a.jobs, a.varpi, a.chunk)
        pg = os.path.join(a.out, "per_graph", f"{spec.file_tag}.json")
        key = {"fingerprint": str(A["fingerprint"]), "version": RESULT_VERSION, "seed": a.seed, "varpi": a.varpi}
        cached = _io.read_json(pg)
        if cached and cached.get("key") == key:
            o = cached["result"]
        else:
            o = evaluate(spec, A, a.seed)
            _io.atomic_json({"key": key, "result": o}, pg)
        results.append(o)
        write_summary(results, a.out)
        print(f"{spec.name:26s} E={o['E']:6d} viol={o['violations']} (max {o['max_violation']:.1e}) "
              f"lo JL/TLW/ours={fnum(o['gap_lo_JL'])}/{fnum(o['gap_lo_TLW'])}/{fnum(o['gap_lo_ours'])} "
              f"cert prior/match/BF/sharp={fpct(o['cert_prior'])}/{fpct(o['cert_ours'])}/"
              f"{fpct(o['cert_bf'])}/{fpct(o['cert_sharp'])} ({time.time() - t0:.1f}s)", flush=True)
    bad = sum(o["violations"] for o in results)
    print(f"[compare_bounds] wrote {a.out}; certified-bound violations: {bad}")
    if bad:
        raise SystemExit("a certified bound is violated on some edge; see bound_comparison.csv (violations)")


if __name__ == "__main__":
    main()
