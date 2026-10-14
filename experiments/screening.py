#!/usr/bin/env python3
"""Certify-then-solve experiment (Section 7.3).

For every graph and every certified interval (BF, matching, sharpened, as
defined in Section 7.1, so the matching and sharpened upper ends include the
TLW dual certificate, whose computation is timed) this
reports, for the sign query (tau = 0) and for the 10% most negatively curved
edges (bottom-k), the fraction of transport problems avoided, and the
wall-clock time of certify-then-solve relative to exact evaluation of every
edge. The answer is checked against the exact answer on every run.

Usage: python screening.py [--only "Karate;Jazz"] [--no-real] [--jobs -1]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from curvature_transfer import CurvatureEngine, Graph, certify_then_solve  # noqa: E402
import suite  # noqa: E402


def run_graph(name, n, edges, n_jobs=None, frac=0.1):
    n, edges = suite.largest_component(n, edges)
    g = Graph.from_edges(edges, n)
    rows = []
    # exact on every edge (reference answer and reference time)
    eng = CurvatureEngine(g, n_jobs=n_jobs)
    t = time.perf_counter(); c = eng.exact(("c_OR",))["c_OR"]; t_exact = time.perf_counter() - t
    k = max(1, int(math.ceil(frac * g.num_edges)))
    truth_bottom = set(np.argsort(c, kind="stable")[:k])
    for kind in ("bf", "matching", "sharpened"):
        e2 = CurvatureEngine(g, n_jobs=n_jobs)  # fresh engine: time the bounds too
        t = time.perf_counter()
        b = e2.bounds()
        if kind == "bf":  # Corollary 5.6: needs only BF and the degrees
            lo, up = b["bf_lower"], b["bf_upper"]
        else:             # Section 7.1: upper end intersected with the prior certificate
            up_m = np.minimum(b["envelope_upper"], e2.prior_upper())
            if kind == "matching":
                lo, up = b["matching_lower_lazy"], up_m
            else:
                lo, up = b["sharpened_lower"], np.minimum(up_m, b["bf_upper"])
        t_bounds = time.perf_counter() - t
        fn = lambda ids: e2.exact(("c_OR",), edges=ids)["c_OR"]
        for query, kw in (("sign", {"tau": 0.0}), (f"bottom{int(100 * frac)}", {"tau": None, "k": k})):
            r = certify_then_solve(lo, up, fn, **kw)
            if query == "sign":
                ok = np.array_equal(r["answer"], c < 0)
            else:
                ans = set(np.flatnonzero(r["answer"]))
                # equal up to ties at the k-th value
                kth = np.sort(c)[k - 1]
                ok = len(ans) == k and all(c[e] <= kth + 1e-12 for e in ans) and \
                    all(e in ans for e in truth_bottom if c[e] < kth - 1e-12)
            total = t_bounds + r["time_certify"] + r["time_solve"]
            rows.append(dict(graph=name, E=g.num_edges, interval=kind, query=query,
                             frac_avoided=r["frac_avoided"], n_solved=r["n_solved"],
                             time_exact_all=t_exact, time_certify_then_solve=total,
                             speedup=t_exact / max(total, 1e-12), correct=bool(ok)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "screening"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _w = CurvatureEngine.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
    _w.exact(); _w.prior_upper()  # compile every kernel once, before timing
    jobs = suite.full_suite(a.seed, include_real=not a.no_real, only=a.only.split(";") if a.only else None)
    rows = []
    for name, n, e, _ in jobs:
        rr = run_graph(name, n, e, a.jobs, a.frac)
        rows += rr
        for r in rr:
            print(f"{name:26s} {r['interval']:9s} {r['query']:9s} avoided={100 * r['frac_avoided']:5.1f}% "
                  f"speedup={r['speedup']:6.1f}x correct={r['correct']}", flush=True)
        with open(os.path.join(a.out, "screening.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if not all(r["correct"] for r in rows):
        raise SystemExit("certify-then-solve returned a wrong answer")
    # one row per graph: avoided LPs (%) for sign / bottom-k with the three intervals, and the sharpened speedup
    by = {}
    for r in rows:
        by.setdefault(r["graph"], {})[(r["interval"], r["query"].startswith("sign"))] = r
    with open(os.path.join(a.out, "screening_rows.tex"), "w") as fh:
        fh.write("% Graph & sign: BF / matching / sharpened & bottom-k: BF / matching / sharpened & speedup (sharpened, sign)\n")
        for gname, d in by.items():
            cells = [f"{100 * d[(iv, q)]['frac_avoided']:.1f}" for q in (True, False) for iv in ("bf", "matching", "sharpened")]
            fh.write(f"{gname} & " + " & ".join(cells) + f" & {d[('sharpened', True)]['speedup']:.1f} \\\\\n")


if __name__ == "__main__":
    main()
