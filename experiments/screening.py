#!/usr/bin/env python3
"""Certify-then-solve experiment (Section 7.3).

For every graph of the registry and every certified interval of Section 7.1
(BF, matching, sharpened; :mod:`curvature_transfer.intervals`, the same
arrays as in the bound tables) this reports, for the sign query (tau = 0) and
for the 10% most negatively curved edges (bottom-k):

* the fraction of transport problems avoided. Decisions use the interval
  widened by ``intervals.ROUNDING_TOL`` (as the sign columns of the tables do).
  It is computed from the stored
  per-edge intervals (``edges.py``), so it is a deterministic function of the
  same arrays that give the "sign certified" columns of Tables 3 and 5; since
  {lower > 0 or upper < 0} is contained in {lower >= 0 or upper < 0}, the
  avoided fraction for tau = 0 is never below the certified-sign fraction.
* the wall-clock time of certify-then-solve (interval computed from scratch,
  certification, exact solves on the undecided edges) and of exact evaluation
  of every edge, with the timing functions of ``timing.py`` that the runtime
  figure also uses; speedup = exact / certify-then-solve.

Checks on every run (the step fails otherwise): the interval recomputed for
timing equals the stored one, the timed run solves exactly the edges the
stored intervals leave undecided, and the returned answer equals the answer
from exact curvatures.

Each graph's rows are written to ``<out>/per_graph/<file_tag>.json`` as soon as
they are computed; a restarted run skips finished graphs.

Usage: python screening.py --results ../results --out ../results/tables/screening [--only ...] [--jobs -1]
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from curvature_transfer import Graph, certify_bottom_k, certify_then_solve, certify_threshold  # noqa: E402
from curvature_transfer import _io  # noqa: E402
from curvature_transfer import intervals as I  # noqa: E402
import edges as E  # noqa: E402
import suite  # noqa: E402
import timing  # noqa: E402

RESULT_VERSION = 2


def queries(M, frac):
    k = max(1, int(math.ceil(frac * M)))
    return [("sign", {"tau": 0.0}), (f"bottom{int(round(100 * frac))}", {"tau": None, "k": k})]


def avoided_from_intervals(lo, up, query_kw):
    """Fraction of edges decided by the intervals alone (no transport solve)."""
    if query_kw.get("k") is None:
        _, _, und = certify_threshold(lo, up, query_kw["tau"])
    else:
        _, _, und = certify_bottom_k(lo, up, query_kw["k"])
    return 1.0 - float(und.mean()), int(und.sum())


def run_graph(spec, A, n_jobs=None, frac=0.1, min_total=1.0, max_reps=5):
    g = Graph.from_edges(A["edges"], int(A["num_nodes"]))
    c = A["c_OR"]
    M = g.num_edges
    iv = I.from_arrays(A, A["prior_upper"])
    t_exact, c_check, reps_exact = timing.time_exact(g, n_jobs, min_total=min_total, max_reps=max_reps)
    if not np.array_equal(c_check, c):
        raise RuntimeError(f"{spec.name}: exact c_OR recomputed for timing differs from the stored one")
    efn = timing.exact_fn(g, n_jobs)
    rows = []
    for kind in I.KINDS:
        lo0, up0 = iv[kind]
        t_iv, (lo2, up2), reps_iv = timing.time_interval(g, kind, n_jobs, min_total=min_total, max_reps=max_reps)
        if not (np.allclose(lo2, lo0, rtol=0, atol=1e-12) and np.allclose(up2, up0, rtol=0, atol=1e-12)):
            raise RuntimeError(f"{spec.name}/{kind}: interval recomputed for timing differs from the stored one")
        lo, up = I.widen(lo0, up0)  # every decision uses the rounding-safe interval
        for qname, kw in queries(M, frac):
            avoided, n_und = avoided_from_intervals(lo, up, kw)
            r = certify_then_solve(lo, up, efn, **kw)
            if r["n_solved"] != n_und:
                raise RuntimeError(f"{spec.name}/{kind}/{qname}: solved {r['n_solved']} edges, expected {n_und}")
            if kw.get("k") is None:
                ok = bool(np.array_equal(r["answer"], c < kw["tau"]))
            else:
                k = kw["k"]
                ans = np.flatnonzero(r["answer"])
                kth = np.sort(c)[k - 1]
                ok = bool(len(ans) == k and np.all(c[ans] <= kth + 1e-12)
                          and np.all(np.isin(np.flatnonzero(c < kth - 1e-12), ans)))
            total = t_iv + r["time_certify"] + r["time_solve"]
            rows.append(dict(graph=spec.name, tag=spec.tag, group=spec.group, E=M, interval=kind, query=qname,
                             frac_avoided=avoided, n_solved=int(r["n_solved"]),
                             time_exact_all=t_exact, reps_exact=reps_exact,
                             time_interval=t_iv, reps_interval=reps_iv,
                             time_certify=r["time_certify"], time_solve=r["time_solve"],
                             time_certify_then_solve=total, speedup=t_exact / max(total, 1e-12), correct=ok))
    return rows


def write_summary(rows, out):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
    _io.atomic_text(buf.getvalue(), os.path.join(out, "screening.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--min-total", type=float, default=1.0, help="repeat each timed phase until this many seconds")
    ap.add_argument("--max-reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "per_graph"), exist_ok=True)
    timing.warmup()
    import numba
    threads = numba.get_num_threads() if a.jobs is None else (numba.config.NUMBA_NUM_THREADS if a.jobs == -1 else a.jobs)
    rows = []
    for spec in suite.specs(a.only.split(";") if a.only else None, quick=a.quick):
        A = E.compute(spec, a.results, a.seed, a.jobs, "paper", a.chunk)
        pg = os.path.join(a.out, "per_graph", f"{spec.file_tag}.json")
        key = {"fingerprint": str(A["fingerprint"]), "version": RESULT_VERSION, "frac": a.frac,
               "threads": threads, "min_total": a.min_total, "max_reps": a.max_reps}
        cached = _io.read_json(pg)
        if cached and cached.get("key") == key:
            rr = cached["rows"]
        else:
            t0 = time.time()
            rr = run_graph(spec, A, a.jobs, a.frac, a.min_total, a.max_reps)
            _io.atomic_json({"key": key, "rows": rr}, pg)
            print(f"[screening] {spec.name} done ({time.time() - t0:.1f}s)", flush=True)
        rows += rr
        for r in rr:
            print(f"{spec.name:26s} {r['interval']:9s} {r['query']:9s} avoided={100 * r['frac_avoided']:5.1f}% "
                  f"speedup={r['speedup']:6.2f}x correct={r['correct']}", flush=True)
        write_summary(rows, a.out)
    if not all(r["correct"] for r in rows):
        raise SystemExit("certify-then-solve returned a wrong answer")


if __name__ == "__main__":
    main()
