#!/usr/bin/env python3
"""Certify-then-solve experiment (Section 6.3).

Two parts (``--part``):

benchmark  every graph of the registry (``suite.py``), from the stored per-edge
           arrays (``edges.py``), no timing: the fraction of transport problems
           avoided for the query {c_OR < 0} with the BF, matching, and sharpened
           intervals (the arrays of the bound tables, so the fraction is never
           below the certified-sign fraction of Tables 3 and 5), and with the
           two-hop interval of the first stage of the cascade.
           -> <out>/screening_benchmark.csv

large      every graph of ``large_graphs.LARGE`` (10^5 to 10^6 edges), timed in
           one process with a fixed number of threads, after JIT compilation:
           exact evaluation of every edge (``timing.exact_all``, the baseline),
           and the cascade ``curvature_transfer.screen`` from scratch for
           {c_OR < 0}. Each phase is repeated until ``--min-total`` seconds or
           ``--max-reps`` runs (median reported). Every answer is checked
           against the exact curvatures (the step fails otherwise).
           -> <out>/screening_large.csv (Table 6), one JSON per graph in
           <out>/per_graph_large/ (a restarted run skips finished graphs).

Usage: python screening.py --results ../results --out ../results/tables/screening [--part all] [--jobs N]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from curvature_transfer import Graph, certify_threshold, screen  # noqa: E402
from curvature_transfer import _io  # noqa: E402
from curvature_transfer import intervals as I  # noqa: E402
import edges as E  # noqa: E402
import large_graphs as LG  # noqa: E402
import suite  # noqa: E402
import timing  # noqa: E402

RESULT_VERSION = 4  # 4: 2-hop kernel, TLW stage, and the denser large networks


# ------------------------------------------------------------------ benchmark
def benchmark_rows(specs, a):
    rows = []
    for spec in specs:
        A = E.compute(spec, a.results, a.seed, a.jobs, "paper", a.chunk)
        c = A["c_OR"]
        M = len(c)
        iv = I.from_arrays(A, A["prior_upper"])
        r = dict(graph=spec.name, tag=spec.tag, group=spec.group, E=M)
        for kind in I.KINDS:
            lo, up = I.widen(*iv[kind])
            _, _, und = certify_threshold(lo, up, 0.0)
            r[f"sign_avoided_{kind}"] = 1.0 - float(und.mean())
        lo, up = I.widen(*I.two_hop_from_arrays(A))
        _, _, und = certify_threshold(lo, up, 0.0)
        r["sign_avoided_two_hop"] = 1.0 - float(und.mean())
        rows.append(r)
        print(f"[screening/benchmark] {spec.name:26s} sign: BF {100 * r['sign_avoided_bf']:5.1f}% "
              f"sharpened {100 * r['sign_avoided_sharpened']:5.1f}%", flush=True)
    return rows


# ---------------------------------------------------------------------- large
def run_large(spec, a, threads):
    n, e, info = LG.load(spec, a.data_dir)
    g = Graph.from_edges(e, n)
    M = g.num_edges
    deg = g.deg
    rep = dict(min_total=a.min_total, max_reps=a.max_reps)
    print(f"[screening/large] {spec.name}: n={n:,} |E|={M:,} max degree={int(deg.max()):,}", flush=True)
    t_exact, c, reps_exact = timing.repeat(lambda: timing.exact_all(g, a.jobs, a.seed), **rep)
    print(f"[screening/large] {spec.name}: exact on every edge {t_exact:.2f}s ({reps_exact} runs)", flush=True)
    row = dict(graph=spec.name, tag=spec.tag, n=n, E=M, max_deg=int(deg.max()), mean_deg=float(deg.mean()),
               raw_nodes=info["raw_nodes"], raw_edges=info["raw_edges"], threads=threads,
               time_exact=t_exact, reps_exact=reps_exact, frac_negative=float((c < 0).mean()))
    t, r, reps = timing.repeat(lambda: screen(g, tau=0.0, n_jobs=a.jobs, seed=a.seed), **rep)
    ok = bool(np.array_equal(r["answer"], c < 0.0))
    row.update({f"sign_{x}": r[x] for x in ("n_bf", "n_two_hop", "n_tlw", "n_solved")})
    row.update({"sign_avoided": r["frac_avoided"], "sign_time": t, "sign_time_stage1": r["time_stage1"],
                "sign_reps": reps, "sign_speedup": t_exact / max(t, 1e-12), "sign_correct": ok})
    print(f"[screening/large] {spec.name}: sign avoided={100 * r['frac_avoided']:5.1f}% "
          f"(BF {100 * r['n_bf'] / M:.1f}%, two-hop {100 * r['n_two_hop'] / M:.1f}%, "
          f"TLW {100 * r['n_tlw'] / M:.1f}%) time={t:.2f}s (stage one {r['time_stage1']:.2f}s) "
          f"speedup={t_exact / max(t, 1e-12):.2f}x correct={ok}", flush=True)
    return row


def write_csv(rows, path):
    if not rows:
        return
    buf = io.StringIO()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    _io.atomic_text(buf.getvalue(), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", choices=["benchmark", "large", "all"], default="all")
    ap.add_argument("--only", default=None, help="benchmark part: registry names; large part: large-graph names")
    ap.add_argument("--data-dir", default=None, help="download cache of the large graphs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--min-total", type=float, default=2.0, help="repeat each timed phase until this many seconds")
    ap.add_argument("--max-reps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = [x.strip() for x in a.only.split(";")] if a.only else None
    b_only = l_only = None
    if only:  # each part runs only the names that belong to it
        b_only = [x for x in only if x in suite.BY_NAME or x in suite.BY_TAG]
        l_only = [x for x in only if x in LG.BY_NAME or x in LG.BY_TAG]
        unknown = set(only) - set(b_only) - set(l_only)
        if unknown:
            raise SystemExit(f"unknown graph(s): {sorted(unknown)}; see suite.py and large_graphs.py")
    if a.part in ("benchmark", "all") and (only is None or b_only):
        rows = benchmark_rows(suite.specs(b_only, quick=a.quick), a)
        write_csv(rows, os.path.join(a.out, "screening_benchmark.csv"))
    if a.part in ("large", "all") and (only is None or l_only):
        import numba
        timing.warmup()
        threads = numba.get_num_threads() if a.jobs is None else (numba.config.NUMBA_NUM_THREADS if a.jobs == -1 else a.jobs)
        pg_dir = os.path.join(a.out, "per_graph_large")
        os.makedirs(pg_dir, exist_ok=True)
        rows = []
        for spec in LG.specs(l_only, quick=a.quick):
            pg = os.path.join(pg_dir, f"{spec.tag}.json")
            key = {"version": RESULT_VERSION, "threads": threads, "seed": a.seed,
                   "min_total": a.min_total, "max_reps": a.max_reps}
            cached = _io.read_json(pg)
            if cached and cached.get("key") == key:
                row = cached["row"]
            else:
                t0 = time.time()
                row = run_large(spec, a, threads)
                _io.atomic_json({"key": key, "row": row}, pg)
                print(f"[screening/large] {spec.name} done ({time.time() - t0:.0f}s)", flush=True)
            rows.append(row)
            write_csv(rows, os.path.join(a.out, "screening_large.csv"))
        if not all(r["sign_correct"] for r in rows):
            raise SystemExit("certify-then-solve returned a wrong answer (large part)")


if __name__ == "__main__":
    main()
