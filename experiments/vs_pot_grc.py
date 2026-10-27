"""Per-edge timing of the exact solver against POT and GraphRicciCurvature.

For each graph (or a random sample of its edges) this times, single-threaded
and with warm JIT, the lazy OR curvature computed by

* this package (``_kernels.exact_or_ids``): everything included (support,
  adjacency and coverage, flow);
* the reference solver (``_reference``, the previous implementation; --old);
* POT on identical problems (fastest of three passes, like the timings above): ``ot.emd2`` (the public API, which
  GraphRicciCurvature calls), and POT's C++ network simplex called directly
  (``emd_c``) on the full problem (mu on N[i], nu on N[j]) and on the
  mass-cancelled one ((mu - nu)^+ to (mu - nu)^-). POT's inputs (marginals and
  local distance matrix, built with the reference kernels) are given for free:
  only the solve is timed;
* with --grc, GraphRicciCurvature end to end (alpha = 0, i.e. the non-lazy
  curvature, which both packages compute; method "OTD", one process).

Every value is checked against POT (and GraphRicciCurvature). Run with the
``compare`` extra. GraphRicciCurvature 0.6.1 caps scipy<=1.13.1, so --grc needs
its own environment, for example::

    PYTHONPATH=. uv run --no-project --python 3.12 --with numba==0.62.1 --with numpy==2.2.0 \\
        --with scipy==1.13.1 --with POT==0.9.7.post1 --with GraphRicciCurvature==0.6.1 \\
        --with networkx==3.3 --with pandas python experiments/vs_pot_grc.py --grc

Graph names: karate, yeast, jazz, power_grid, arxiv (experiments/data), er_d10,
ba_m5, rgg, er_d90 (seeded synthetic graphs).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numba  # noqa: E402
import networkx as nx  # noqa: E402
import ot  # noqa: E402
from ot.lp.emd_wrap import emd_c  # noqa: E402

from curvature_transfer import Graph  # noqa: E402
from curvature_transfer import _kernels as K  # noqa: E402
from curvature_transfer import _reference as RF  # noqa: E402

DATA = os.path.join(ROOT, "experiments", "data")


def _lcc(G):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def load(name):
    if name in ("karate", "yeast", "jazz", "power_grid", "arxiv"):
        e = np.loadtxt(os.path.join(DATA, f"{name}.csv"), delimiter=",", dtype=np.int64)
        G = nx.Graph()
        G.add_edges_from(map(tuple, e))
        return _lcc(G)
    if name == "er_d10":
        return _lcc(nx.fast_gnp_random_graph(2000, 10 / 1999, seed=0))
    if name == "ba_m5":
        return _lcc(nx.barabasi_albert_graph(5000, 5, seed=0))
    if name == "rgg":
        return _lcc(nx.random_geometric_graph(3000, 0.05, seed=0))
    if name == "er_d90":
        return _lcc(nx.fast_gnp_random_graph(600, 90 / 599, seed=0))
    raise SystemExit(f"unknown graph {name!r}")


def pot_problem(g, e, scratch, epoch, net):
    """(a, b, M) for the lazy measures of edge e, from the reference kernels."""
    i, j = int(g.edges[e, 0]), int(g.edges[e, 1])
    S = RF.local_support(g.indptr, g.indices, i, j)
    D = RF.local_distances(g.indptr, g.indices, S, scratch, epoch).astype(np.float64)
    Ni = np.isin(S, g.indices[g.indptr[i]:g.indptr[i + 1]]) | (S == i)
    Nj = np.isin(S, g.indices[g.indptr[j]:g.indptr[j + 1]]) | (S == j)
    mu = Ni / Ni.sum()
    nu = Nj / Nj.sum()
    if net:
        w = mu - nu
        P, N = w > 1e-15, w < -1e-15
        return w[P].copy(), -w[N].copy(), np.ascontiguousarray(D[np.ix_(P, N)])
    return mu[Ni].copy(), nu[Nj].copy(), np.ascontiguousarray(D[np.ix_(Ni, Nj)])


def time_ours(g, ids, reps=3):
    ts, v = [], None
    for _ in range(reps):
        t = time.perf_counter()
        v = K.exact_or_ids(g.indptr, g.indices, g.edges, ids, 1)
        ts.append(time.perf_counter() - t)
    return min(ts) / len(ids), v


def time_old(g, ids, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        RF.exact_ids_reference(g.indptr, g.indices, g.edges, ids, True, False, False, 1)
        ts.append(time.perf_counter() - t)
    return min(ts) / len(ids)


def time_pot(g, ids, reps=3):
    """Like :func:`time_ours`, the fastest of ``reps`` passes over the edges."""
    best = None
    for _ in range(reps):
        r = _pot_pass(g, ids)
        best = r if best is None else (min(best[0], r[0]), min(best[1], r[1]), min(best[2], r[2]), r[3])
    return best


def _pot_pass(g, ids):
    scratch = RF.new_scratch(g.num_nodes)
    t2 = tf = tn = 0.0
    vals = np.empty((len(ids), 3))
    for k, e in enumerate(ids):
        a, b, M = pot_problem(g, int(e), scratch, 2 * k, net=False)
        b = b * (a.sum() / b.sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = time.perf_counter()
            W = ot.emd2(a, b, M, numItermax=10**7)
            t2 += time.perf_counter() - t
        t = time.perf_counter()
        r = emd_c(a, b, M, 10**7, 1)
        tf += time.perf_counter() - t
        vals[k, 0], vals[k, 1] = 1 - W, 1 - r[1]
        a, b, M = pot_problem(g, int(e), scratch, 2 * k + 1, net=True)
        if a.size == 0:  # mu == nu (emd_c does not accept empty input)
            vals[k, 2] = 1.0
            continue
        b = b * (a.sum() / b.sum())
        t = time.perf_counter()
        r = emd_c(a, b, M, 10**7, 1)
        tn += time.perf_counter() - t
        vals[k, 2] = 1 - r[1]
    n = len(ids)
    return t2 / n, tf / n, tn / n, vals


def time_grc(G, g, ids, sample):
    """Whole graph (as a user runs it, so the all-pairs distance matrix is
    amortized over every edge) when it has at most 40k edges; otherwise a
    random sample with on-demand shortest paths (all_pairs needs O(n^2) memory)."""
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    full = g.num_edges <= 40_000
    if not full:
        ids = ids[np.random.default_rng(1).permutation(len(ids))[:sample]]
    edge_list = None if full else [tuple(map(int, g.edges[e])) for e in ids]
    sp = "all_pairs" if full and g.num_nodes <= 6000 else "pairwise"
    t = time.perf_counter()
    orc = OllivierRicci(G, alpha=0.0, method="OTD", proc=1, shortest_path=sp, verbose="ERROR")
    res = orc.compute_ricci_curvature_edges(edge_list)
    dt = time.perf_counter() - t
    v = np.array([res[(u, w)] if (u, w) in res else res[(w, u)] for u, w in map(tuple, g.edges[ids].tolist())])
    return dt / (g.num_edges if full else len(ids)), v, sp, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graphs", nargs="+", default=["yeast", "power_grid", "er_d10", "ba_m5", "rgg", "jazz", "er_d90", "arxiv"])
    ap.add_argument("--sample", type=int, default=3000, help="edges per graph (random sample; 0 = all)")
    ap.add_argument("--old", action="store_true", help="also time the reference (previous) solver")
    ap.add_argument("--grc", action="store_true", help="also time GraphRicciCurvature (needs scipy<=1.13.1)")
    ap.add_argument("--grc-sample", type=int, default=300, help="GRC edges on graphs above 40k edges")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()
    numba.set_num_threads(1)
    warm = Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])
    K.exact_or_ids(warm.indptr, warm.indices, warm.edges, np.arange(warm.num_edges), 1)
    K.exact_ids(warm.indptr, warm.indices, warm.edges, np.arange(warm.num_edges), True, True, True, 1)
    RF.exact_ids_reference(warm.indptr, warm.indices, warm.edges, np.arange(warm.num_edges), True, False, False, 1)
    rows = []
    hdr = f"{'graph':11s} {'edges':>7s} {'ours':>9s} " + (f"{'old':>9s} " if args.old else "") + \
          f"{'POT core':>9s} {'ot.emd2':>9s} " + (f"{'GRC':>9s} " if args.grc else "") + \
          f"{'core/ours':>9s}  (microseconds per edge)"
    print(hdr, flush=True)
    for name in args.graphs:
        G = load(name)
        g = Graph.from_edges(np.array(G.edges()), G.number_of_nodes())
        m = g.num_edges
        if args.sample and args.sample < m:
            ids = np.sort(np.random.default_rng(0).choice(m, args.sample, replace=False)).astype(np.int64)
        else:
            ids = np.arange(m, dtype=np.int64)
        r_ours, v_ours = time_ours(g, ids)
        r_old = time_old(g, ids) if args.old else float("nan")
        r_e2, r_full, r_net, vp = time_pot(g, ids)
        dmax = float(np.max(np.abs(vp - v_ours[:, None])))
        row = dict(graph=name, n=g.num_nodes, m=m, timed=len(ids), ours_us=1e6 * r_ours, old_us=1e6 * r_old,
                   emd2_us=1e6 * r_e2, core_full_us=1e6 * r_full, core_net_us=1e6 * r_net,
                   max_abs_diff_vs_pot=dmax)
        if args.grc:
            r_grc, v_grc, sp, gids = time_grc(G, g, ids, args.grc_sample)
            v0 = K.exact_ids(g.indptr, g.indices, g.edges, gids, False, True, False, 1)[:, 1]
            row.update(grc_us=1e6 * r_grc, grc_shortest_path=sp, max_abs_diff_vs_grc=float(np.max(np.abs(v_grc - v0))))
        core = min(row["core_full_us"], row["core_net_us"])
        print(f"{name:11s} {len(ids):7d} {row['ours_us']:9.1f} " + (f"{row['old_us']:9.1f} " if args.old else "") +
              f"{core:9.1f} {row['emd2_us']:9.1f} " + (f"{row['grc_us']:9.1f} " if args.grc else "") +
              f"{core / row['ours_us']:9.2f}  max|diff| {dmax:.0e}" +
              (f" / GRC {row['max_abs_diff_vs_grc']:.0e}" if args.grc else ""), flush=True)
        rows.append(row)
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
