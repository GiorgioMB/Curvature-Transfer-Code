#!/usr/bin/env python3
"""Runtime scaling of exact OR curvature versus the certified bounds (Section 7.4).

Unlike the preliminary benchmark, which spawned a fresh process pool per phase
and therefore timed Numba JIT compilation in every worker (the flat plateau of
the old Figure 1), all kernels are compiled and cached once before timing, and
every phase runs in the same process with the same thread count. Phases:

  exact        exact lazy OR on every edge (min-cut solver)
  bf           Balanced Forman curvature (2-hop statistics without matching)
  bf_interval  BF + both transfer moduli (Corollary 5.6)
  prior        JL and TLW lower bounds (from the same statistics)
  sharpened    statistics with matching + all bounds (Proposition 5.7)
  legacy       (optional, --legacy-dir) the preliminary transportation simplex,
               timed on a uniform edge sample and extrapolated

Usage: python benchmark.py --family ER --min-n 50 --max-n 15000 --graphs 40 --jobs -1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import generators as gen  # noqa: E402
from curvature_transfer import CurvatureEngine, Graph  # noqa: E402
from curvature_transfer import _kernels as K  # noqa: E402
from curvature_transfer import bounds as B  # noqa: E402
from curvature_transfer.engine import _threads  # noqa: E402

COLS = ["family", "n", "param", "num_edges", "avg_deg", "max_deg", "threads",
        "exact_s", "bf_s", "bf_interval_s", "prior_s", "sharpened_s", "legacy_s_extrapolated"]


def make_graph(family, n, seed):
    p = float(0.2 * (50 / n) ** 0.6)  # average degree grows as O(n^0.4)
    if family == "ER":
        return p, gen.erdos_renyi(n, p, seed=seed)[1]
    if family == "BA":
        m = max(1, int(round(n * p / 2)))
        return m, gen.barabasi_albert(n, m, seed=seed)[1]
    if family == "RGG":
        r = float(np.sqrt(p / np.pi))
        return r, gen.random_geometric(n, r, seed=seed)[1]
    raise ValueError(family)


def warmup():
    g = Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 1)])
    e = CurvatureEngine(g)
    e.exact(); e.bounds()
    K.edge_stats_all(g.indptr, g.indices, g.edges, False, False)


def time_phases(g: Graph, n_jobs):
    res = {}
    with _threads(n_jobs):
        di = g.deg[g.edges[:, 0]].astype(float); dj = g.deg[g.edges[:, 1]].astype(float)
        t = time.perf_counter()
        ids = np.arange(g.num_edges, dtype=np.int64)
        import numba
        K.exact_ids(g.indptr, g.indices, g.edges, ids, True, False, False, 8 * numba.get_num_threads())
        res["exact_s"] = time.perf_counter() - t

        t = time.perf_counter()
        raw = K.edge_stats_all(g.indptr, g.indices, g.edges, False, False)
        C4 = B.c4_term(raw[:, 3], raw[:, 4], di, dj)
        bf = B.bf_curvature(di, dj, raw[:, 0], C4)
        res["bf_s"] = time.perf_counter() - t

        t = time.perf_counter()
        raw = K.edge_stats_all(g.indptr, g.indices, g.edges, False, False)
        bf = B.bf_curvature(di, dj, raw[:, 0], B.c4_term(raw[:, 3], raw[:, 4], di, dj))
        B.phi_bf_to_or(bf, di, dj); B.psi_bf_to_or(bf, di, dj)
        res["bf_interval_s"] = time.perf_counter() - t

        t = time.perf_counter()
        raw = K.edge_stats_all(g.indptr, g.indices, g.edges, False, False)
        B.jl_lower(di, dj, raw[:, 0]); B.tlw_lower(di, dj, raw[:, 0], raw[:, 1], raw[:, 2])
        res["prior_s"] = time.perf_counter() - t

        t = time.perf_counter()
        CurvatureEngine(g, n_jobs=n_jobs).bounds()
        res["sharpened_s"] = time.perf_counter() - t
    return res


def time_legacy(legacy_dir, n, edges, sample, seed):
    """Preliminary solver on a sample of edges, extrapolated (sequential)."""
    import importlib
    import types
    stubs = {}
    if "torch" not in sys.modules:  # the old module imports torch only for typing
        torch = types.ModuleType("torch"); torch.Tensor = object; torch.long = "long"
        tgd = types.ModuleType("torch_geometric.data"); tgd.Data = object
        stubs = {"torch": torch, "torch_geometric": types.ModuleType("torch_geometric"),
                 "torch_geometric.data": tgd}
        sys.modules.update(stubs)
    sys.path.insert(0, legacy_dir)
    try:
        old = importlib.import_module("pyg_curvature")
    finally:
        for k in stubs:  # do not leave fake torch behind (it confuses matplotlib)
            sys.modules.pop(k, None)
    eng = old.CurvatureEngine.__new__(old.CurvatureEngine)
    und = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
    nb = [set() for _ in range(n)]
    for a, b in und:
        nb[a].add(b); nb[b].add(a)
    eng.edges, eng.neighbors, eng._cache = und, nb, {}
    eng.deg = np.array([len(s) for s in nb])
    eng.c_OR_edge(0)  # compile
    ids = np.random.default_rng(seed).choice(len(und), min(sample, len(und)), replace=False)
    t = time.perf_counter()
    for k in ids:
        eng.c_OR_edge(int(k))
    return (time.perf_counter() - t) / len(ids) * len(und)


def plot(csv_path, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    ns = np.array([int(r["n"]) for r in rows])
    order = np.argsort(ns)
    plt.rcParams.update({"font.size": 13, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [("exact_s", "Exact OR (min-cut)", "o", "-"), ("legacy_s_extrapolated", "Exact OR (preliminary solver)", "x", ":"),
              ("sharpened_s", "Sharpened interval", "s", "--"), ("bf_interval_s", "BF interval", "D", "-."),
              ("bf_s", "Balanced Forman", "^", ":"), ("prior_s", "JL / TLW bounds", "v", "-.")]
    for key, lab, mk, ls in series:
        y = np.array([float(r[key]) if r.get(key) not in (None, "", "nan") else np.nan for r in rows])
        if np.isfinite(y).any():
            ax.plot(ns[order], y[order], marker=mk, linestyle=ls, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of nodes $n$"); ax.set_ylabel("Execution time (s)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_benchmark(family="ER", min_n=50, max_n=15000, graphs=40, seed=42, n_jobs=None,
                  legacy_dir=None, legacy_sample=200, out_dir=None, figures=True):
    out_dir = out_dir or os.path.join(HERE, "out", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"benchmark_{family.upper()}_{min_n}_{max_n}.csv")
    done = set()
    if os.path.exists(path):
        done = {int(r["n"]) for r in csv.DictReader(open(path))}
    else:
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(COLS)
    warmup()
    import numba
    threads = numba.get_num_threads() if n_jobs is None else n_jobs
    for n in sorted(set(np.geomspace(min_n, max_n, graphs, dtype=int).tolist())):
        if n in done:
            continue
        param, edges = make_graph(family, n, seed)
        g = Graph.from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n)
        if g.num_edges == 0:
            continue
        res = time_phases(g, n_jobs)
        res["legacy_s_extrapolated"] = (time_legacy(legacy_dir, n, edges, legacy_sample, seed)
                                        if legacy_dir else float("nan"))
        row = [family, n, param, g.num_edges, f"{2 * g.num_edges / n:.2f}", int(g.deg.max()), threads] + \
              [f"{res[c]:.6f}" for c in COLS[7:]]
        with open(path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)
        print(f"[benchmark] {family} n={n} M={g.num_edges} " +
              " ".join(f"{c}={res[c]:.3f}" for c in COLS[7:] if np.isfinite(res[c])), flush=True)
    if figures:
        plot(path, os.path.join(out_dir, f"runtime_scaling_loglog_{family.lower()}.pdf"))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="ER", choices=["ER", "BA", "RGG"])
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--max-n", type=int, default=15000)
    ap.add_argument("--graphs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--legacy-dir", default=None, help="path to the preliminary repository (for the old solver)")
    ap.add_argument("--legacy-sample", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run_benchmark(a.family, a.min_n, a.max_n, a.graphs, a.seed, a.jobs, a.legacy_dir, a.legacy_sample, a.out)


if __name__ == "__main__":
    main()
