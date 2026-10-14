"""The benchmark graphs of the paper (Tables 3-5) and the real networks."""
from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

import numpy as np

import generators as gen

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REAL = [("Arxiv", "arxiv"), ("Jazz", "jazz"), ("Karate", "karate"), ("Power Grid", "power_grid"), ("Yeast", "yeast")]  # table order


def hrg(n, R, alpha, T, seed):
    """HRG with a machine-independent tile size. The generator draws one RNG
    stream per tile, so its output depends on the tile size, which by default
    depends on the CPU count when n > 1024. For n <= 1024 (all graphs of the
    paper) this equals the default of every machine."""
    import math
    return gen.make_hyperbolic_random_graph(n, R, alpha, T, seed=seed, n_jobs=1,
                                            block_size=max(128, int(math.ceil(n / 8))))


def largest_component(n: int, edges) -> Tuple[int, np.ndarray]:
    """Largest connected component, relabeled 0..n'-1 (as in the preliminary code)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(e) == 0:
        return 0, e
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    keep = np.flatnonzero(lab == big)
    remap = -np.ones(n, np.int64)
    remap[keep] = np.arange(len(keep))
    e = remap[e]
    e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0)]
    return len(keep), e


def load_edgelist(path: str) -> Tuple[int, np.ndarray]:
    edges = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or row[0].strip().startswith("#"):
                continue
            try:
                u, v = int(row[0].strip()), int(row[1].strip())
            except ValueError:
                continue
            if u != v:
                edges.append((min(u, v), max(u, v)))
    e = np.unique(np.array(edges, dtype=np.int64).reshape(-1, 2), axis=0)
    return (int(e.max()) + 1 if len(e) else 0), e


def real_graphs(names: Optional[List[str]] = None):
    out = []
    for pretty, fname in REAL:
        if names and pretty not in names and fname not in names:
            continue
        path = os.path.join(DATA_DIR, f"{fname}.csv")
        if os.path.exists(path):
            out.append((pretty,) + load_edgelist(path))
    return out


def synthetic_suite(seed: int = 0):
    """(name, n, edges, max_exact_edges) for the 16 synthetic graphs + Karate.
    ``max_exact_edges`` mirrors the subsampling of the preliminary tables for
    the dense HRGs (None = all edges)."""
    import networkx as nx

    def G(t):
        return t[0], np.asarray(t[1], dtype=np.int64).reshape(-1, 2)

    s = []
    s.append(("BA(800, 2)",) + G(gen.barabasi_albert(800, 2, seed=seed)) + (None,))
    s.append(("BA(800, 5)",) + G(gen.barabasi_albert(800, 5, seed=seed)) + (None,))
    s.append(("BA(1600, 2)",) + G(gen.barabasi_albert(1600, 2, seed=seed)) + (None,))
    s.append(("BA(1600, 5)",) + G(gen.barabasi_albert(1600, 5, seed=seed)) + (None,))
    s.append(("G(800, 0.010013)",) + G(gen.erdos_renyi(800, 0.010013, seed=seed)) + (None,))
    s.append(("G(1600, 0.005003)",) + G(gen.erdos_renyi(1600, 0.005003, seed=seed)) + (None,))
    s.append(("RGG(800, 0.056419)",) + G(gen.random_geometric(800, 0.056419, seed=seed)) + (None,))
    s.append(("RGG(1600, 0.039894)",) + G(gen.random_geometric(1600, 0.039894, seed=seed)) + (None,))
    s.append(("HRG(800, 5.0, 1.0, 0.0)",) + G(hrg(800, 5.0, 1.0, 0.0, seed)) + (2000,))
    s.append(("HRG(800, 5.0, 1.0, 0.5)",) + G(hrg(800, 5.0, 1.0, 0.5, seed)) + (1000,))
    s.append(("WS(800,10,0.05)",) + G(gen.watts_strogatz(800, 10, 0.05, seed=seed)) + (None,))
    s.append(("WS(800,10,0.2)",) + G(gen.watts_strogatz(800, 10, 0.2, seed=seed)) + (None,))
    s.append(("WS(1600,10,0.05)",) + G(gen.watts_strogatz(1600, 10, 0.05, seed=seed)) + (None,))
    s.append(("WS(1600,10,0.2)",) + G(gen.watts_strogatz(1600, 10, 0.2, seed=seed)) + (None,))
    s.append(("SBM(2x500, 0.012, 0.004)",) + G(gen.stochastic_block_model(
        [500, 500], [[0.012018, 0.004006], [0.004006, 0.012018]], seed=seed)) + (None,))
    s.append(("Grid(40,40)",) + G(gen.grid_graph(40, 40)) + (None,))
    k = nx.karate_club_graph()
    s.append(("Karate", k.number_of_nodes(), np.array(k.edges(), dtype=np.int64), None))
    return s


def full_suite(seed: int = 0, include_real: bool = True, only=None):
    jobs = synthetic_suite(seed)
    if include_real:
        jobs += [(name, n, e, None) for name, n, e in real_graphs() if name != "Karate"]
    if only:
        keep = {x.strip() for x in only}
        jobs = [j for j in jobs if j[0] in keep]
    return jobs
