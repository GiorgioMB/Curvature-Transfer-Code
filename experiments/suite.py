"""The graphs of the paper, defined once.

Every step that evaluates graphs (the per-edge tables, Tables 2-5, the
screening table, Figures 2-4, the numbers quoted in the text) draws its graphs
from :data:`REGISTRY` through :func:`load`, so the same name always denotes
the same edge set. In particular:

* every graph is restricted to its largest connected component (the paper
  assumes connected graphs), in one place, :func:`load`;
* there is no edge subsampling anywhere: every statistic uses every edge;
* the per-edge file tag (``Spec.tag``) is what the figure file names use, so
  a figure and a table row with the same graph are computed from the same
  per-edge arrays.

Groups:

* ``random``    random graph models (degree heterogeneity, clustering,
                geometry, community structure);
* ``canonical`` deterministic families with known curvature (clean extremes);
* ``real``      the real networks in ``experiments/data``.

Aggregate statements in the text (ranges, medians, counts "on k of the N
graphs") are computed over :data:`AGGREGATE_GROUPS`; the canonical graphs are
reported in the tables but are excluded from those aggregates, because on
them several bounds are exact by construction (Theorem 4.8 on trees, JL on
trees, BF on K_n), which makes ratios such as the gap reduction 0/0.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

import generators as gen

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GROUPS = ("random", "canonical", "real")
AGGREGATE_GROUPS = ("random", "real")
GROUP_TITLES = {"random": "Random graph models", "canonical": "Canonical graphs", "real": "Real networks"}


@dataclass(frozen=True)
class Spec:
    name: str              # name in the tables
    tag: str               # per-edge file tag (figure file names use tag with '.' -> '_')
    group: str             # random | canonical | real
    make: Callable[[int], Tuple[int, object]] = field(compare=False, repr=False)  # seed -> (n, edges)
    prose: str = ""        # name used in running text (LaTeX); defaults to ``name``

    @property
    def file_tag(self) -> str:
        return self.tag.replace(".", "_")

    @property
    def display(self) -> str:
        return self.prose or self.name


def hrg(n, R, alpha, T, seed):
    """HRG with a machine-independent tile size (the generator draws one RNG
    stream per tile, so its output depends on the tile size)."""
    return gen.make_hyperbolic_random_graph(n, R, alpha, T, seed=seed, n_jobs=1,
                                            block_size=max(128, int(math.ceil(n / 8))))


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


def _real(fname):
    return lambda seed: load_edgelist(os.path.join(DATA_DIR, f"{fname}.csv"))


def _R(name, tag, make, prose=""):
    return Spec(name, tag, "random", make, prose)


def _C(name, tag, make, prose=""):
    return Spec(name, tag, "canonical", make, prose)


REGISTRY: List[Spec] = [
    # --- random graph models -------------------------------------------------
    _R("BA(800, 2)", "ba_n800_m2", lambda s: gen.barabasi_albert(800, 2, seed=s), "BA$(800,2)$"),
    _R("BA(800, 5)", "ba_n800_m5", lambda s: gen.barabasi_albert(800, 5, seed=s), "BA$(800,5)$"),
    _R("BA(1600, 2)", "ba_n1600_m2", lambda s: gen.barabasi_albert(1600, 2, seed=s), "BA$(1600,2)$"),
    _R("BA(1600, 5)", "ba_n1600_m5", lambda s: gen.barabasi_albert(1600, 5, seed=s), "BA$(1600,5)$"),
    _R("ER(800, 0.0100125)", "er_n800_p0.0100125", lambda s: gen.erdos_renyi(800, 0.0100125, seed=s), "ER$(800)$"),
    _R("ER(1600, 0.0050031)", "er_n1600_p0.0050031", lambda s: gen.erdos_renyi(1600, 0.0050031, seed=s), "ER$(1600)$"),
    _R("RGG(800, 0.056419)", "rg_n800_r0.056419", lambda s: gen.random_geometric(800, 0.056419, seed=s), "RGG$(800)$"),
    _R("RGG(1600, 0.039894)", "rg_n1600_r0.039894", lambda s: gen.random_geometric(1600, 0.039894, seed=s), "RGG$(1600)$"),
    _R("HRG(800, 5.0, 1.0, 0.0)", "hrg_n800_R5.0_a1.0_T0.0", lambda s: hrg(800, 5.0, 1.0, 0.0, s), "HRG ($T=0$)"),
    _R("HRG(800, 5.0, 1.0, 0.5)", "hrg_n800_R5.0_a1.0_T0.5", lambda s: hrg(800, 5.0, 1.0, 0.5, s), "HRG ($T=0.5$)"),
    _R("WS(800, 10, 0.05)", "ws_n800_k10_b0.05", lambda s: gen.watts_strogatz(800, 10, 0.05, seed=s), "WS$(800,\\beta=0.05)$"),
    _R("WS(800, 10, 0.2)", "ws_n800_k10_b0.2", lambda s: gen.watts_strogatz(800, 10, 0.2, seed=s), "WS$(800,\\beta=0.2)$"),
    _R("WS(1600, 10, 0.05)", "ws_n1600_k10_b0.05", lambda s: gen.watts_strogatz(1600, 10, 0.05, seed=s), "WS$(1600,\\beta=0.05)$"),
    _R("WS(1600, 10, 0.2)", "ws_n1600_k10_b0.2", lambda s: gen.watts_strogatz(1600, 10, 0.2, seed=s), "WS$(1600,\\beta=0.2)$"),
    _R("SBM(2x500, 0.012, 0.004)", "sbm2_n1000_pin0.012018_pout0.004006",
       lambda s: gen.make_sbm_graph([500, 500], 0.012018, 0.004006, seed=s), "SBM (assortative)"),
    _R("SBM(2x500, 0.004, 0.012)", "sbm2_n1000_pin0.004002_pout0.012006",
       lambda s: gen.make_sbm_graph([500, 500], 0.004002, 0.012006, seed=s), "SBM (disassortative)"),
    _R("RReg(1000, 8)", "rreg_n1000_d8", lambda s: gen.d_regular_graph(1000, 8, seed=s), "RReg$(1000,8)$"),
    _R("RReg(2000, 8)", "rreg_n2000_d8", lambda s: gen.d_regular_graph(2000, 8, seed=s), "RReg$(2000,8)$"),
    # --- canonical families -------------------------------------------------
    _C("Cycle(600)", "cycle_n600", lambda s: gen.cycle_graph(600), "Cycle"),
    _C("Grid(40,40)", "grid_40x40", lambda s: gen.grid_graph(40, 40), "Grid"),
    _C("Torus(32,32)", "torus_32x32", lambda s: gen.torus_graph(32, 32), "Torus$(32,32)$"),
    _C("Torus(40,40)", "torus_40x40", lambda s: gen.torus_graph(40, 40), "Torus$(40,40)$"),
    _C("Tree(4,6)", "tree_d4_h6", lambda s: gen.dary_tree(4, 6), "Tree"),
    _C("Complete(120)", "complete_n120", lambda s: gen.complete_graph(120), "$K_{120}$"),
    # --- real networks (table order) -----------------------------------------
    Spec("Arxiv", "real_arxiv", "real", _real("arxiv")),
    Spec("Jazz", "real_jazz", "real", _real("jazz")),
    Spec("Karate", "real_karate", "real", _real("karate")),
    Spec("Power Grid", "real_power_grid", "real", _real("power_grid")),
    Spec("Yeast", "real_yeast", "real", _real("yeast")),
]

BY_NAME = {s.name: s for s in REGISTRY}
BY_TAG = {s.tag: s for s in REGISTRY}
BY_FILE_TAG = {s.file_tag: s for s in REGISTRY}
assert len(BY_NAME) == len(REGISTRY) and len(BY_TAG) == len(REGISTRY), "duplicate names or tags in the registry"

# graphs used by --quick (a smoke test of the pipeline, not the paper's numbers)
QUICK = ["BA(800, 2)", "ER(800, 0.0100125)", "HRG(800, 5.0, 1.0, 0.0)", "Grid(40,40)", "Tree(4,6)", "Karate", "Yeast"]


def largest_component(n: int, edges) -> Tuple[int, np.ndarray]:
    """Largest connected component, relabeled 0..n'-1, edges (u < v) unique."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    if len(e) == 0:
        return 0, e
    e = np.unique(np.sort(e, axis=1), axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    big = np.bincount(lab).argmax()
    keep = np.flatnonzero(lab == big)
    remap = -np.ones(n, np.int64)
    remap[keep] = np.arange(len(keep))
    e = remap[e]
    e = e[(e[:, 0] >= 0) & (e[:, 1] >= 0)]
    return len(keep), e


def load(spec: Spec, seed: int = 0) -> Tuple[int, np.ndarray]:
    """(n, edges) of the largest connected component of the graph ``spec``."""
    n, e = spec.make(seed)
    return largest_component(n, np.asarray(e, dtype=np.int64).reshape(-1, 2))


def specs(only: Optional[Sequence[str]] = None, groups: Optional[Sequence[str]] = None,
          quick: bool = False) -> List[Spec]:
    """Registry entries, in table order. ``only``: names or tags."""
    out = list(REGISTRY)
    if quick:
        out = [s for s in out if s.name in QUICK]
    if groups:
        out = [s for s in out if s.group in groups]
    if only:
        keep = {x.strip() for x in only}
        out = [s for s in out if s.name in keep or s.tag in keep or s.file_tag in keep]
        missing = keep - {s.name for s in out} - {s.tag for s in out} - {s.file_tag for s in out}
        if missing:
            raise SystemExit(f"unknown graph(s): {sorted(missing)}; see experiments/suite.py")
    return out
