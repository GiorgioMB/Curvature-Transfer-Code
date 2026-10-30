"""Timing of exact evaluation and of the certified intervals.

Used by both ``screening.py`` (Section 6.3) and ``benchmark.py`` (Section 6.4),
so the two report the cost of the same computations. Every timed phase is a
function ``f(graph, ids, n_jobs)`` of a set of edge ids (``None`` = every
edge), listed in :data:`PHASES`:

* ``exact``                 exact lazy OR (min-cut solver);
* ``bf``                    Balanced Forman curvature;
* ``prior``                 the JL and TLW lower bounds;
* ``bf_interval``, ``matching_interval``, ``sharpened_interval``:
  :func:`curvature_transfer.intervals.compute`, the intervals of Section 6.1
  computed from scratch with only the work each needs (the matching and
  sharpened intervals include the TLW dual certificate, which the 2-hop kernel
  evaluates in the same pass as the statistics: no transport solve and no local
  distance structure).

Every phase is edgewise, so it can be timed either

* whole (:func:`repeat`): the median of repeated runs on all edges, repeated
  until ``min_total`` seconds or ``max_reps`` runs (small graphs, where a
  single run is dominated by timer noise); or
* in chunks (:func:`time_chunked`): the sum of the times of disjoint chunks of
  edges, each run once (large graphs; each chunk time can be saved as soon as
  it is measured, so a killed job loses at most one chunk).

Kernels are JIT-compiled before any timing (:func:`warmup`).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from curvature_transfer import CurvatureEngine, Graph
from curvature_transfer import _kernels as K
from curvature_transfer import bounds as B
from curvature_transfer import intervals as I
from curvature_transfer.engine import _threads

_WARM = False


def _ids(g: Graph, ids):
    return np.arange(g.num_edges, dtype=np.int64) if ids is None else np.ascontiguousarray(ids, dtype=np.int64)


def phase_exact(g: Graph, ids=None, n_jobs=None):
    import numba
    ids = _ids(g, ids)
    with _threads(n_jobs):
        return K.exact_ids(g.indptr, g.indices, g.edges, ids, True, False, False, 8 * numba.get_num_threads())[:, 0]


def _stats(g, ids, matching):
    E = np.ascontiguousarray(g.edges[_ids(g, ids)])
    di = g.deg[E[:, 0]].astype(float); dj = g.deg[E[:, 1]].astype(float)
    return di, dj, K.edge_stats_all(g.indptr, g.indices, E, False, matching)


def phase_bf(g: Graph, ids=None, n_jobs=None):
    with _threads(n_jobs):
        di, dj, raw = _stats(g, ids, False)
        return B.bf_curvature(di, dj, raw[:, 0], B.c4_term(raw[:, 3], raw[:, 4], di, dj))


def phase_prior(g: Graph, ids=None, n_jobs=None):
    with _threads(n_jobs):
        di, dj, raw = _stats(g, ids, False)
        return B.jl_lower(di, dj, raw[:, 0]), B.tlw_lower(di, dj, raw[:, 0], raw[:, 1], raw[:, 2])


def _interval(kind):
    return lambda g, ids=None, n_jobs=None: I.compute(g, kind, n_jobs, ids)


# name -> (function, uses the local distance matrix)
PHASES: Dict[str, Tuple[Callable, bool]] = {
    "bf": (phase_bf, False),
    "prior": (phase_prior, False),
    "bf_interval": (_interval("bf"), False),
    "matching_interval": (_interval("matching"), True),
    "sharpened_interval": (_interval("sharpened"), True),
    "exact": (phase_exact, True),
}


def warmup():
    """Compile every kernel that is timed, once per process."""
    global _WARM
    if _WARM:
        return
    g = Graph.from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 1), (4, 5)])
    e = CurvatureEngine(g)
    e.exact(); e.bounds(); e.prior_upper()
    for f, _ in PHASES.values():
        f(g); f(g, np.array([0, 2], dtype=np.int64))
    from curvature_transfer import screen  # the kernels of the certify-then-solve cascade
    exact_all(g)
    screen(g, tau=0.0)
    _WARM = True


def repeat(fn: Callable[[], object], min_total: float = 1.0, max_reps: int = 5) -> Tuple[float, object, int]:
    """(median seconds, last result, repetitions)."""
    ts, out = [], None
    while True:
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
        if len(ts) >= max_reps or sum(ts) >= min_total:
            break
    return float(np.median(ts)), out, len(ts)


def time_chunked(g: Graph, phase: str, chunk: int, n_jobs=None, done: Optional[Dict[int, float]] = None,
                 on_chunk: Optional[Callable[[int, float], None]] = None) -> float:
    """Sum of the times of ``phase`` on consecutive chunks of ``chunk`` edges.
    ``done`` maps chunk index -> seconds already measured (skipped);
    ``on_chunk(k, seconds)`` is called after every new chunk (to save it)."""
    f = PHASES[phase][0]
    done = dict(done or {})
    M = g.num_edges
    for k in range(-(-M // chunk)):
        if k in done:
            continue
        ids = np.arange(k * chunk, min(M, (k + 1) * chunk), dtype=np.int64)
        t = time.perf_counter()
        f(g, ids, n_jobs)
        done[k] = time.perf_counter() - t
        if on_chunk:
            on_chunk(k, done[k])
    return float(sum(done.values()))


# ------------------------------------------------- used by screening.py
def exact_fn(g: Graph, n_jobs: Optional[int] = None):
    """ids -> exact lazy c_OR on those edges (fresh, nothing cached)."""
    return lambda ids: phase_exact(g, ids, n_jobs)


def time_exact(g: Graph, n_jobs=None, **kw):
    return repeat(lambda: phase_exact(g, None, n_jobs), **kw)


def time_interval(g: Graph, kind: str, n_jobs=None, **kw):
    return repeat(lambda: I.compute(g, kind, n_jobs), **kw)

def exact_all(g: Graph, n_jobs=None, seed: int = 0) -> np.ndarray:
    """Exact lazy c_OR on every edge, the baseline of the certify-then-solve
    experiment: the min-cut solver (``K.exact_or_ids``; it also evaluates the
    TLW certificate on its local structure, a small share of a solve),
    edges in a fixed random order so that expensive edges (hubs) are spread
    over the parallel chunks, exactly as in :func:`curvature_transfer.screen`.
    Returns the values in edge order."""
    import numba
    ids = np.random.default_rng(seed).permutation(g.num_edges).astype(np.int64)
    out = np.empty(g.num_edges)
    with _threads(n_jobs):
        out[ids] = K.exact_or_ids(g.indptr, g.indices, g.edges, ids, 8 * numba.get_num_threads())
    return out
