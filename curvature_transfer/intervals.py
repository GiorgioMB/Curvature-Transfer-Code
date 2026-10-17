"""The certified intervals of Section 7.1, defined in exactly one place.

Every experiment (the bound tables, certify-then-solve, the runtime figure)
uses these functions, so that "the matching interval" or "the sharpened
interval" denotes the same pair of arrays everywhere:

* ``bf``        [phi(c_BF), psi(c_BF)]                         (Corollary 5.5)
* ``matching``  [Lambda(L^square), min(U^prior, U^env)]         (Section 7.1)
* ``sharpened`` [Lambda(max(L^square, c_BF)), min(U^prior, U^env, psi(c_BF))]
                                                                (Section 7.1, i.e.
                 Proposition 5.6 intersected with the prior certificate)

U^prior = min(TLW dual bound, zero-cost bound). The TLW dual bound needs the
local distance matrix (but no transport solve); the BF interval needs only
c_BF and the two degrees.

:func:`from_arrays` assembles the intervals from per-edge arrays that were
already computed; :func:`compute` computes one interval from scratch on a
graph, doing only the work that interval needs, and is what the runtime
measurements time. The two agree exactly (tested).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import _kernels as K
from . import bounds as B
from .graph import Graph

__all__ = ["KINDS", "ROUNDING_TOL", "from_arrays", "compute", "needs_distances", "widen"]

KINDS = ("bf", "matching", "sharpened")

# The bounds are evaluated in floating point while the exact curvatures are
# rational numbers computed in integer arithmetic. Every decision taken from an
# interval (certified sign, certify-then-solve) uses the interval widened by
# this tolerance, which keeps it valid under rounding. Without it, an edge whose
# lower and upper bound coincide in exact arithmetic (e.g. every edge of K_n)
# can get lower > upper by one ulp and be "certified" wrongly.
ROUNDING_TOL = 1e-9


def widen(lo: np.ndarray, up: np.ndarray, tol: float = ROUNDING_TOL) -> Tuple[np.ndarray, np.ndarray]:
    """The interval [lo - tol, up + tol] used for every certified decision."""
    return np.asarray(lo, float) - tol, np.asarray(up, float) + tol


def needs_distances(kind: str) -> bool:
    return kind in ("matching", "sharpened")


def from_arrays(b: Dict[str, np.ndarray], prior_upper: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Intervals from the bound arrays of :meth:`CurvatureEngine.bounds` and
    the prior upper certificate min(TLW dual, zero-cost bound)."""
    up_m = np.minimum(b["envelope_upper"], prior_upper)
    return {
        "bf": (b["bf_lower"], b["bf_upper"]),
        "matching": (b["matching_lower_lazy"], up_m),
        "sharpened": (b["sharpened_lower"], np.minimum(up_m, b["bf_upper"])),
    }


def compute(graph: Graph, kind: str, n_jobs: Optional[int] = None,
            ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one certified interval from scratch (nothing cached).

    This is the code path that the screening and runtime experiments time.
    ``ids`` restricts the computation to those edge ids (every quantity is
    edgewise, so computing the intervals chunk by chunk and concatenating gives
    exactly the intervals of the whole graph; tested).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    import numba

    from .engine import _threads

    g = graph
    ids = np.arange(g.num_edges, dtype=np.int64) if ids is None else np.ascontiguousarray(ids, dtype=np.int64)
    E = np.ascontiguousarray(g.edges[ids])
    with _threads(n_jobs):
        di = g.deg[E[:, 0]].astype(float)
        dj = g.deg[E[:, 1]].astype(float)
        raw = K.edge_stats_all(g.indptr, g.indices, E, False, kind != "bf")
        tri = raw[:, 0].astype(float)
        C4 = B.c4_term(raw[:, 3].astype(float), raw[:, 4].astype(float), di, dj)
        bf = B.bf_curvature(di, dj, tri, C4)
        bf_up = B.psi_bf_to_or(bf, di, dj)
        if kind == "bf":
            return B.phi_bf_to_or(bf, di, dj), bf_up
        Lsq = B.matching_lower(di, dj, tri, raw[:, 6].astype(float))
        env = B.envelope_upper(di, dj, tri, raw[:, 1].astype(float), raw[:, 2].astype(float), raw[:, 3].astype(float))
        tlw = K.tlw_lazy_ids(g.indptr, g.indices, g.edges, ids, 8 * numba.get_num_threads())
        up = np.minimum(env, np.minimum(tlw, B.zero_cost_upper(di, dj, tri)))
        if kind == "matching":
            return B.lift(Lsq, di, dj), up
        return B.sharpened_lower(Lsq, bf, di, dj), np.minimum(up, bf_up)
