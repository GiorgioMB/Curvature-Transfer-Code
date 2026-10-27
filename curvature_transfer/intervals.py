"""The certified intervals of Section 6.1, defined in exactly one place.

Every experiment (the bound tables, certify-then-solve, the runtime figure)
uses these functions, so that "the matching interval" or "the sharpened
interval" denotes the same pair of arrays everywhere:

* ``bf``        [phi^Z(c_BF), psi(c_BF)]                     (Corollary 5.8)
* ``matching``  [Lambda^mix(L^square), min(U^prior, U^lazy)]     (lifted Theorem 4.3)
* ``sharpened`` [L^lazy, min(U^prior, U^lazy)]                    (Section 6.1: the
                 2-hop interval of Proposition 5.9 intersected with the prior
                 certificate)

U^prior = min(TLW dual bound, zero-cost bound). The TLW dual bound needs the
local distance matrix (but no transport solve); the BF interval needs only
c_BF and the two degrees.

:func:`from_arrays` assembles the intervals from per-edge arrays that were
already computed; :func:`compute` computes one interval from scratch on a
graph, doing only the work that interval needs, and is what the runtime
measurements time. The two agree exactly (tested).
:func:`stage_one` is the first stage of certify-then-solve
(:mod:`curvature_transfer.screening`): from one pass of 2-hop statistics it
returns the BF interval and the ``two_hop`` interval, i.e. the sharpened
interval without the TLW dual certificate (everything that needs no local
distance matrix):

* ``two_hop``   [L^lazy, U^lazy]  (Proposition 5.9; contained in the BF interval;
                  the upper end is capped by the zero-cost bound against rounding)

so that the sharpened interval is [two_hop lower, min(two_hop upper, TLW dual)]
(tested). The TLW dual is evaluated later, only on the edges that stage one
leaves undecided, together with the exact solve (``_kernels.screen_ids``).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from . import _kernels as K
from . import bounds as B
from .graph import Graph

__all__ = ["KINDS", "ROUNDING_TOL", "from_arrays", "compute", "stage_one", "two_hop_from_arrays", "needs_distances", "widen"]

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
    up_m = np.minimum(b["lazy_upper"], prior_upper)
    return {
        "bf": (b["bf_lower"], b["bf_upper"]),
        "matching": (b["matching_lower_lazy"], up_m),
        "sharpened": (b["lazy_lower"], up_m),
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
        if kind == "bf":
            return B.phi_int_bf_to_or(bf, di, dj), B.psi_bf_to_or(bf, di, dj)
        m = raw[:, 6].astype(float)
        Ulazy = B.lazy_upper(di, dj, tri, m, raw[:, 7].astype(float), (raw[:, 3] - raw[:, 7]).astype(float))
        tlw = K.tlw_lazy_ids(g.indptr, g.indices, g.edges, ids, 8 * numba.get_num_threads())
        up = np.minimum(Ulazy, np.minimum(tlw, B.zero_cost_upper(di, dj, tri)))
        if kind == "matching":
            return B.lift_mix(B.matching_lower(di, dj, tri, m), di, dj, tri), up
        return B.lazy_lower(di, dj, tri, m), up

def two_hop_from_arrays(b: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The ``two_hop`` interval of :func:`stage_one` from stored bound arrays
    (:meth:`CurvatureEngine.bounds`); equal to :func:`stage_one` (tested)."""
    # U^lazy <= zero-cost bound in exact arithmetic; the minimum only absorbs rounding,
    # so that the sharpened interval is the two_hop interval intersected with TLW bit for bit
    return np.asarray(b["lazy_lower"], float), np.minimum(b["lazy_upper"], b["zero_cost_upper"])


def stage_one(graph: Graph, n_jobs: Optional[int] = None,
              ids: Optional[np.ndarray] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """BF interval and ``two_hop`` interval from one pass of 2-hop statistics
    (one cardinality matching per edge; no local distance matrix, no transport
    solve). Same formulas as :func:`compute`; ``ids`` restricts to those edges."""
    from .engine import _threads

    g = graph
    ids = np.arange(g.num_edges, dtype=np.int64) if ids is None else np.ascontiguousarray(ids, dtype=np.int64)
    E = np.ascontiguousarray(g.edges[ids])
    with _threads(n_jobs):
        di = g.deg[E[:, 0]].astype(float)
        dj = g.deg[E[:, 1]].astype(float)
        raw = K.edge_stats_all(g.indptr, g.indices, E, False, True)
    tri = raw[:, 0].astype(float)
    C4 = B.c4_term(raw[:, 3].astype(float), raw[:, 4].astype(float), di, dj)
    bf = B.bf_curvature(di, dj, tri, C4)
    bf_lo, bf_up = B.phi_int_bf_to_or(bf, di, dj), B.psi_bf_to_or(bf, di, dj)
    m = raw[:, 6].astype(float)
    lo = B.lazy_lower(di, dj, tri, m)
    up = B.lazy_upper(di, dj, tri, m, raw[:, 7].astype(float), (raw[:, 3] - raw[:, 7]).astype(float))
    up = np.minimum(up, B.zero_cost_upper(di, dj, tri))  # rounding only (U^lazy <= zero-cost bound)
    return {"bf": (bf_lo, bf_up), "two_hop": (lo, up)}
