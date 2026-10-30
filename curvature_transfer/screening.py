"""Certify-then-solve: answer threshold queries on OR curvature exactly,
solving transport problems only where the certified intervals leave the answer
undecided.

Rule. Given certified intervals lower <= c_OR <= upper, an edge is certainly in
the threshold query {e : c_OR(e) < tau} if upper < tau and certainly out if
lower >= tau; the rest are undecided.

The cascade (:func:`screen`) runs the whole procedure on a graph from scratch,
with certificates of increasing cost:

1. stage one (:func:`curvature_transfer.intervals.stage_one`): 2-hop statistics
   and one cardinality matching per edge give the BF interval and the two-hop
   interval (the sharpened interval without the TLW dual certificate);
2. on the edges that stage one leaves undecided, the TLW dual certificate
   (``_kernels.tlw_lazy_ids``: a 2-hop pass, no local distance structure), and
   the transport problem only on the edges that the certificate does not decide
   either; the edges are thus decided by exactly the sharpened interval of the
   paper.

Every decision uses the intervals widened by ``intervals.ROUNDING_TOL``.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

__all__ = ["certify_threshold", "certify_then_solve", "screen"]


def certify_then_solve(lower, upper, exact_fn: Callable[[np.ndarray], np.ndarray],
                       tau: float = 0.0) -> Dict[str, object]:
    """Exact answer to the threshold query {c_OR < tau}.

    ``exact_fn(ids)`` must return exact c_OR values for the edge ids it is
    given; it is called once, on the edges that the intervals leave undecided.
    Returns the answer mask, the certified and undecided masks, the number of
    transport problems solved, and timings.
    """
    lo = np.array(lower, float)
    up = np.array(upper, float)
    M = lo.shape[0]
    t0 = time.perf_counter()
    cin, cout, und = certify_threshold(lo, up, tau)
    t1 = time.perf_counter()
    ids = np.flatnonzero(und)
    vals = np.asarray(exact_fn(ids), float) if ids.size else np.empty(0)
    t2 = time.perf_counter()
    ans = cin.copy()
    ans[ids] = vals < tau
    return {"answer": ans, "certified_in": cin, "certified_out": cout, "undecided": und,
            "n_solved": int(ids.size), "frac_avoided": 1.0 - ids.size / max(1, M),
            "time_certify": t1 - t0, "time_solve": t2 - t1}


def screen(graph, tau: float = 0.0, *, n_jobs: Optional[int] = None, seed: int = 0) -> Dict[str, object]:
    """Certify-then-solve on ``graph`` from scratch (the cascade of the module
    docstring) for {c_OR < tau}.

    Returns the exact answer and the work done: edges decided by the BF
    interval alone (``n_bf``, reported but not used), by the two-hop interval
    (``n_two_hop``), by the TLW certificate (``n_tlw``), the number of
    transport problems solved, and wall-clock times of the two stages. Edges
    are processed in a fixed random order (``seed``) so that expensive edges
    (e.g. at hubs) are spread over the parallel chunks.
    """
    import numba

    from . import _kernels as K
    from . import intervals as I
    from .engine import _threads

    g = graph
    M = g.num_edges
    rng = np.random.default_rng(seed)
    tol = I.ROUNDING_TOL
    with _threads(n_jobs):
        nch = 8 * numba.get_num_threads()
        t0 = time.perf_counter()
        s1 = I.stage_one(g)
        lo, up = I.widen(*s1["two_hop"], tol)
        blo, bup = I.widen(*s1["bf"], tol)
        t1 = time.perf_counter()
        cin, _, und = certify_threshold(lo, up, tau)
        _, _, und_bf = certify_threshold(blo, bup, tau)
        ids = np.flatnonzero(und)
        ids = ids[rng.permutation(ids.size)]
        # an edge undecided by the two-hop interval is decided by the sharpened
        # one iff its TLW certificate (widened) is below tau
        by_tlw = K.tlw_lazy_ids(g.indptr, g.indices, g.edges, ids, nch) + tol < tau
        rest = ids[~by_tlw]
        ans = cin.copy()
        ans[ids[by_tlw]] = True
        ans[rest] = K.exact_or_ids(g.indptr, g.indices, g.edges, rest, nch) < tau
        t2 = time.perf_counter()
    n_solved = int(rest.size)
    return {"M": M, "answer": ans, "n_bf": int(M - und_bf.sum()), "n_two_hop": int(M - und.sum()),
            "n_tlw": int(by_tlw.sum()), "n_solved": n_solved, "frac_avoided": 1.0 - n_solved / max(1, M),
            "time_stage1": t1 - t0, "time_rest": t2 - t1, "time_total": t2 - t0}


def certify_threshold(lower, upper, tau: float):
    """Return (in_mask, out_mask, undecided_mask) for the query {c_OR < tau}:
    upper < tau is certainly in, lower >= tau is certainly out."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    cin = upper < tau
    cout = lower >= tau
    return cin, cout, ~(cin | cout)
