"""Certify-then-solve: answer threshold and bottom-k queries on OR curvature
exactly, solving transport problems only where the certified intervals leave
the answer undecided.

Rules. Given certified intervals lower <= c_OR <= upper:

* threshold query {e : c_OR(e) < tau}: an edge with upper < tau is certainly
  in, an edge with lower >= tau is certainly out, the rest are undecided;
* bottom-k query: sort the intervals by `cmp(e, f) = (upper_e < lower_f)`. Edge
  e is certainly in if at most k - 1 other edges f have lower_f <= upper_e
  (then at most k - 1 edges can have c_OR <= c_OR(e)), and certainly out if at
  least k other edges have upper_f < lower_e (then k edges are strictly
  smaller). Both rules are valid under any tie-breaking, and both are monotone:
  shrinking any interval never undoes a decision.

Schedules for bottom-k (:func:`certify_then_solve`):

* ``"static"``: solve every edge that the initial intervals leave undecided;
* ``"adaptive"`` (default): solve the undecided edges in batches of increasing
  size, in increasing order of the midpoint of their interval, and re-apply the
  rules after every batch. An exact value is an interval of zero width (widened
  by the rounding tolerance), so every batch can only decide more edges; the
  loop stops when every undecided edge has been solved. The answer is exact
  for every schedule; only the number of transport problems solved differs.

The cascade (:func:`screen`) runs the whole procedure on a graph from scratch,
with certificates of increasing cost:

1. stage one (:func:`curvature_transfer.intervals.stage_one`): 2-hop statistics
   and one cardinality matching per edge give the BF interval and the two-hop
   interval (the sharpened interval without the TLW dual certificate);
2. threshold queries: on the edges that stage one leaves undecided, one kernel
   builds the local distance matrix, evaluates the TLW dual certificate on it,
   and solves the transport problem only if the certificate does not decide
   the edge (``_kernels.screen_ids``); the edges are thus decided by exactly the
   sharpened interval of the paper, and no distance matrix is built twice;
   bottom-k queries: adaptive exact solves on the two-hop intervals.

Every decision uses the intervals widened by ``intervals.ROUNDING_TOL``.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Optional

import numpy as np

__all__ = ["certify_threshold", "certify_bottom_k", "certify_then_solve", "screen", "SCHEDULES"]

SCHEDULES = ("static", "adaptive")


def _first_batch(M: int) -> int:
    return max(1024, int(math.ceil(0.005 * M)))


def certify_then_solve(lower, upper, exact_fn: Callable[[np.ndarray], np.ndarray],
                       tau: Optional[float] = 0.0, k: Optional[int] = None,
                       schedule: str = "adaptive", batch: Optional[int] = None, growth: float = 2.0,
                       tol: float = 0.0) -> Dict[str, object]:
    """Exact answer to a threshold query (``tau``) or a bottom-k query (``k``).

    ``exact_fn(ids)`` must return exact c_OR values for the edge ids it is
    given. ``schedule`` applies to bottom-k queries (threshold queries are
    decided edge by edge, so every schedule solves the same edges). ``tol`` is
    the width given to an exact value when it replaces an interval (use the
    rounding tolerance with which ``lower``/``upper`` were widened). Returns the
    answer mask, the number of transport problems solved, the number of
    rounds, and timings.
    """
    if (tau is None) == (k is None):
        raise ValueError("give exactly one of tau or k")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    lo = np.array(lower, float)
    up = np.array(upper, float)
    M = lo.shape[0]
    t0 = time.perf_counter()
    if k is None:
        cin, cout, und = certify_threshold(lo, up, tau)
        t1 = time.perf_counter()
        ids = np.flatnonzero(und)
        vals = np.asarray(exact_fn(ids), float) if ids.size else np.empty(0)
        t2 = time.perf_counter()
        ans = cin.copy()
        ans[ids] = vals < tau
        return {"answer": ans, "certified_in": cin, "certified_out": cout, "undecided": und,
                "n_solved": int(ids.size), "frac_avoided": 1.0 - ids.size / max(1, M), "rounds": int(ids.size > 0),
                "time_certify": t1 - t0, "time_solve": t2 - t1}
    cin, cout, und = certify_bottom_k(lo, up, k)
    t_cert = time.perf_counter() - t0
    t_solve = 0.0
    vals = np.full(M, np.nan)
    solved = np.zeros(M, bool)
    und0 = und.copy()
    rounds = 0
    b = batch or _first_batch(M)
    while True:
        cand = np.flatnonzero(und & ~solved)
        if cand.size == 0:
            break
        if schedule == "static":
            take = cand
        else:
            mid = 0.5 * (lo[cand] + up[cand])
            take = cand[np.argsort(mid, kind="stable")[:b]]
            b = int(math.ceil(b * growth))
        t = time.perf_counter()
        v = np.asarray(exact_fn(take), float)
        t_solve += time.perf_counter() - t
        vals[take] = v
        solved[take] = True
        rounds += 1
        t = time.perf_counter()
        # an exact value is an interval of width 2 tol inside the old one
        lo[take] = np.maximum(lo[take], v - tol)
        up[take] = np.minimum(up[take], v + tol)
        cin, cout, und = certify_bottom_k(lo, up, k)
        t_cert += time.perf_counter() - t
    ans = cin.copy()
    need = k - int(cin.sum())
    if need > 0:
        pool = np.flatnonzero(und)  # every undecided edge has been solved
        ans[pool[np.argsort(vals[pool], kind="stable")[:need]]] = True
    n = int(solved.sum())
    return {"answer": ans, "certified_in": cin, "certified_out": cout, "undecided": und0,
            "n_solved": n, "frac_avoided": 1.0 - n / max(1, M), "rounds": rounds,
            "time_certify": t_cert, "time_solve": t_solve}


def screen(graph, tau: Optional[float] = None, k: Optional[int] = None, n_jobs: Optional[int] = None,
           schedule: str = "adaptive", seed: int = 0, batch: Optional[int] = None,
           growth: float = 2.0) -> Dict[str, object]:
    """Certify-then-solve on ``graph`` from scratch (the cascade of the module
    docstring), for {c_OR < tau} or for the ``k`` most negatively curved edges.

    Returns the exact answer and the work done: edges decided by the BF
    interval alone (``n_bf``, reported but not used), by the two-hop interval
    (``n_two_hop``), by the TLW certificate (``n_tlw``, threshold queries), the
    number of transport problems solved, rounds, and wall-clock times of the
    two stages. Edges are processed in a fixed random order (``seed``) so that
    expensive edges (e.g. at hubs) are spread over the parallel chunks.
    """
    import numba

    from . import _kernels as K
    from . import intervals as I
    from .engine import _threads

    if (tau is None) == (k is None):
        raise ValueError("give exactly one of tau or k")
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
        out = {"M": M}
        if k is None:
            cin, cout, und = certify_threshold(lo, up, tau)
            _, _, und_bf = certify_threshold(blo, bup, tau)
            ids = np.flatnonzero(und)
            ids = ids[rng.permutation(ids.size)]
            r = K.screen_ids(g.indptr, g.indices, g.edges, ids, float(tau), tol, nch)
            by_tlw = np.isnan(r[:, 0])
            ans = cin.copy()
            ans[ids[by_tlw]] = True
            ans[ids[~by_tlw]] = r[~by_tlw, 0] < tau
            n_solved = int((~by_tlw).sum())
            out.update(n_bf=int(M - und_bf.sum()), n_two_hop=int(M - und.sum()), n_tlw=int(by_tlw.sum()),
                       rounds=int(ids.size > 0), time_certify=0.0)
        else:
            _, _, und_bf = certify_bottom_k(blo, bup, k)

            def exact_fn(ids):
                p = rng.permutation(ids.size)
                v = np.empty(ids.size)
                v[p] = K.exact_or_ids(g.indptr, g.indices, g.edges, np.ascontiguousarray(ids[p], dtype=np.int64), nch)
                return v

            rr = certify_then_solve(lo, up, exact_fn, tau=None, k=k, schedule=schedule, batch=batch,
                                    growth=growth, tol=tol)
            ans, n_solved = rr["answer"], rr["n_solved"]
            out.update(n_bf=int(M - und_bf.sum()), n_two_hop=int(M - rr["undecided"].sum()), n_tlw=0,
                       rounds=rr["rounds"], time_certify=rr["time_certify"])
        t2 = time.perf_counter()
    out.update(answer=ans, n_solved=n_solved, frac_avoided=1.0 - n_solved / max(1, M),
               time_stage1=t1 - t0, time_rest=t2 - t1, time_total=t2 - t0)
    return out


def certify_bottom_k(lower, upper, k: int):
    """Return (in_mask, out_mask, undecided_mask) for the k smallest c_OR."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    M = lower.shape[0]
    if not 0 <= k <= M:
        raise ValueError("k must be in [0, M]")
    if k == 0:
        z = np.zeros(M, bool)
        return z, ~z, z
    if k == M:
        z = np.zeros(M, bool)
        return ~z, z, z
    lo_sorted = np.sort(lower)
    up_sorted = np.sort(upper)
    # number of other edges f with lower_f <= upper_e (subtract e itself)
    n_le = np.searchsorted(lo_sorted, upper, side="right") - 1
    # number of other edges f with upper_f < lower_e (e itself never counts)
    n_lt = np.searchsorted(up_sorted, lower, side="left")
    cin = n_le <= k - 1
    cout = n_lt >= k
    return cin, cout, ~(cin | cout)


def certify_then_solve(lower, upper, exact_fn: Callable[[np.ndarray], np.ndarray],
                       tau: Optional[float] = 0.0, k: Optional[int] = None) -> Dict[str, object]:
    """Exact answer to a threshold query (``tau``) or a bottom-k query (``k``).

    ``exact_fn(ids)`` must return exact c_OR values for the edge ids it is
    given. Returns the answer mask, the number of transport problems solved,
    and timings.
    """
    if (tau is None) == (k is None):
        raise ValueError("give exactly one of tau or k")
    t0 = time.perf_counter()
    if k is None:
        cin, cout, und = certify_threshold(lower, upper, tau)
    else:
        cin, cout, und = certify_bottom_k(lower, upper, k)
    t1 = time.perf_counter()
    ids = np.flatnonzero(und)
    vals = np.asarray(exact_fn(ids), float) if ids.size else np.empty(0)
    t2 = time.perf_counter()
    ans = cin.copy()
    if k is None:
        ans[ids] = vals < tau
    else:
        need = k - int(cin.sum())
        if need > 0:
            take = ids[np.argsort(vals, kind="stable")[:need]]
            ans[take] = True
    return {"answer": ans, "certified_in": cin, "certified_out": cout, "undecided": und,
            "n_solved": int(ids.size), "frac_avoided": 1.0 - ids.size / max(1, len(ans)),
            "time_certify": t1 - t0, "time_solve": t2 - t1}
