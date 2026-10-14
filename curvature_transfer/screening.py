"""Certified screening (Section 7.3): answer threshold and bottom-k queries on
OR curvature exactly, solving transport problems only on undecided edges.

Given certified intervals lower <= c_OR <= upper:

* threshold query {e : c_OR(e) < tau}: an edge with upper < tau is certainly
  in, an edge with lower >= tau is certainly out, the rest are undecided;
* bottom-k query (the k most negatively curved edges, ties broken arbitrarily):
  e is certainly in if at most k - 1 other edges f have lower_f <= upper_e
  (then at most k - 1 edges can have c_OR <= c_OR(e)), and certainly out if at
  least k other edges have upper_f < lower_e (then k edges are strictly
  smaller). Both rules are valid under any tie-breaking.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

__all__ = ["certify_threshold", "certify_bottom_k", "certify_then_solve"]


def certify_threshold(lower, upper, tau: float = 0.0):
    """Return (in_mask, out_mask, undecided_mask) for {c_OR < tau}."""
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    if np.any(lower > upper + 1e-12):
        raise ValueError("lower > upper on some edge: the intervals are not valid")
    cin = upper < tau
    cout = lower >= tau
    return cin, cout, ~(cin | cout)


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
