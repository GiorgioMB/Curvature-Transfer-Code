"""Screen-then-solve (Algorithm 1): the edges with lazy OR curvature below a threshold, deciding as many as
possible with the bound and solving the others exactly with orcut. Every decision compares integers."""
import time
from fractions import Fraction

import numpy as np
import orcut

from .tq_lazy import tq_lazy


def screen_then_solve(graph, theta=0, n_jobs=None):
    """For an ``orcut.Graph``, the mask of its edges (in ``graph.edges`` order) with kappa < theta, for a
    rational ``theta`` (an int, a Fraction, or a float read exactly). Returns a dict with the mask, the edges
    the bound decided, and the time of the bound and of the solve stage."""
    th = Fraction(theta)
    a, b = th.numerator, th.denominator                     # kappa < theta  <=>  b (den - num) < a den
    ge = np.ascontiguousarray(np.asarray(graph.edges), np.int64)
    ip = np.ascontiguousarray(np.asarray(graph.indptr), np.int64)
    ix = np.ascontiguousarray(np.asarray(graph.indices), np.int64)
    t0 = time.perf_counter()
    out = tq_lazy(ip, ix, ge)
    t1 = time.perf_counter()
    L, Wu, Wl = (out[:, c].astype(object) for c in range(3))
    below = b * (L - Wl) < a * L                             # upper end 1 - Wl/L < theta
    above = b * (L - Wu) >= a * L                            # lower end 1 - Wu/L >= theta
    decided = np.asarray(below | above, bool)
    mask = np.asarray(below, bool).copy()
    und = np.flatnonzero(~decided)
    t2 = t1
    if len(und):
        r = orcut.curvature(graph, edges=und, exact_fraction=True, n_jobs=n_jobs)
        t2 = time.perf_counter()
        num, den = r["w1_num"].astype(object), r["w1_den"].astype(object)
        mask[und] = np.asarray(b * (den - num) < a * den, bool)
    return {"below": mask, "decided": decided, "t_bound": t1 - t0, "t_solve": t2 - t1}
