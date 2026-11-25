"""Experiments of Section 6 (Experiments) of the paper.

Steps (each writes results/<step>/<tag>.json, and .npz with per-edge arrays where later steps need them;
a graph already done is skipped unless force=True):

  tightness   non-lazy measure, every registry graph: exact curvature (orcut, integer fractions), TQ, and the
              bounds of BM and TLW, all decided exactly -> Table 2 (BM, TLW, TQ)
  range       non-lazy measure: the full range of Theorem 5.2 (orcut on local graphs, integer program)
              -> Table 2 (full range), GapLoMean, GapHiMean, SignGapMax
  measures    lazy measure, idleness 1/2 through Corollary 5.4, degree-biased walk: TQ and the full range
              -> MeasGapLo, MeasGapHi, MeasSignGap
  kstar       lazy measure: the full range as a middle stage between TQ and orcut -> KStarDec*, KStarSlow*
  screening   lazy measure, large networks: screen-then-solve (Algorithm 1) against full evaluation -> Table 3
  benchmark   lazy measure, four random families of growing size: full evaluation, TQ, BF -> Figure 3
"""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import orcut
from orcut.measures import Walk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import generators as gen                                   # noqa: E402
import large_graphs                                        # noqa: E402
import suite                                               # noqa: E402
from fractions import Fraction                             # noqa: E402
from tqbound.baselines import bm_tlw, snap                  # noqa: E402
from tqbound.full_range import (                            # noqa: E402
    chunks_by_size, local_batch, lower_and_relaxed, upper_end)
from tqbound.screen import screen_then_solve                # noqa: E402
from tqbound.tq_general import tq_general                   # noqa: E402
from tqbound.tq_lazy import bf, tq_lazy                     # noqa: E402

RESULTS = os.path.join(ROOT, "results")


# ------------------------------------------------------------------------------------------------- helpers
def _path(step, tag, ext="json"):
    d = os.path.join(RESULTS, step)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tag}.{ext}")


def done(step, tag):
    return os.path.exists(_path(step, tag))


def save(step, tag, obj, arrays=None):
    if arrays is not None:
        np.savez_compressed(_path(step, tag, "npz"), **arrays)
    with open(_path(step, tag), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def load_json(step, tag):
    with open(_path(step, tag)) as fh:
        return json.load(fh)


def load_npz(step, tag):
    return np.load(_path(step, tag, "npz"), allow_pickle=True)


def build(n, E):
    """orcut graph (sorted CSR, canonical edges) and the arrays the kernels read."""
    g = orcut.Graph.from_edges(np.asarray(E, np.int64), num_nodes=n)
    ge = np.ascontiguousarray(np.asarray(g.edges), np.int64)
    ip = np.ascontiguousarray(np.asarray(g.indptr), np.int64)
    ix = np.ascontiguousarray(np.asarray(g.indices), np.int64)
    return g, ge, ip, ix


def adjacency(ip, ix):
    n = len(ip) - 1
    return [set(ix[ip[v]:ip[v + 1]].tolist()) for v in range(n)], np.diff(ip)


def median_time(f, reps):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def grid(measure, a, b):
    """Denominator of the masses of a one-step measure on an edge with degrees a <= b (None: not rational with
    a small denominator); bounds and curvatures are multiples of 1/grid, so decisions are exact after snap."""
    if measure == "nonlazy":
        return a * b
    if measure == "lazy":
        return (a + 1) * (b + 1)
    if measure == "half":
        return a * (b + 1)          # idleness w_j = 1/(b + 1) at both endpoints, before the lift
    return None


def exact_kappa(g, measure, deg, threads):
    """Exact curvature with integer fractions (W1 = num/den) for 'nonlazy', 'lazy', 'half'; floats for 'degree'."""
    if measure == "degree":
        w = 1.0 / deg[np.asarray(g.indices)]
        return orcut.curvature(g, measure=Walk(weights=w, idleness="lazy"), n_jobs=threads), None, None
    ms = {"nonlazy": Fraction(0), "lazy": "lazy", "half": Fraction(1, 2)}[measure]
    r = orcut.curvature(g, measure=ms, exact_fraction=True, n_jobs=threads)
    return r["curvature"], r["w1_num"].astype(object), r["w1_den"].astype(object)


def exact_outside(lo_n, up_n, D, num, den):
    """Edges whose exact curvature 1 - num/den lies outside [lo_n/D, up_n/D]."""
    lo_n, up_n, D = (np.asarray(x).astype(object) for x in (lo_n, up_n, D))
    return int(np.sum((lo_n * den > (den - num) * D) | (up_n * den < (den - num) * D)))


def environment(threads, reps):
    import numba
    return {"orcut": orcut.__version__, "numba": numba.__version__, "numpy": np.__version__,
            "threads": int(threads), "reps": int(reps)}


# ---------------------------------------------------------------------------------------------- tightness
def tightness(spec, threads, force=False):
    if done("tightness", spec.tag) and not force:
        return load_json("tightness", spec.tag)
    n, E = suite.load(spec)
    g, ge, ip, ix = build(n, E)
    deg = np.diff(ip)
    k, num, den = exact_kappa(g, "nonlazy", deg, threads)
    o = np.empty((len(ge), 4))
    tq_general(ip, ix, np.ascontiguousarray(ge[:, 0]), np.ascontiguousarray(ge[:, 1]), 1, 0.0, 0, o,
               max(1, min(len(ge), 8 * threads)))
    B = bm_tlw(ip, ix, ge)
    D = B["den"]
    lo, up = snap(o[:, 0], D), snap(o[:, 1], D)
    bm, tl, ub = snap(B["bm_lower"], D), snap(B["tlw_lower"], D), snap(B["upper"], D)
    res = {"name": spec.name, "tag": spec.tag, "group": spec.group, "n": int(n), "m": int(len(ge)),
           "dmax": int(deg.max()),
           "tq_sign": 100 * float(np.mean((lo >= 0) | (up < 0))), "tq_width": float(np.mean((up - lo) / D)),
           "bm_sign": 100 * float(np.mean((bm >= 0) | (ub < 0))), "bm_width": float(np.mean((ub - bm) / D)),
           "tlw_sign": 100 * float(np.mean((tl >= 0) | (ub < 0))), "tlw_width": float(np.mean((ub - tl) / D)),
           "violations": {"tq": exact_outside(lo, up, D, num, den), "bm": exact_outside(bm, ub, D, num, den),
                          "tlw": exact_outside(tl, ub, D, num, den)}}
    save("tightness", spec.tag, res, {"edges": ge, "k": k, "num": num, "den": den, "D": D, "tq_lo": lo, "tq_up": up})
    return res


# ----------------------------------------------------------------------------------------- full range
_W = {}


def _milp_task(args):
    t, u, v, measure, limit = args
    val, solved = upper_end(_W["adj"], _W["deg"], measure, u, v, time_limit=limit)
    return t, val, solved


def _upper_ends(adj, deg, edges, idx, measure, threads, limit, budget=None):
    """kappa-bar* for the edges idx (integer program, in parallel, at most ``limit`` seconds each); after
    ``budget`` seconds the remaining edges are left unsolved. Returns (values, solved)."""
    vals = np.full(len(idx), np.nan); ok = np.zeros(len(idx), bool)
    if len(idx) == 0:
        return vals, ok
    _W["adj"], _W["deg"] = adj, deg
    tasks = [(t, edges[e][0], edges[e][1], measure, limit) for t, e in enumerate(idx)]
    t0 = time.perf_counter()
    if threads > 1:
        pool = get_context("fork").Pool(threads)
        try:
            for t, val, solved in pool.imap_unordered(_milp_task, tasks, chunksize=1):
                vals[t] = val; ok[t] = solved
                if budget and time.perf_counter() - t0 > budget:
                    break
        finally:
            pool.terminate()
    else:
        for task in tasks:
            t, val, solved = _milp_task(task)
            vals[t] = val; ok[t] = solved
            if budget and time.perf_counter() - t0 > budget:
                break
    return vals, ok


def _range_for(adj, deg, edges, measure, lo_tq, up_tq, threads, limit, exact, budget=None):
    """Full range for every edge: lower end, relaxation, upper end (integer program only where it can differ
    from the lower end, that is, where both the relaxation and the TQ upper end exceed the lower end)."""
    r = lower_and_relaxed(adj, deg, edges, measure, n_jobs=threads, exact=exact)
    lo_s, hi_f = r[0], r[1]
    need = np.flatnonzero((hi_f - lo_s > 1e-12) & (up_tq - lo_s > 1e-12))
    hi_s = lo_s.copy()
    vals, ok = _upper_ends(adj, deg, edges, need, measure, threads, limit, budget)
    hi_s[need] = vals
    solved = np.ones(len(edges), bool); solved[need] = ok
    lo_frac = r[2] if exact else None
    return lo_s, hi_s, hi_f, solved, lo_frac, len(need)


def full_range(spec, threads, time_limit=60.0, force=False, budget=None):
    if done("range", spec.tag) and not force:
        return load_json("range", spec.tag)
    T = load_npz("tightness", spec.tag)
    ge, k, D = T["edges"], T["k"], T["D"]
    lo_n, up_n = T["tq_lo"], T["tq_up"]
    lo, up = lo_n / D, up_n / D
    n = int(ge.max()) + 1
    _, _, ip, ix = build(n, ge)
    adj, deg = adjacency(ip, ix)
    edges = [(int(a), int(b)) for a, b in ge]
    t0 = time.perf_counter()
    lo_s, hi_s, hi_f, solved, lo_frac, nmilp = _range_for(adj, deg, edges, 0.0, lo, up, threads, time_limit,
                                                        exact=True, budget=budget)
    secs = time.perf_counter() - t0
    hi_n = np.zeros(len(ge), np.int64)
    hi_n[solved] = snap(hi_s[solved], D[solved])
    lo_nonneg = lo_frac[:, 0] <= lo_frac[:, 1]                   # exact: kappa_lower* >= 0
    settled_r = lo_nonneg | (hi_n < 0)
    settled_tq = (lo_n >= 0) | (up_n < 0)
    tol = 1e-9
    viol = int(np.sum(((lo > lo_s + tol) | (lo_s > k + tol) | (k > hi_s + tol) | (hi_s > hi_f + tol)
                       | (hi_s > up + tol))[solved]))
    res = {"name": spec.name, "tag": spec.tag, "m": int(len(ge)), "complete": bool(solved.all()),
           "unsolved": int((~solved).sum()), "milp_edges": int(nmilp), "seconds": secs,
           "tq_sign": 100 * float(settled_tq.mean()), "range_sign": 100 * float(settled_r[solved].mean()),
           "range_width": float(np.mean((hi_s - lo_s)[solved])),
           "gap_lo": float(np.mean((lo_s - lo)[solved])), "gap_hi": float(np.mean((up - hi_s)[solved])),
           "slots_bind": int(np.sum(np.abs(hi_f - hi_s)[solved] > 1e-9)), "violations": viol}
    save("range", spec.tag, res, {"lo_star": lo_s, "hi_star": hi_s, "hi_free": hi_f, "solved": solved})
    return res


# ------------------------------------------------------------------------------------------- other measures
MEASURES = (("lazy", "lazy", (0, 0.0, 0)), ("half", "wj", (2, 0.0, 0)), ("degree", "degree", (0, 0.0, 1)))


def measures(spec, threads, time_limit=60.0, force=False, budget=None):
    if done("measures", spec.tag) and not force:
        return load_json("measures", spec.tag)
    n, E = suite.load(spec)
    g, ge, ip, ix = build(n, E)
    adj, deg = adjacency(ip, ix)
    edges = [(int(a), int(b)) for a, b in ge]
    a_ = np.minimum(deg[ge[:, 0]], deg[ge[:, 1]]); b_ = np.maximum(deg[ge[:, 0]], deg[ge[:, 1]])
    res = {"name": spec.name, "tag": spec.tag, "m": int(len(ge))}
    for name, fr_measure, (im, p, wm) in MEASURES:
        k, _, _ = exact_kappa(g, name, deg, threads)
        o = np.empty((len(ge), 4))
        tq_general(ip, ix, np.ascontiguousarray(ge[:, 0]), np.ascontiguousarray(ge[:, 1]), im, p, wm, o,
                   max(1, min(len(ge), 8 * threads)))
        lo, up = o[:, 0], o[:, 1]
        lo_s, hi_s, hi_f, solved, _, _ = _range_for(adj, deg, edges, fr_measure, lo, up, threads, time_limit,
                                                    exact=False, budget=budget)
        D = grid(name, a_, b_)
        if D is not None:               # exact decisions on the grid of the measure (before the lift)
            s_tq = (snap(lo, D) >= 0) | (snap(up, D) < 0)
            lo_ok = np.where(solved, lo_s, 0.0); hi_ok = np.where(solved, hi_s, 0.0)
            s_r = (snap(lo_ok, D) >= 0) | (snap(hi_ok, D) < 0)
        else:
            tol = 1e-9                  # no small common denominator: conservative, the same rule for both
            s_tq = (lo > tol) | (up < -tol)
            s_r = (lo_s > tol) | (hi_s < -tol)
        if name == "half":              # Corollary 5.4: kappa_{1/2} = (1/2)/(1 - w_j) kappa_{w_j}
            c = 0.5 / (1.0 - 1.0 / (b_ + 1.0))
            lo, up, lo_s, hi_s, hi_f = c * lo, c * up, c * lo_s, c * hi_s, c * hi_f
        tol = 1e-9
        viol = int(np.sum(((lo > lo_s + tol) | (lo_s > k + tol) | (k > hi_s + tol) | (hi_s > hi_f + tol)
                           | (hi_s > up + tol))[solved]))
        res[name] = {"complete": bool(solved.all()), "violations": viol,
                     "gap_lo": float(np.mean((lo_s - lo)[solved])), "gap_hi": float(np.mean((up - hi_s)[solved])),
                     "tq_sign": 100 * float(s_tq.mean()), "range_sign": 100 * float(s_r[solved].mean())}
    save("measures", spec.tag, res)
    return res


# ------------------------------------------------------------------------------------- the kappa* stage
def kstar(tag, name, n, E, threads, reps, force=False):
    if done("kstar", tag) and not force:
        return load_json("kstar", tag)
    g, ge, ip, ix = build(n, E)
    out = tq_lazy(ip, ix, ge)
    L, Wu, Wl = out[:, 0], out[:, 1], out[:, 2]
    und = np.flatnonzero(~((Wl > L) | (Wu <= L)))
    res = {"name": name, "tag": tag, "m": int(len(ge)), "undecided": int(len(und))}
    if len(und) == 0:
        save("kstar", tag, res)
        return res
    orcut.curvature(g, edges=und[:1], n_jobs=threads)
    t_skip, _ = median_time(lambda: orcut.curvature(g, edges=und, exact_fraction=True, n_jobs=threads), reps)
    adj, deg = adjacency(ip, ix)
    edges = [(int(ge[e, 0]), int(ge[e, 1])) for e in und]
    t_lo = t_hi = 0.0
    dec = np.zeros(len(und), bool)
    for ch in chunks_by_size(adj, deg, edges):                      # batches that fit in memory
        sub = [edges[c] for c in ch]
        ga, ida, _ = local_batch(adj, deg, sub, friendly=False)     # construction is not timed
        gf, idf, _ = local_batch(adj, deg, sub, friendly=True)
        ta, ra = median_time(lambda: orcut.curvature(ga, edges=ida, exact_fraction=True, n_jobs=threads), reps)
        tf, rf = median_time(lambda: orcut.curvature(gf, edges=idf, exact_fraction=True, n_jobs=threads), reps)
        t_lo += ta; t_hi += tf
        dec[np.array(ch)] = (ra["w1_num"] <= ra["w1_den"]) | (rf["w1_num"] > rf["w1_den"])
    rest = und[~dec]
    t_rest = 0.0
    if len(rest):
        t_rest, _ = median_time(lambda: orcut.curvature(g, edges=rest, exact_fraction=True, n_jobs=threads),
                                reps)
    res.update({"decided_by_kstar": 100 * float(dec.mean()), "t_orcut_undecided": t_skip,
                "t_kstar": t_lo + t_hi, "t_orcut_rest": t_rest,
                "slowdown": (t_lo + t_hi + t_rest) / t_skip})
    save("kstar", tag, res)
    return res


# ------------------------------------------------------------------------------------------- screening
def screening(tag, name, n, E, threads, reps, force=False):
    if done("screening", tag) and not force:
        return load_json("screening", tag)
    g, ge, ip, ix = build(n, E)
    orcut.curvature(g, edges=np.arange(min(5, len(ge))), n_jobs=threads)
    tq_lazy(ip, ix, ge[:5])
    t_full, rf = median_time(lambda: orcut.curvature(g, exact_fraction=True, n_jobs=threads), reps)
    runs = [screen_then_solve(g, 0, n_jobs=threads) for _ in range(reps)]
    t_tq = float(np.median([r["t_bound"] for r in runs])); t_solve = float(np.median([r["t_solve"] for r in runs]))
    last = runs[-1]
    truth = rf["w1_num"] > rf["w1_den"]
    res = {"name": name, "tag": tag, "n": int(n), "m": int(len(ge)), "dmax": int(np.diff(ip).max()),
           "t_full": t_full, "t_tq": t_tq, "t_solve": t_solve, "decided": 100 * float(last["decided"].mean()),
           "tq_frac": 100 * t_tq / t_full, "speedup": t_full / (t_tq + t_solve),
           "exact_answer": bool(np.array_equal(last["below"], truth)), "negative_edges": int(truth.sum())}
    save("screening", tag, res)
    return res


# ------------------------------------------------------------------------------------------- benchmark
FAMILIES = {
    "er": lambda n, s: gen.erdos_renyi(n, 10.0 / n, seed=s),
    "ba": lambda n, s: gen.barabasi_albert(n, 5, seed=s),
    "rgg": lambda n, s: gen.random_geometric(n, (10.0 / (np.pi * n)) ** 0.5, seed=s),
    "ws": lambda n, s: gen.watts_strogatz(n, 10, 0.2, seed=s),
}


def benchmark(family, sizes, threads, reps, force=False):
    tag = family
    old = load_json("benchmark", tag) if done("benchmark", tag) and not force else {"family": family, "points": []}
    have = {p["n"] for p in old["points"]}
    for n in sizes:
        if n in have:
            continue
        nn, E = suite.largest_component(*_as_ne(FAMILIES[family](n, 0)))
        g, ge, ip, ix = build(nn, E)
        orcut.curvature(g, edges=np.arange(min(5, len(ge))), n_jobs=threads)
        tq_lazy(ip, ix, ge[:5]); bf(ip, ix, ge[:5])
        t_full, _ = median_time(lambda: orcut.curvature(g, n_jobs=threads), reps)
        t_tq, _ = median_time(lambda: tq_lazy(ip, ix, ge), reps)
        t_bf, _ = median_time(lambda: bf(ip, ix, ge), reps)
        old["points"].append({"n": int(n), "m": int(len(ge)), "t_full": t_full, "t_tq": t_tq, "t_bf": t_bf})
        old["points"].sort(key=lambda p: p["n"])
        save("benchmark", tag, old)
    return old


def _as_ne(x):
    """Generators return (n, edges) or edges."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), np.asarray(x[1], np.int64)
    e = np.asarray(x, np.int64)
    return int(e.max()) + 1, e
