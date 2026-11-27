"""Experiments of Section 6 (Experiments) of the paper, as independent tasks.

  tightness   (graph)              non-lazy measure: exact curvature, TQ, BM and TLW, decided exactly -> Table 2
  range       (graph, chunk)       non-lazy measure: the full range of Theorem 5.2 on one chunk of edges
  measures    (graph, measure, chunk)  lazy, idleness 1/2 through Corollary 5.4, degree-biased: TQ and full range
  kstar       (graph)              lazy measure: the full range as a middle stage between TQ and orcut
  screening   (network)            lazy measure: screen-then-solve (Algorithm 1) against full evaluation -> Table 3
  benchmark   (family, size)       lazy measure: full evaluation, TQ and BF -> Figure 3

A chunk c of K holds the edges c, c + K, c + 2K, ... (round robin, so the expensive edges at hubs spread over
the chunks); every task loads its graph and computes what it needs, so tasks are independent. A task writes
results/<step>/<tag>.json (chunks: results/<step>/<tag>/<part>.json and .npz), and is skipped when its output
exists unless force=True. merge_* assemble the chunks into the per-graph results that tq_artifacts reads.
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


def _part(step, tag, key, ext):
    d = os.path.join(RESULTS, step, tag)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.{ext}")


def part_done(step, tag, key):
    return os.path.exists(_part(step, tag, key, "json"))


def save_part(step, tag, key, obj, arrays):
    np.savez_compressed(_part(step, tag, key, "npz"), **arrays)
    with open(_part(step, tag, key, "json"), "w") as fh:
        json.dump(obj, fh, indent=1, default=float)


def parts(step, tag, prefix=""):
    """(json, npz arrays) of every part of a graph whose key starts with prefix."""
    import glob
    out = []
    for pj in sorted(glob.glob(os.path.join(RESULTS, step, tag, f"{prefix}*.json"))):
        with open(pj) as fh:
            meta = json.load(fh)
        out.append((meta, np.load(pj[:-5] + ".npz", allow_pickle=True)))
    return out


def edge_slice(m, c, K):
    """Chunk c of K: the edges c, c + K, c + 2K, ... (round robin)."""
    return np.arange(c, m, K, dtype=np.int64)


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


def range_chunk(spec, c, K, threads, time_limit=60.0, budget=None, force=False):
    """Full range of Theorem 5.2 (non-lazy measure) on chunk c of K of the edges of a graph."""
    key = f"{c:05d}of{K:05d}"
    if part_done("range", spec.tag, key) and not force:
        return False
    n, E = suite.load(spec)
    g, ge, ip, ix = build(n, E)
    adj, deg = adjacency(ip, ix)
    idx = edge_slice(len(ge), c, K)
    sub = ge[idx]
    k = orcut.curvature(g, measure=Fraction(0), edges=idx, n_jobs=threads)
    o = np.empty((len(idx), 4))
    tq_general(ip, ix, np.ascontiguousarray(sub[:, 0]), np.ascontiguousarray(sub[:, 1]), 1, 0.0, 0, o,
               max(1, min(len(idx), 8 * threads)))
    D = np.minimum(deg[sub[:, 0]], deg[sub[:, 1]]) * np.maximum(deg[sub[:, 0]], deg[sub[:, 1]])
    lo, up = o[:, 0], o[:, 1]
    lo_n, up_n = snap(lo, D), snap(up, D)
    edges = [(int(u), int(v)) for u, v in sub]
    lo_s, hi_s, hi_f, solved, lo_frac, nmilp = _range_for(adj, deg, edges, 0.0, lo, up, threads, time_limit,
                                                        exact=True, budget=budget)
    hi_n = np.zeros(len(idx), np.int64)
    hi_n[solved] = snap(hi_s[solved], D[solved])
    settled_tq = (lo_n >= 0) | (up_n < 0)
    settled_r = (lo_frac[:, 0] <= lo_frac[:, 1]) | (hi_n < 0)          # exact: lower end >= 0, upper < 0
    tol = 1e-9
    viol = int(np.sum(((lo > lo_s + tol) | (lo_s > k + tol) | (k > hi_s + tol) | (hi_s > hi_f + tol)
                       | (hi_s > up + tol))[solved]))
    save_part("range", spec.tag, key,
              {"name": spec.name, "tag": spec.tag, "c": c, "K": K, "m_part": int(len(idx)),
               "milp_edges": int(nmilp), "violations": viol},
              {"idx": idx, "lo": lo, "up": up, "lo_s": lo_s, "hi_s": hi_s, "hi_f": hi_f, "solved": solved,
               "settled_tq": settled_tq, "settled_r": settled_r})
    return True


def merge_range(spec):
    """Assemble the chunks of a graph into results/range/<tag>.json (None if no chunk exists)."""
    ps = parts("range", spec.tag)
    if not ps:
        return None
    K = ps[0][0]["K"]
    have = {meta["c"] for meta, _ in ps}
    cat = {key: np.concatenate([a[key] for _, a in ps]) for key in
           ("lo", "up", "lo_s", "hi_s", "hi_f", "solved", "settled_tq", "settled_r")}
    sv = cat["solved"].astype(bool)
    res = {"name": spec.name, "tag": spec.tag, "m": int(sum(meta["m_part"] for meta, _ in ps)),
           "chunks": int(K), "chunks_done": len(have), "complete": bool(len(have) == K and sv.all()),
           "unsolved": int((~sv).sum()), "milp_edges": int(sum(meta["milp_edges"] for meta, _ in ps)),
           "tq_sign": 100 * float(cat["settled_tq"].mean()),
           "range_sign": 100 * float(cat["settled_r"][sv].mean()) if sv.any() else float("nan"),
           "range_width": float(np.mean((cat["hi_s"] - cat["lo_s"])[sv])) if sv.any() else float("nan"),
           "gap_lo": float(np.mean((cat["lo_s"] - cat["lo"])[sv])) if sv.any() else float("nan"),
           "gap_hi": float(np.mean((cat["up"] - cat["hi_s"])[sv])) if sv.any() else float("nan"),
           "slots_bind": int(np.sum(np.abs(cat["hi_f"] - cat["hi_s"])[sv] > 1e-9)),
           "violations": int(sum(meta["violations"] for meta, _ in ps))}
    save("range", spec.tag, res)
    return res


# ------------------------------------------------------------------------------------------- other measures
MEASURES = (("lazy", "lazy", (0, 0.0, 0)), ("half", "wj", (2, 0.0, 0)), ("degree", "degree", (0, 0.0, 1)))


def _kappa_slice(g, name, deg, idx, threads):
    if name == "degree":
        return orcut.curvature(g, measure=Walk(weights=1.0 / deg[np.asarray(g.indices)], idleness="lazy"),
                               edges=idx, n_jobs=threads)
    ms = {"lazy": "lazy", "half": Fraction(1, 2)}[name]
    return orcut.curvature(g, measure=ms, edges=idx, n_jobs=threads)


def measures_chunk(spec, name, c, K, threads, time_limit=60.0, budget=None, force=False):
    """TQ and the full range for one measure on chunk c of K of the edges of a graph."""
    key = f"{name}-{c:05d}of{K:05d}"
    if part_done("measures", spec.tag, key) and not force:
        return False
    fr_measure, (im, p, wm) = {m[0]: (m[1], m[2]) for m in MEASURES}[name]
    n, E = suite.load(spec)
    g, ge, ip, ix = build(n, E)
    adj, deg = adjacency(ip, ix)
    idx = edge_slice(len(ge), c, K)
    sub = ge[idx]
    k = _kappa_slice(g, name, deg, idx, threads)
    o = np.empty((len(idx), 4))
    tq_general(ip, ix, np.ascontiguousarray(sub[:, 0]), np.ascontiguousarray(sub[:, 1]), im, p, wm, o,
               max(1, min(len(idx), 8 * threads)))
    lo, up = o[:, 0], o[:, 1]
    edges = [(int(u), int(v)) for u, v in sub]
    lo_s, hi_s, hi_f, solved, _, _ = _range_for(adj, deg, edges, fr_measure, lo, up, threads, time_limit,
                                                exact=False, budget=budget)
    a_ = np.minimum(deg[sub[:, 0]], deg[sub[:, 1]]); b_ = np.maximum(deg[sub[:, 0]], deg[sub[:, 1]])
    D = grid(name, a_, b_)
    if D is not None:                   # exact decisions on the grid of the measure (before the lift)
        s_tq = (snap(lo, D) >= 0) | (snap(up, D) < 0)
        s_r = (snap(np.where(solved, lo_s, 0.0), D) >= 0) | (snap(np.where(solved, hi_s, 0.0), D) < 0)
    else:                               # no small common denominator: conservative, the same rule for both
        tol = 1e-9
        s_tq = (lo > tol) | (up < -tol)
        s_r = (lo_s > tol) | (hi_s < -tol)
    if name == "half":                  # Corollary 5.4: kappa_{1/2} = (1/2)/(1 - w_j) kappa_{w_j}
        cf = 0.5 / (1.0 - 1.0 / (b_ + 1.0))
        lo, up, lo_s, hi_s, hi_f = cf * lo, cf * up, cf * lo_s, cf * hi_s, cf * hi_f
    tol = 1e-9
    viol = int(np.sum(((lo > lo_s + tol) | (lo_s > k + tol) | (k > hi_s + tol) | (hi_s > hi_f + tol)
                       | (hi_s > up + tol))[solved]))
    save_part("measures", spec.tag, key,
              {"name": spec.name, "tag": spec.tag, "measure": name, "c": c, "K": K, "m_part": int(len(idx)),
               "violations": viol},
              {"lo": lo, "up": up, "lo_s": lo_s, "hi_s": hi_s, "solved": solved, "s_tq": s_tq, "s_r": s_r})
    return True


def merge_measures(spec):
    """Assemble the chunks of a graph into results/measures/<tag>.json (None if no chunk exists)."""
    res = {"name": spec.name, "tag": spec.tag}
    any_ = False
    for name, _, _ in MEASURES:
        ps = parts("measures", spec.tag, prefix=f"{name}-")
        if not ps:
            continue
        any_ = True
        K = ps[0][0]["K"]
        have = {meta["c"] for meta, _ in ps}
        cat = {key: np.concatenate([a[key] for _, a in ps]) for key in ("lo", "up", "lo_s", "hi_s", "solved", "s_tq", "s_r")}
        sv = cat["solved"].astype(bool)
        res["m"] = int(sum(meta["m_part"] for meta, _ in ps))
        res[name] = {"complete": bool(len(have) == K and sv.all()),
                     "violations": int(sum(meta["violations"] for meta, _ in ps)),
                     "gap_lo": float(np.mean((cat["lo_s"] - cat["lo"])[sv])) if sv.any() else float("nan"),
                     "gap_hi": float(np.mean((cat["up"] - cat["hi_s"])[sv])) if sv.any() else float("nan"),
                     "tq_sign": 100 * float(cat["s_tq"].mean()),
                     "range_sign": 100 * float(cat["s_r"][sv].mean()) if sv.any() else float("nan")}
    if not any_:
        return None
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


def benchmark_point(family, n, threads, reps, force=False):
    tag = f"{family}"; key = f"{n:09d}"
    if part_done("benchmark", tag, key) and not force:
        return False
    nn, E = suite.largest_component(*_as_ne(FAMILIES[family](n, 0)))
    g, ge, ip, ix = build(nn, E)
    orcut.curvature(g, edges=np.arange(min(5, len(ge))), n_jobs=threads)
    tq_lazy(ip, ix, ge[:5]); bf(ip, ix, ge[:5])
    t_full, _ = median_time(lambda: orcut.curvature(g, n_jobs=threads), reps)
    t_tq, _ = median_time(lambda: tq_lazy(ip, ix, ge), reps)
    t_bf, _ = median_time(lambda: bf(ip, ix, ge), reps)
    save_part("benchmark", tag, key, {"family": family, "n": int(n), "m": int(len(ge)), "t_full": t_full,
                                      "t_tq": t_tq, "t_bf": t_bf}, {"n": np.array([n])})
    return True


def merge_benchmark(family):
    ps = parts("benchmark", family)
    if not ps:
        return None
    res = {"family": family, "points": sorted((meta for meta, _ in ps), key=lambda p: p["n"])}
    save("benchmark", family, res)
    return res


def _as_ne(x):
    """Generators return (n, edges) or edges."""
    if isinstance(x, tuple) and len(x) == 2:
        return int(x[0]), np.asarray(x[1], np.int64)
    e = np.asarray(x, np.int64)
    return int(e.max()) + 1, e
