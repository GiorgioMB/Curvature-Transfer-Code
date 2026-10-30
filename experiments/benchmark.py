#!/usr/bin/env python3
"""Runtime scaling of exact OR curvature versus the certified bounds (Section 6.4).

Families. Four random graph families with the same scaling law: the average
degree grows as d(n) = 0.2 (50/n)^0.6 n ~ 2.1 n^0.4 (the law of the ER example
of the paper), so that graph size and density grow together:

  ER   Erdos-Renyi G(n, p),          p = d / (n - 1)          homogeneous degrees
  BA   Barabasi-Albert,              m = round(d / 2)         heavy-tailed degrees (hubs)
  RGG  random geometric, unit square, r = sqrt(d / (pi (n-1)))  clustered, many triangles
  WS   Watts-Strogatz, beta = 0.2,   k = 2 round(d / 2)       clustered ring lattice

The generators here are vectorized (numpy / a k-d tree), so a graph with 10^5-10^6
nodes is generated in seconds; they are seeded per (seed, family, n), so a restart
regenerates exactly the same graph. (They are not the generators of the bound
tables, which are kept unchanged in generators.py.)

Phases (``timing.PHASES``, the same code as the speedup column of the screening
table): exact (exact lazy OR, min-cut), bf, prior (JL and TLW lower bounds),
bf_interval, matching_interval, sharpened_interval (the intervals of Section 6.1;
the last two include the TLW dual certificate, evaluated by the 2-hop kernel in
the same pass as the statistics). Cheap phases run first.

How a phase is timed. Graphs with at most ``--chunk-above`` edges: the median of
repeated runs on all edges (``--min-total``, ``--max-reps``). Larger graphs: the
sum of the times of consecutive chunks of ``--chunk`` edges, each run once (the
CSV column ``chunked`` says which); the figure marks where chunking starts.

Checkpointing. After every phase, and inside a chunked phase after every chunk,
the measurement is written (atomically) to ``rows_<...>/n<n>.partial.json``; the
finished size goes to ``n<n>.json``. A killed job resumes at the first chunk not
yet measured. Each size is processed in increasing order of n.

Infeasible phases are skipped, not crashed on, and once skipped for a family a
phase stays skipped for every larger n:

* time: the phase's time is extrapolated from the last measured sizes of the
  same family (log-log fit on the last 3 points, slope >= 1); if the prediction
  exceeds ``--budget-hours``, the phase is not run;
* memory: the exact solver holds per-edge bitsets of order |S|^2 / 8 bytes,
  |S| <= deg(i) + deg(j). If threads x 4 x |S|_max^2 bytes (a rough estimate)
  exceeds 80% of ``--mem-gb``, exact OR is not run. The other phases need O(n)
  stamps per thread and O(deg) scratch per edge (plus the cross edges of the
  edge), and are not guarded;
* kills: if the same chunk was started ``--max-attempts`` times without finishing
  (out of memory, or a wall-time kill that hit the same long chunk twice), the
  phase is not run for this and larger sizes.

A phase that is not run has an empty time, its reason in ``<phase>_status``, and
the extrapolated time in ``<phase>_pred_s``, drawn as hollow markers on a dotted
line ("predicted, not run").

Usage
  python benchmark.py --family BA --min-n 50 --max-n 200000 --graphs 100 --jobs 32 --out ../results/benchmark
  python benchmark.py --plot-only --out ../results/benchmark      # redraw every family's figure
Families are independent (separate row folders), so they can run as parallel jobs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from curvature_transfer import Graph  # noqa: E402
from curvature_transfer import _io  # noqa: E402
import timing  # noqa: E402

FAMILIES = ("ER", "BA", "RGG", "WS")
FAMILY_TITLES = {"ER": "Erdős-Rényi", "BA": "Barabási-Albert", "RGG": "Random geometric",
                 "WS": r"Watts-Strogatz ($\beta=0.2$)"}
PHASE_ORDER = ["bf", "prior", "bf_interval", "matching_interval", "sharpened_interval", "exact"]
PHASE_LABELS = {"exact": "Exact OR (min-cut)", "sharpened_interval": "Sharpened interval",
                "matching_interval": "Matching interval", "bf_interval": "BF interval",
                "bf": "Balanced Forman", "prior": "JL / TLW lower bounds"}
PHASE_STYLE = {"exact": ("o", "-"), "sharpened_interval": ("s", "--"), "matching_interval": ("P", "--"),
               "bf_interval": ("D", "-."), "bf": ("^", ":"), "prior": ("v", "-.")}
MEM_FACTOR = {"exact": 4.0}  # bytes per |S|^2 per thread (the intervals need no |S|^2 structure)
FAMILY_CODE = {"ER": 1, "BA": 2, "RGG": 3, "WS": 4}
GEN_VERSION = 1
COLS = (["family", "n", "param", "num_edges", "avg_deg", "max_deg", "s_max", "threads", "chunked", "chunk",
         "chunk_above", "budget_hours", "mem_gb", "gen_s"]
        + [f"{p}_{x}" for p in PHASE_ORDER for x in ("s", "pred_s", "status")] + ["legacy_s_extrapolated"])


# ------------------------------------------------------------ generators
def target_degree(n):
    return 0.2 * (50 / n) ** 0.6 * n  # ~ 2.1 n^0.4; ER of the paper: p = 0.2 (50/n)^0.6


def _rng(family, n, seed):
    return np.random.default_rng([int(seed), FAMILY_CODE[family], int(n), GEN_VERSION])


def gen_er(n, p, rng):
    """G(n, p): M ~ Binomial(N, p) distinct pairs, uniformly (exact law)."""
    N = n * (n - 1) // 2
    M = int(rng.binomial(N, p))
    keys = np.empty(0, np.int64)
    while len(keys) < M:
        extra = rng.integers(0, N, size=int(1.05 * (M - len(keys))) + 1000, dtype=np.int64)
        keys = np.unique(np.concatenate([keys, extra]))
    keys = np.sort(rng.choice(keys, M, replace=False))
    j = np.floor((1 + np.sqrt(1 + 8 * keys.astype(np.float64))) / 2).astype(np.int64)
    j = np.where(j * (j - 1) // 2 > keys, j - 1, j)
    j = np.where((j + 1) * j // 2 <= keys, j + 1, j)
    i = keys - j * (j - 1) // 2
    return np.stack([i, j], axis=1)


def gen_ba(n, m, rng):
    """Preferential attachment from an (m+1)-clique; targets drawn from the list of
    edge endpoints (probability proportional to degree), without repetition."""
    m = max(1, min(m, n - 1))
    c = [(u, v) for u in range(m + 1) for v in range(u + 1, m + 1)]
    E = np.empty((len(c) + m * (n - m - 1), 2), np.int64)
    E[:len(c)] = c
    rep = np.empty(2 * len(E), np.int64)
    rep[:2 * len(c)] = np.array(c, np.int64).ravel()
    L, e = 2 * len(c), len(c)
    for new in range(m + 1, n):
        t = np.unique(rep[rng.integers(0, L, size=m)])
        while len(t) < m:
            t = np.unique(np.concatenate([t, rep[rng.integers(0, L, size=m - len(t))]]))
        E[e:e + m, 0] = t; E[e:e + m, 1] = new
        rep[L:L + m] = t; rep[L + m:L + 2 * m] = new
        L += 2 * m; e += m
    return E


def gen_rgg(n, r, rng):
    from scipy.spatial import cKDTree
    pts = rng.random((n, 2))
    return np.asarray(cKDTree(pts).query_pairs(r, output_type="ndarray"), np.int64).reshape(-1, 2)


def gen_ws(n, k, beta, rng):
    """Ring lattice (k/2 neighbors per side); each lattice edge (u, u+d) is rewired with
    probability beta to (u, w), w uniform, avoiding self-loops and multi-edges."""
    h = k // 2
    U = np.repeat(np.arange(n, dtype=np.int64), h)
    V = (U + np.tile(np.arange(1, h + 1, dtype=np.int64), n)) % n
    rew = rng.random(len(U)) < beta
    V[rew] = rng.integers(0, n, size=int(rew.sum()))
    for _ in range(1000):
        a, b = np.minimum(U, V), np.maximum(U, V)
        key = a * n + b
        order = np.lexsort((rew, key))  # among equal keys, the non-rewired edge comes first
        dup = np.zeros(len(U), bool)
        dup[order[1:]] = key[order[1:]] == key[order[:-1]]
        bad = rew & ((U == V) | dup)
        if not bad.any():
            break
        V[bad] = rng.integers(0, n, size=int(bad.sum()))
    return np.stack([np.minimum(U, V), np.maximum(U, V)], axis=1)


def make_graph(family, n, seed):
    """(parameter, edges) with average degree ~ target_degree(n)."""
    d = target_degree(n)
    rng = _rng(family, n, seed)
    if family == "ER":
        p = min(1.0, d / (n - 1))
        return p, gen_er(n, p, rng)
    if family == "BA":
        m = max(1, int(round(d / 2)))
        return m, gen_ba(n, m, rng)
    if family == "RGG":
        r = float(np.sqrt(d / (np.pi * (n - 1))))
        return r, gen_rgg(n, r, rng)
    if family == "WS":
        k = max(2, 2 * int(round(d / 2)))
        k = min(k, 2 * ((n - 1) // 2))
        return k, gen_ws(n, k, 0.2, rng)
    raise ValueError(family)


# --------------------------------------------------------------- helpers
def mem_budget_gb(arg):
    if arg:
        return float(arg)
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return float(os.environ["SLURM_MEM_PER_NODE"]) / 1024
    if os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_PER_TASK"):
        return float(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ["SLURM_CPUS_PER_TASK"]) / 1024
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 64.0


def predict(rows, phase, n):
    """Extrapolated time of ``phase`` at size n from the measured sizes (log-log fit on
    the last 3 points, slope >= 1); None with fewer than 2 points."""
    pts = [(int(r["n"]), float(r[f"{phase}_s"])) for r in rows
           if r.get(f"{phase}_status") == "measured" and int(r["n"]) < n and float(r[f"{phase}_s"]) > 0]
    pts = sorted(pts)[-3:]
    if len(pts) < 2:
        return None
    x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    slope = max(1.0, float(np.polyfit(x, y, 1)[0]))
    return float(pts[-1][1] * (n / pts[-1][0]) ** slope)


def time_legacy(legacy_dir, n, edges, sample, seed):
    """Preliminary solver on a sample of edges, extrapolated (sequential)."""
    import importlib
    import types
    stubs = {}
    if "torch" not in sys.modules:  # the old module imports torch only for typing
        torch = types.ModuleType("torch"); torch.Tensor = object; torch.long = "long"
        tgd = types.ModuleType("torch_geometric.data"); tgd.Data = object
        stubs = {"torch": torch, "torch_geometric": types.ModuleType("torch_geometric"), "torch_geometric.data": tgd}
        sys.modules.update(stubs)
    sys.path.insert(0, legacy_dir)
    try:
        old = importlib.import_module("pyg_curvature")
    finally:
        for k in stubs:
            sys.modules.pop(k, None)
    eng = old.CurvatureEngine.__new__(old.CurvatureEngine)
    und = sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges if a != b})
    nb = [set() for _ in range(n)]
    for a, b in und:
        nb[a].add(b); nb[b].add(a)
    eng.edges, eng.neighbors, eng._cache = und, nb, {}
    eng.deg = np.array([len(s) for s in nb])
    eng.c_OR_edge(0)  # compile
    ids = np.random.default_rng(seed).choice(len(und), min(sample, len(und)), replace=False)
    t = time.perf_counter()
    for k in ids:
        eng.c_OR_edge(int(k))
    return (time.perf_counter() - t) / len(ids) * len(und)


def read_rows(rows_dir):
    rows = [_io.read_json(p) for p in glob.glob(os.path.join(rows_dir, "n*.json")) if not p.endswith(".partial.json")]
    return sorted([r for r in rows if r], key=lambda r: int(r["n"]))


def write_csv(rows, path):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in COLS})
    _io.atomic_text(buf.getvalue(), path)


# ------------------------------------------------------------------ run
def run_size(family, n, seed, n_jobs, threads, rows_dir, prev, a, mem_gb):
    """Time every phase on the graph of size n (resumable); returns the finished row."""
    p_part = os.path.join(rows_dir, f"n{n}.partial.json")
    t0 = time.perf_counter()
    param, edges = make_graph(family, n, seed)
    g = Graph.from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n)
    gen_s = time.perf_counter() - t0
    if g.num_edges == 0:
        return None
    s_max = int((g.deg[g.edges[:, 0]] + g.deg[g.edges[:, 1]]).max())
    chunked = g.num_edges > a.chunk_above
    fp = _io.fingerprint(family, n, seed, g.edges, chunked, a.chunk, length=12)
    part = _io.read_json(p_part)
    if not part or part.get("fingerprint") != fp:
        part = {"fingerprint": fp, "done": {}, "chunks": {}, "status": {}, "pred": {}, "attempts": {}}
    save = lambda: _io.atomic_json(part, p_part)
    log = lambda msg: print(f"[benchmark] {family} n={n} M={g.num_edges}: {msg}", flush=True)
    log(f"generated in {gen_s:.1f}s; max |S| = {s_max}; {'chunked' if chunked else 'whole-graph'} timing")
    budget = a.budget_hours * 3600
    for ph in PHASE_ORDER:
        if ph in part["status"]:
            continue
        pred = predict(prev, ph, n)
        part["pred"][ph] = pred
        earlier = [r for r in prev if str(r.get(f"{ph}_status", "")).startswith("skipped")]
        reason = None
        if earlier:
            reason = f"skipped (not run from n={int(earlier[0]['n'])} on)"
        elif pred is not None and pred > budget:
            reason = f"skipped (predicted {pred:.3g} s > budget {budget:.3g} s)"
        elif not a.no_mem_guard and ph in MEM_FACTOR:
            need = threads * MEM_FACTOR[ph] * s_max ** 2 / 2 ** 30
            if need > 0.8 * mem_gb:
                reason = f"skipped (memory: ~{need:.0f} GB estimated for {threads} threads at |S|_max={s_max} > 80% of {mem_gb:.0f} GB)"
        if reason is None:
            att = part["attempts"].setdefault(ph, {})
            killed = [k for k, c in att.items() if c >= a.max_attempts]
            if killed:
                reason = f"skipped (killed {a.max_attempts} times in chunk {killed[0]}: out of memory or wall time)"
        if reason:
            part["status"][ph] = reason
            save(); log(f"{ph}: {reason}")
            continue
        f = timing.PHASES[ph][0]
        if not chunked:
            att = part["attempts"][ph]
            att["all"] = att.get("all", 0) + 1; save()
            t, _, reps = timing.repeat(lambda: f(g, None, n_jobs), a.min_total, a.max_reps)
        else:
            done = {int(k): v for k, v in part["chunks"].get(ph, {}).items()}
            M, C = g.num_edges, a.chunk
            nch = -(-M // C)
            if done:
                log(f"{ph}: resuming at {len(done)}/{nch} chunks already timed")
            for k in range(nch):  # chunk by chunk, so a kill loses at most one chunk
                if k in done:
                    continue
                att = part["attempts"][ph]
                att[str(k)] = att.get(str(k), 0) + 1
                save()  # recorded before the chunk starts: a chunk killed twice is given up
                ids = np.arange(k * C, min(M, (k + 1) * C), dtype=np.int64)
                tc = time.perf_counter()
                f(g, ids, n_jobs)
                done[k] = time.perf_counter() - tc
                part["chunks"].setdefault(ph, {})[str(k)] = done[k]
                att.pop(str(k), None)
                save()
            t = float(sum(done.values()))
        part["done"][ph], part["status"][ph] = t, "measured"
        part["attempts"].pop(ph, None)
        save(); log(f"{ph}: {t:.4g} s")
    row = {"family": family, "n": n, "param": param, "num_edges": g.num_edges,
           "avg_deg": round(2 * g.num_edges / n, 2), "max_deg": int(g.deg.max()), "s_max": s_max,
           "threads": threads, "chunked": chunked, "chunk": a.chunk if chunked else None,
           "chunk_above": a.chunk_above, "budget_hours": a.budget_hours, "mem_gb": round(mem_gb, 1), "gen_s": gen_s}
    for ph in PHASE_ORDER:
        row[f"{ph}_s"] = part["done"].get(ph, float("nan"))
        row[f"{ph}_pred_s"] = part["pred"].get(ph) if part["status"][ph] != "measured" else None
        row[f"{ph}_status"] = part["status"][ph]
    row["legacy_s_extrapolated"] = (time_legacy(a.legacy_dir, n, edges, a.legacy_sample, seed)
                                    if a.legacy_dir and not chunked else float("nan"))
    return row


def run_family(family, a):
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    timing.warmup()
    import numba
    threads = numba.get_num_threads() if a.jobs is None else (numba.config.NUMBA_NUM_THREADS if a.jobs == -1 else a.jobs)
    mem_gb = mem_budget_gb(a.mem_gb)
    cfg = _io.fingerprint({"threads": threads, "min_total": a.min_total, "max_reps": a.max_reps, "seed": a.seed,
                           "legacy": bool(a.legacy_dir), "chunk": a.chunk, "chunk_above": a.chunk_above,
                           "gen": GEN_VERSION, "version": 4}, length=8)  # 4: 2-hop kernel with the TLW certificate
    rows_dir = os.path.join(out_dir, f"rows_{family}_{a.min_n}_{a.max_n}_{cfg}")
    os.makedirs(rows_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"benchmark_{family}_{a.min_n}_{a.max_n}.csv")
    print(f"[benchmark] {family}: rows in {rows_dir}; {threads} threads; memory budget {mem_gb:.0f} GB; "
          f"time budget {a.budget_hours:g} h per phase and size", flush=True)
    for n in sorted(set(np.geomspace(a.min_n, a.max_n, a.graphs, dtype=int).tolist())):
        p_row = os.path.join(rows_dir, f"n{n}.json")
        if _io.read_json(p_row):
            continue
        row = run_size(family, n, a.seed, a.jobs, threads, rows_dir, read_rows(rows_dir), a, mem_gb)
        if row is None:
            continue
        _io.atomic_json(row, p_row)
        try:
            os.remove(os.path.join(rows_dir, f"n{n}.partial.json"))
        except OSError:
            pass
        write_csv(read_rows(rows_dir), csv_path)
    write_csv(read_rows(rows_dir), csv_path)
    if not a.no_figures:
        plot_family(csv_path, os.path.join(out_dir, f"runtime_scaling_loglog_{family.lower()}.pdf"), family,
                    predicted=not a.no_predicted)
    return csv_path


# ----------------------------------------------------------------- plots
def _f(r, k):
    try:
        v = float(r.get(k))
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _draw(ax, rows, family, predicted=True, legend=True):
    rows = sorted(rows, key=lambda r: int(r["n"]))
    ns = np.array([int(r["n"]) for r in rows], float)
    shown_pred = False
    for ph in ["exact", "sharpened_interval", "matching_interval", "bf_interval", "bf", "prior"]:
        mk, ls = PHASE_STYLE[ph]
        y = np.array([_f(r, f"{ph}_s") for r in rows])
        if not np.isfinite(y).any():
            continue
        line, = ax.plot(ns, y, marker=mk, markersize=3, linewidth=1.2, linestyle=ls, label=PHASE_LABELS[ph])
        if predicted:
            yp = np.array([_f(r, f"{ph}_pred_s") for r in rows])
            if np.isfinite(yp).any():
                last = np.flatnonzero(np.isfinite(y))[-1]
                xs, ys = np.r_[ns[last], ns[np.isfinite(yp)]], np.r_[y[last], yp[np.isfinite(yp)]]
                ax.plot(xs, ys, linestyle=":", color=line.get_color(), alpha=0.6, linewidth=1)
                ax.plot(ns[np.isfinite(yp)], yp[np.isfinite(yp)], linestyle="none", marker=mk, markersize=3,
                        markerfacecolor="none", color=line.get_color(), alpha=0.6)
                shown_pred = True
    ch = [int(r["n"]) for r in rows if str(r.get("chunked")) in ("True", "true", "1")]
    if ch:
        ax.axvline(min(ch), color="gray", linestyle=":", linewidth=0.8)
        ax.text(min(ch), 0.02, " chunked", transform=ax.get_xaxis_transform(), fontsize=7, color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of nodes $n$"); ax.set_ylabel("Execution time (s)")
    ax.set_title(FAMILY_TITLES.get(family, family), fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    if legend:
        h, l = ax.get_legend_handles_labels()
        if shown_pred:
            from matplotlib.lines import Line2D
            h.append(Line2D([], [], color="gray", linestyle=":", marker="o", markerfacecolor="none", alpha=0.6))
            l.append("predicted, not run")
        ax.legend(h, l, frameon=False, fontsize=7, loc="upper left", handlelength=2.2)


def _save(fig, out_pdf):
    for path, kw in ((out_pdf, {}), (out_pdf[:-4] + ".png", {"dpi": 300})):
        tmp = path[:-4] + ".tmp" + path[-4:]
        fig.savefig(tmp, bbox_inches="tight", **kw)
        os.replace(tmp, path)


def plot_family(csv_path, out_pdf, family, predicted=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    # drawn at the size it is printed (half the text width), so fonts stay legible
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    _draw(ax, rows, family, predicted)
    fig.tight_layout()
    _save(fig, out_pdf)
    plt.close(fig)


def plot_all(out_dir, predicted=True):
    """One figure per family (runtime_scaling_loglog_<family>.pdf) and a 2x2 panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    found = {}
    for fam in FAMILIES:
        cands = sorted(glob.glob(os.path.join(out_dir, f"benchmark_{fam}_*.csv")),
                       key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
        if cands:
            found[fam] = cands[-1]
            plot_family(cands[-1], os.path.join(out_dir, f"runtime_scaling_loglog_{fam.lower()}.pdf"), fam, predicted)
    if not found:
        return
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
    k = len(found)
    cols = 2 if k > 1 else 1
    fig, axes = plt.subplots((k + cols - 1) // cols, cols, figsize=(4.6 * cols, 3.4 * ((k + cols - 1) // cols)), squeeze=False)
    for ax, (fam, path) in zip(axes.ravel(), found.items()):
        _draw(ax, list(csv.DictReader(open(path))), fam, predicted, legend=(ax is axes.ravel()[0]))
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "runtime_scaling_all.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="ER", choices=FAMILIES)
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--max-n", type=int, default=15000)
    ap.add_argument("--graphs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--min-total", type=float, default=1.0, help="whole-graph timing: repeat until this many seconds")
    ap.add_argument("--max-reps", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=250000, help="edges per timed chunk on large graphs")
    ap.add_argument("--chunk-above", type=int, default=2000000, help="use chunked timing above this many edges")
    ap.add_argument("--budget-hours", type=float, default=12.0, help="skip a phase whose predicted time exceeds this")
    ap.add_argument("--mem-gb", type=float, default=None, help="memory available (default: Slurm allocation or RAM)")
    ap.add_argument("--no-mem-guard", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=2, help="give up on a chunk started this many times")
    ap.add_argument("--legacy-dir", default=None, help="path to the preliminary repository (for the old solver)")
    ap.add_argument("--legacy-sample", type=int, default=200)
    ap.add_argument("--no-predicted", action="store_true", help="do not draw predicted times of skipped phases")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="only redraw the figures from the CSVs in --out")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "benchmark"))
    a = ap.parse_args()
    if not a.plot_only:
        run_family(a.family, a)
    if not a.no_figures:
        plot_all(os.path.abspath(a.out), predicted=not a.no_predicted)


if __name__ == "__main__":
    main()
