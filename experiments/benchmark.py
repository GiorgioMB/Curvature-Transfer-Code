#!/usr/bin/env python3
"""Runtime scaling of exact OR curvature versus the certified intervals (Section 6.4, Table 8).

Families: four random graph models whose average degree grows as dbar(n) = 0.2 (50/n)^0.6 n,
about 2.1 n^0.4 (the law of the ER example of the paper): ER G(n, p) with p = dbar / (n - 1); BA
with m = round(dbar / 2); RGG in the unit square with r = sqrt(dbar / (pi (n - 1))); WS with k = 2
round(dbar / 2) and rewiring 0.2. The generators are vectorized and seeded per (seed, family, n),
so a restart regenerates the same graph (they are not those of the bound tables, generators.py).

Phases (``timing.PHASES``, shared with the screening table): exact, bf, prior, bf_interval,
matching_interval, sharpened_interval, cheap phases first. Up to ``--chunk-above`` edges a phase is
timed by the median of repeated runs; above, by the sum of the times of chunks of ``--chunk``
edges, each run once and saved as soon as it is measured (``rows_<...>/n<n>.partial.json``), so a
killed job resumes at the next chunk. A phase is not run, from that size on, if its time
extrapolated from the last sizes exceeds ``--budget-hours``, if the per-edge bitsets of exact OR
(about threads x 4 |S|_max^2 bytes) exceed 80% of ``--mem-gb``, or if the same chunk was started
``--max-attempts`` times; its reason is in ``<phase>_status`` and its extrapolated time, drawn as
"predicted, not run", in ``<phase>_pred_s``.

Usage
  python benchmark.py --family BA --min-n 50 --max-n 200000 --graphs 100 --jobs 32 --out ../results/benchmark
  python benchmark.py --plot-only --out ../results/benchmark      # redraw every family's figure
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
PHASE_ORDER = ["bf", "prior", "bf_interval", "matching_interval", "sharpened_interval", "exact"]
PHASE_STYLE = {"exact": ("o", "-"), "sharpened_interval": ("s", "--"), "matching_interval": ("P", "--"),
               "bf_interval": ("D", "-."), "bf": ("^", ":"), "prior": ("v", "-.")}
MEM_FACTOR = {"exact": 4.0}  # bytes per |S|^2 per thread (the intervals need no |S|^2 structure)
FAMILY_CODE = {"ER": 1, "BA": 2, "RGG": 3, "WS": 4}
GEN_VERSION = 1
COLS = (["family", "n", "param", "num_edges", "avg_deg", "max_deg", "s_max", "threads", "chunked", "chunk",
         "chunk_above", "budget_hours", "mem_gb", "gen_s"]
        + [f"{p}_{x}" for p in PHASE_ORDER for x in ("s", "pred_s", "status")])


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
    return row


def run_family(family, a):
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    timing.warmup()
    import numba
    threads = numba.get_num_threads() if a.jobs is None else (numba.config.NUMBA_NUM_THREADS if a.jobs == -1 else a.jobs)
    mem_gb = mem_budget_gb(a.mem_gb)
    cfg = _io.fingerprint({"threads": threads, "min_total": a.min_total, "max_reps": a.max_reps, "seed": a.seed,
                           "legacy": False, "chunk": a.chunk, "chunk_above": a.chunk_above,  # (as the rows of v1.1)
                           "gen": GEN_VERSION, "version": 4}, length=8)  # 4: 2-hop kernel and TLW of v10
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


# Figure 4 of the paper: one panel per family, drawn at its printed size (the manuscript includes each
# panel at width=.49\textwidth of JMLR's 6-in text width) with the axes in the same rectangle in every
# panel, so that fonts print at the sizes below and the four panels line up. Same fonts and palette as
# Figures 2-3 (experiments/make_paper_figures.py). Only the ER panel, the first one in the figure,
# carries a legend; the caption describes the vertical line where chunked timing starts.
PANEL_W, PANEL_H = 2.94, 2.13
AXES_RECT = (0.17, 0.175, 0.80, 0.715)
LEGEND_FAMILY = "ER"
# Phases drawn, in legend order. The matching (2-hop) interval is in the CSV but not drawn: its times
# coincide with those of the sharpened interval at the scale of the figure, and the caption does not list it.
FIGURE_PHASES = ["exact", "sharpened_interval", "bf_interval", "bf", "prior"]
PHASE_COLORS = {"exact": "#000000", "sharpened_interval": "#0072B2", "matching_interval": "#56B4E9",
                "bf_interval": "#D55E00", "bf": "#CC79A7", "prior": "#009E73"}
_PLOT_RC = {
    # Computer Modern as bundled with matplotlib (cmr10) has no accented letters or en dash (Erdős–Rényi):
    # matplotlib takes those glyphs, and only those, from the next family in the list
    "font.family": ["cmr10", "STIXGeneral"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "font.size": 7.5, "axes.titlesize": 8.0, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6, "axes.edgecolor": "#333333",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 3.5, "axes.labelpad": 2.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "legend.frameon": False, "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5, "legend.labelspacing": 0.25, "legend.borderaxespad": 0.3,
    "savefig.dpi": 300, "pdf.fonttype": 42,
}
FIGURE_LABELS = {"exact": "Exact OR (min-cut)", "sharpened_interval": "Sharpened interval",
                 "matching_interval": "2-hop interval", "bf_interval": "BF interval", "bf": "BF",
                 "prior": "JL / TLW lower bounds"}
FIGURE_TITLES = {"ER": "Erdős–Rényi", "BA": "Barabási–Albert", "RGG": "Random geometric",
                 "WS": r"Watts$\endash$Strogatz ($p=0.2$)"}  # a title with math is set by mathtext, which has no fallback


def _draw(ax, rows, family, predicted=True, legend=True):
    from matplotlib import ticker
    from matplotlib.lines import Line2D
    rows = sorted(rows, key=lambda r: int(r["n"]))
    ns = np.array([int(r["n"]) for r in rows], float)
    every = max(1, int(round(len(ns) / 10)))  # about ten markers per series, enough to tell them apart
    shown_pred = False
    for ph in FIGURE_PHASES:
        mk, ls = PHASE_STYLE[ph]
        y = np.array([_f(r, f"{ph}_s") for r in rows])
        if not np.isfinite(y).any():
            continue
        lw = 1.3 if ph == "exact" else 1.0
        line, = ax.plot(ns, y, marker=mk, markersize=2.6, markevery=every, linewidth=lw, linestyle=ls,
                        color=PHASE_COLORS[ph], label=FIGURE_LABELS[ph], zorder=3 if ph == "exact" else 2)
        if predicted:
            yp = np.array([_f(r, f"{ph}_pred_s") for r in rows])
            if np.isfinite(yp).any():
                last = np.flatnonzero(np.isfinite(y))[-1]
                xs, ys = np.r_[ns[last], ns[np.isfinite(yp)]], np.r_[y[last], yp[np.isfinite(yp)]]
                ax.plot(xs, ys, linestyle=":", color=line.get_color(), alpha=0.6, linewidth=1)
                ax.plot(ns[np.isfinite(yp)], yp[np.isfinite(yp)], linestyle="none", marker=mk, markersize=2.6,
                        markevery=every, markerfacecolor="none", color=line.get_color(), alpha=0.6)
                shown_pred = True
    ch = [int(r["n"]) for r in rows if str(r.get("chunked")) in ("True", "true", "1")]
    if ch:
        ax.axvline(min(ch), color="#9a9a9a", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("number of nodes $n$"); ax.set_ylabel("execution time (s)")
    ax.set_title(FIGURE_TITLES.get(family, family))
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        h, l = ax.get_legend_handles_labels()
        if shown_pred:
            h.append(Line2D([], [], color="gray", linestyle=":", marker="o", markerfacecolor="none", alpha=0.6))
            l.append("predicted, not run")
        ax.legend(h, l, loc="upper left")


def _save(fig, out_pdf, tight=True):
    for path, kw in ((out_pdf, {}), (out_pdf[:-4] + ".png", {"dpi": 300})):
        tmp = path[:-4] + ".tmp" + path[-4:]
        fig.savefig(tmp, bbox_inches="tight" if tight else None, **kw)
        os.replace(tmp, path)


def plot_family(csv_path, out_pdf, family, predicted=True, legend=None):
    """Panel of Figure 4 for one family; ``legend`` defaults to the ER panel only."""
    import matplotlib
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)  # timestamp notices when embedding cmr10
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    plt.rcParams.update(_PLOT_RC)
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AXES_RECT)
    _draw(ax, rows, family, predicted, legend=(family == LEGEND_FAMILY) if legend is None else legend)
    _save(fig, out_pdf, tight=False)  # fixed axes rectangle, so that the panels line up in the manuscript
    plt.close(fig)


def plot_all(out_dir, predicted=True):
    """One figure per family: runtime_scaling_loglog_<family>.pdf."""
    for fam in FAMILIES:
        cands = sorted(glob.glob(os.path.join(out_dir, f"benchmark_{fam}_*.csv")),
                       key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
        if cands:
            plot_family(cands[-1], os.path.join(out_dir, f"runtime_scaling_loglog_{fam.lower()}.pdf"), fam, predicted)


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
