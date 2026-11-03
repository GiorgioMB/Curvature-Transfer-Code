"""
Figures of the BF interval (Figures 2 and 3 of the paper) from the per-edge CSVs.

Reads the run directories written by ``experiments/distributions.py`` (through ``edges.py --csv``):
``manifest.json``, which lists the graphs under ``"runs"`` (each with its registry ``"name"`` and
``"tag"``), and one ``{tag}_edges.csv`` per graph. For every graph it writes, into
``<run>/figures_paper/``,

  fig_scatter_cBF_cOR__{tag}_2d_Hist.png   density of the edges in the (kappa_BF, kappa_OR) plane, with
                                           the conditional medians of kappa_OR and of the two ends of
                                           the BF interval (Figure 2)
  fig_cOR_hist__{tag}.png                  distributions of kappa_OR and of the two ends, on shared bins
                                           (Figure 3, left)
  fig_cOR_quantile__{tag}.png              quantile profiles of kappa_OR and of the two ends; the shaded
                                           band is the certified range of every quantile, and the bar
                                           at q = 0.1 the certified range of the lowest decile
                                           (Figure 3, right)
  fig_scatter_cBF_cOR__{tag}.png,          the same views as a point cloud, and for the reverse
  fig_cBF_hist__{tag}.png,                 direction (bounds on kappa_BF from kappa_OR, Appendix C);
  fig_cBF_quantile__{tag}.png              not used by the paper

under the file names that ``main.py`` (``FIGURES``) and the manuscript expect.

Style. Every panel is drawn at its printed size, PANEL_W x PANEL_H inches: the manuscript includes
each panel at ``width=0.49\\textwidth`` and JMLR's ``\\textwidth`` is 6 in, so 8-pt labels print at
8 pt. The axes occupy the same rectangle in every panel, so that panels placed side by side line up.
Text uses Computer Modern (matplotlib's mathtext ``cm``), the font of the paper. Colors are from
the Okabe-Ito palette (distinguishable with color-vision deficiencies and in grayscale): exact
curvature black, lower end blue, upper end vermillion, edge density in gray. Only the first panel
of each paper figure carries a legend (the captions describe the encodings); every other graph gets
a legend on every panel.

Conditional medians (Figure 2) are computed on bins of kappa_BF that hold about equally many edges
and never split edges with the same BF value, so every point of a median line summarizes at least
MIN_PER_BIN edges; histograms of different quantities share their bins.

Usage
  python experiments/make_paper_figures.py --out-root results/edges --run-name csv
  (``python main.py --steps distributions`` runs it as part of the pipeline.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ==============================
# Style
# ==============================

PANEL_W, PANEL_H = 2.94, 2.25           # inches; 0.49 x JMLR's 6-in text width
AXES_RECT = (0.175, 0.17, 0.79, 0.715)  # left, bottom, width, height (figure fractions), every panel

_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,   # cmr10 has no minus sign; mathtext draws it
    "font.size": 8.0,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "axes.titlepad": 3.5,
    "axes.labelpad": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "lines.linewidth": 1.1,
    "lines.solid_capstyle": "round",
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

C_OR = "#1a1a1a"      # exact kappa_OR
C_LO = "#0072B2"      # lower end of the BF interval
C_UP = "#D55E00"      # upper end of the BF interval
C_FILL = "#c9c9c9"    # histogram of kappa_OR
C_BAND = "#a6bddb"    # certified band of the quantile profiles
C_GRID = "#e6e6e6"
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "density", [plt.get_cmap("Greys")(x) for x in np.linspace(0.16, 0.78, 64)])

MIN_PER_BIN = 25      # edges behind every point of a conditional median
MEDIAN_BINS = 16      # target number of bins for the conditional medians
CORE = (0.001, 0.999)  # Figure 2 axes show the edges between these quantiles of each coordinate
DENSITY_BINS = 56     # cells per axis of the 2D histogram

# Panels of the paper's Figures 2 and 3, in the order of the manuscript: only the first panel of
# each figure carries a legend. File tags as written by edges.py (dots replaced by underscores).
FIG2_TAGS = ["er_n800_p0_0100125", "ws_n800_k10_b0_2", "ba_n1600_m2", "rg_n1600_r0_039894",
             "hrg_n800_R5_0_a1_0_T0_5", "sbm2_n1000_pin0_012018_pout0_004006"]
FIG3_TAGS = ["er_n800_p0_0100125", "hrg_n800_R5_0_a1_0_T0_0", "real_yeast"]

K_BF = r"$\kappa_{\mathrm{BF}}$"
K_OR = r"$\kappa_{\mathrm{OR}}$"


def _apply_style():
    """Apply the paper style (fonts, sizes, line widths) to all figures created afterwards."""
    matplotlib.rcParams.update(_PAPER_RC)


def _legend_on(tag: str, paper_tags: Sequence[str]) -> bool:
    return tag not in paper_tags or tag == paper_tags[0]


def _halo(lw: float):
    """White outline that separates a line from the density underneath."""
    return [pe.Stroke(linewidth=lw + 1.4, foreground="white", alpha=0.9), pe.Normal()]


def _new_panel():
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AXES_RECT)
    return fig, ax


def _save(fig, path: str, keep_existing: bool) -> None:
    if not keep_existing or not os.path.exists(path):
        fig.savefig(path)  # fixed axes rectangle: no tight bbox, so panels align when placed side by side
    plt.close(fig)


# ==============================
# CSV columns and I/O
# ==============================

# CSV columns written by edges.py (CurvatureEngine.edge_table) that this module uses.
EXPECTED_COLS = [
    "u", "v", "deg_i", "deg_j", "triangle", "Xi", "sho_max",
    "c_BF", "c_OR", "c_OR0", "env_upper",
    "cOR_lower_from_cBF", "cOR_upper_from_cBF", "cBF_lower_from_cOR", "cBF_upper_from_cOR",
]


def _load_manifest(run_root: str) -> Dict:
    """Parse ``<run_root>/manifest.json``; FileNotFoundError if it is missing."""
    man_path = os.path.join(run_root, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"No manifest.json in {run_root}")
    with open(man_path, "r") as f:
        return json.load(f)


def _edge_csv_for_tag(run_root: str, tag: str) -> Optional[str]:
    """Path of ``{tag}_edges.csv`` (dots in the tag replaced by underscores), or of the first
    ``{tag}*edges.csv``; None if there is none."""
    base = tag.replace(".", "_")
    cand = os.path.join(run_root, f"{base}_edges.csv")
    if os.path.exists(cand):
        return cand
    gl = glob.glob(os.path.join(run_root, f"{base}*edges.csv"))
    return gl[0] if gl else None


def _read_edges(path_csv: str) -> pd.DataFrame:
    """Read a per-edge CSV, mapping differently-cased headers to the expected names."""
    df = pd.read_csv(path_csv)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        lower_map = {c.lower(): c for c in df.columns}
        new_cols = {lower_map[c.lower()]: c for c in missing if c.lower() in lower_map}
        if new_cols:
            df = df.rename(columns=new_cols)
    return df


# ==============================
# Titles
# ==============================

def _title(name: Optional[str], tag: str) -> str:
    """Panel title: the registry name, as in the tables (``2x500`` printed as 2 x 500)."""
    t = name or _pretty_from_tag(tag)
    return re.sub(r"(\d)x(\d)", r"\1$\\times$\2", t)


def _pretty_from_tag(tag: str) -> str:
    """Fallback title for a run without a registry name in the manifest."""
    pats = [
        (r"er_n(\d+)_p([0-9._]+)", lambda m: f"ER({m[1]}, {m[2].replace('_', '.')})"),
        (r"ws_n(\d+)_k(\d+)_b([0-9._]+)", lambda m: f"WS({m[1]}, {m[2]}, {m[3].replace('_', '.')})"),
        (r"ba_n(\d+)_m(\d+)", lambda m: f"BA({m[1]}, {m[2]})"),
        (r"rg_n(\d+)_r([0-9._]+)", lambda m: f"RGG({m[1]}, {m[2].replace('_', '.')})"),
        (r"real_(.+)", lambda m: m[1].replace("_", " ").title()),
    ]
    for pat, f in pats:
        m = re.fullmatch(pat, tag)
        if m:
            return f(m)
    return tag


# ==============================
# Statistics
# ==============================

def _finite(*arrays):
    """The arrays restricted to the entries where all of them are finite."""
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return [a[m] for a in arrays]


def _cond_medians(x: np.ndarray, ys: Sequence[np.ndarray], nbins: int = MEDIAN_BINS,
                  min_count: int = MIN_PER_BIN) -> Tuple[np.ndarray, list]:
    """Medians of x and of every y on bins of x that hold about equally many points.

    Bin edges are quantiles of x; equal values of x always fall in the same bin, and bins with
    fewer than ``min_count`` points are merged into their left neighbor (or dropped if they
    are the only one), so that every point of the curves summarizes at least ``min_count``
    points."""
    data = _finite(x, *ys)
    x, ys = data[0], data[1:]
    if len(x) == 0:
        return np.array([]), [np.array([]) for _ in ys]
    nb = int(max(1, min(nbins, len(x) // max(1, min_count))))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, nb + 1)))
    if len(edges) < 2:
        groups = [np.arange(len(x))]
    else:
        idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
        groups = [np.flatnonzero(idx == b) for b in range(len(edges) - 1)]
        groups = [g for g in groups if len(g)]
        merged = []
        for g in groups:
            if merged and len(g) < min_count:
                merged[-1] = np.concatenate([merged[-1], g])
            else:
                merged.append(g)
        if len(merged) > 1 and len(merged[0]) < min_count:
            merged[1] = np.concatenate([merged[0], merged[1]])
            merged = merged[1:]
        groups = [g for g in merged if len(g) >= min(min_count, len(x))]
    cx = np.array([np.median(x[g]) for g in groups])
    cy = [np.array([np.median(y[g]) for g in groups]) for y in ys]
    return cx, cy


def _limits(*arrays, pad: float = 0.04) -> Tuple[float, float]:
    """Common axis limits of the finite values of the arrays, padded on both sides."""
    vals = [a[np.isfinite(a)] for a in arrays if a is not None and len(a)]
    vals = [v for v in vals if len(v)]
    if not vals:
        return -1.0, 1.0
    lo, hi = min(float(v.min()) for v in vals), max(float(v.max()) for v in vals)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _n_bins(bins, n: int):
    """An integer ``bins`` is a maximum: below it, the Rice rule 2 n^(1/3) (at least 12) sets the
    number of bins for n values, so that small graphs do not get noisy histograms. A numpy rule
    name is passed through."""
    if isinstance(bins, (int, np.integer)):
        return int(min(int(bins), max(12, round(2.0 * max(n, 1) ** (1.0 / 3.0)))))
    return bins


def _core_limits(a: np.ndarray, pad: float) -> Tuple[float, float]:
    """Limits covering the edges between the CORE quantiles of ``a`` (a few extreme edges, such as
    the edges at a degree-one vertex, would otherwise squeeze the bulk of the data)."""
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return -1.0, 1.0
    lo, hi = np.quantile(a, CORE)
    return _limits(np.array([lo, hi]), pad=pad)


def _shared_bins(lim: Tuple[float, float], ref: np.ndarray, bins) -> np.ndarray:
    """Bin edges covering ``lim``: ``bins`` equal bins if an integer, otherwise the width that
    numpy's rule ``bins`` gives for the reference sample, so that every series uses the same bins."""
    if isinstance(bins, (int, np.integer)):
        return np.linspace(lim[0], lim[1], int(bins) + 1)
    ref = ref[np.isfinite(ref)]
    w = np.diff(np.histogram_bin_edges(ref, bins=bins))[0] if len(ref) > 1 else 0.0
    if not np.isfinite(w) or w <= 0:
        return np.linspace(lim[0], lim[1], 61)
    n = min(max(int(np.ceil((lim[1] - lim[0]) / w)), 10), 400)
    return np.linspace(lim[0], lim[1], n + 1)


def _quantile_profile(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Empirical quantiles F^{-1}(q) = min{x : F(x) >= q} of the finite values of a."""
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.full_like(q, np.nan)
    return np.quantile(a, q, method="inverted_cdf")


# ==============================
# Panels
# ==============================

def _density_panel(ax, bf, orv, lo=None, up=None, title="", legend=True, points=False):
    """Edges in the (kappa_BF, kappa_OR) plane, with conditional medians of kappa_OR and of the
    two ends of the BF interval (``lo``, ``up``) given kappa_BF."""
    bf, orv = np.asarray(bf, float), np.asarray(orv, float)
    xb, yb = _finite(bf, orv)
    ys = [orv] + [a for a in (lo, up) if a is not None]
    cx, cy = _cond_medians(bf, ys)
    med_or = cy[0]
    med_lo = cy[1] if lo is not None else None
    med_up = cy[1 + (lo is not None)] if up is not None else None

    xlim = _core_limits(xb, pad=0.03)
    ylim = _limits(np.array(_core_limits(yb, pad=0.0)), *(m for m in (med_lo, med_up) if m is not None), pad=0.05)
    if points:
        n = len(xb)
        sel = np.random.default_rng(1234).choice(n, size=25000, replace=False) if n > 25000 else slice(None)
        ax.scatter(xb[sel], yb[sel], s=1.6, color="#7a7a7a", alpha=0.35, linewidths=0, rasterized=True)
    else:
        xe = np.linspace(*xlim, DENSITY_BINS + 1)
        ye = np.linspace(*ylim, DENSITY_BINS + 1)
        h, _, _ = np.histogram2d(xb, yb, bins=[xe, ye])
        hmax = max(1.0, float(h.max()))
        ax.hist2d(xb, yb, bins=[xe, ye], cmap=DENSITY_CMAP, cmin=1,
                  norm=mcolors.LogNorm(vmin=1, vmax=hmax), rasterized=True, zorder=1)

    if len(cx):
        if med_up is not None and med_lo is not None:
            ax.fill_between(cx, med_lo, med_up, color=C_BAND, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(cx, med_or, color=C_OR, lw=1.35, zorder=3, path_effects=_halo(1.35))
        # the ends are dashed and drawn on top, so that a median of kappa_OR that coincides with
        # one of them (e.g. the upper end on clustered graphs) stays visible between the dashes
        if med_lo is not None:
            ax.plot(cx, med_lo, color=C_LO, lw=1.2, ls=(0, (3.2, 1.8)), zorder=4)
        if med_up is not None:
            ax.plot(cx, med_up, color=C_UP, lw=1.2, ls=(0, (3.2, 1.8)), zorder=4)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(K_BF)
    ax.set_ylabel(K_OR)
    ax.set_title(title)
    ax.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        handles = [Patch(facecolor=DENSITY_CMAP(0.55), edgecolor="none", label="edges (log density)"),
                   Line2D([], [], color=C_OR, lw=1.35, label=r"median of " + K_OR)]
        if med_lo is not None:
            handles.append(Line2D([], [], color=C_LO, lw=1.2, ls=(0, (3.2, 1.8)), label="median lower end"))
        if med_up is not None:
            handles.append(Line2D([], [], color=C_UP, lw=1.2, ls=(0, (3.2, 1.8)), label="median upper end"))
        curves = [(cx, m) for m in (med_or, med_lo, med_up) if m is not None]
        _legend(ax, handles, xb, yb, xlim, ylim, curves)


def _legend(ax, handles, xb, yb, xlim, ylim, curves=()):
    """Legend in the corner where it covers the fewest edges and median-curve points (a curve point
    weighs as much as 2% of the edges), measured with the legend's actual extent."""
    pts = [(np.asarray(xb), np.asarray(yb), 1.0)] + [(np.asarray(x), np.asarray(y), 0.02 * len(xb)) for x, y in curves]
    fig = ax.figure
    best, best_cost = "upper left", None
    for loc in ("upper left", "lower right", "upper right", "lower left"):
        leg = ax.legend(handles=handles, loc=loc)
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
        cost = 0.0
        for x, y, w in pts:
            fx = (x - xlim[0]) / (xlim[1] - xlim[0])
            fy = (y - ylim[0]) / (ylim[1] - ylim[0])
            cost += w * float(np.sum((fx >= bb.x0) & (fx <= bb.x1) & (fy >= bb.y0) & (fy <= bb.y1)))
        leg.remove()
        if best_cost is None or cost < best_cost:
            best, best_cost = loc, cost
    leg = ax.legend(handles=handles, loc=best)
    leg.set_zorder(6)


def _hist_panel(ax, obs, lo, up, lim, bins, title="", legend=True,
                labels=(K_OR, "lower end", "upper end"), xlabel="curvature", ylabel="density"):
    """Distributions of the exact values and of the two ends, on shared bins. Descriptive only:
    the histograms of the ends do not bound that of the exact values pointwise."""
    obs = np.asarray(obs, float)
    edges = _shared_bins(lim, obs, _n_bins(bins, int(np.isfinite(obs).sum())))
    ax.hist(obs[np.isfinite(obs)], bins=edges, density=True, color=C_FILL, edgecolor="none",
            label=labels[0], zorder=2)
    for a, c, lab in ((lo, C_LO, labels[1]), (up, C_UP, labels[2])):
        if a is not None:
            a = np.asarray(a, float)
            ax.hist(a[np.isfinite(a)], bins=edges, density=True, histtype="step", color=c, lw=1.0,
                    label=lab, zorder=3)
    ax.set_xlim(*lim)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * (1.42 if legend else 1.04))   # headroom above the bars for the legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        h, lab = ax.get_legend_handles_labels()
        # step histograms are listed as patches; show them as lines
        h = [Patch(facecolor=C_FILL, edgecolor="none") if i == 0 else Line2D([], [], color=hh.get_edgecolor(), lw=1.0)
             for i, hh in enumerate(h)]
        ax.legend(h, lab, loc="upper left", ncol=3, columnspacing=0.9, handlelength=1.3)


def _quantile_panel(ax, obs, lo, up, lim, title="", legend=True, q_mark=0.1,
                    labels=(K_OR, "lower end", "upper end"), ylabel="curvature"):
    """Quantile profiles of the exact values and of the two ends. Since lo <= obs <= up on every
    edge, F_lo^{-1}(q) <= F_obs^{-1}(q) <= F_up^{-1}(q) for every q: the shaded band is a certified
    range for every quantile, and the bar at ``q_mark`` the one quoted in the text."""
    q = np.linspace(1.0 / 2000, 1.0, 2000)
    po = _quantile_profile(np.asarray(obs, float), q)
    pl = _quantile_profile(np.asarray(lo, float), q) if lo is not None else None
    pu = _quantile_profile(np.asarray(up, float), q) if up is not None else None
    if pl is not None and pu is not None:
        ax.fill_between(q, pl, pu, color=C_BAND, alpha=0.32, linewidth=0, zorder=1, label="certified band")
    if pl is not None:
        ax.plot(q, pl, color=C_LO, lw=0.95, zorder=2, label=labels[1])
    if pu is not None:
        ax.plot(q, pu, color=C_UP, lw=0.95, zorder=2, label=labels[2])
    ax.plot(q, po, color=C_OR, lw=1.35, zorder=3, label=labels[0])
    if q_mark and pl is not None and pu is not None:
        qm = np.array([q_mark])
        a, b = _quantile_profile(np.asarray(lo, float), qm)[0], _quantile_profile(np.asarray(up, float), qm)[0]
        ax.axvline(q_mark, color="#8c8c8c", lw=0.6, ls=":", zorder=1)
        ax.plot([q_mark, q_mark], [a, b], color="#333333", lw=2.6, alpha=0.55, solid_capstyle="butt", zorder=4,
                label=rf"band at $q={q_mark:g}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"quantile level $q$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        h, lab = ax.get_legend_handles_labels()
        order = [lab.index(x) for x in (labels[0], labels[1], labels[2]) if x in lab]
        order += [i for i in range(len(lab)) if i not in order]
        ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left")


# ==============================
# Figure orchestration per run
# ==============================

def _make_run_figures(run_root: str, tag: str, pretty: Optional[str], bins=60, keep_existing: bool = False) -> None:
    """Write the figures of one graph (see the module docstring) into ``run_root/figures_paper``."""
    csv_path = _edge_csv_for_tag(run_root, tag)
    if csv_path is None or not os.path.exists(csv_path):
        print(f"[make_paper_figures] No edge CSV for '{tag}' under {run_root}; skipping.")
        return
    df = _read_edges(csv_path)
    col = lambda name: pd.to_numeric(df[name], errors="coerce").to_numpy(float) if name in df.columns else None

    cOR, cBF = col("c_OR"), col("c_BF")
    cOR_lo, cOR_up = col("cOR_lower_from_cBF"), col("cOR_upper_from_cBF")
    cBF_lo, cBF_up = col("cBF_lower_from_cOR"), col("cBF_upper_from_cOR")

    fig_dir = os.path.join(run_root, "figures_paper")
    os.makedirs(fig_dir, exist_ok=True)
    safe_tag = tag.replace("/", "-").replace(" ", "_").replace(".", "_")
    title = _title(pretty, safe_tag)
    out = lambda stem: os.path.join(fig_dir, f"{stem}__{safe_tag}.png")

    if cOR is not None:
        lim = _limits(cOR, *(a for a in (cOR_lo, cOR_up) if a is not None))
        leg = _legend_on(safe_tag, FIG3_TAGS)
        fig, ax = _new_panel()
        _hist_panel(ax, cOR, cOR_lo, cOR_up, lim, bins, title=title, legend=leg)
        _save(fig, out("fig_cOR_hist"), keep_existing)
        fig, ax = _new_panel()
        _quantile_panel(ax, cOR, cOR_lo, cOR_up, lim, title=title, legend=leg)
        _save(fig, out("fig_cOR_quantile"), keep_existing)

    if cBF is not None:
        lim = _limits(cBF, *(a for a in (cBF_lo, cBF_up) if a is not None))
        labels = (K_BF, "lower bound from " + K_OR, "upper bound from " + K_OR)
        fig, ax = _new_panel()
        _hist_panel(ax, cBF, cBF_lo, cBF_up, lim, bins, title=title, labels=labels)
        _save(fig, out("fig_cBF_hist"), keep_existing)
        fig, ax = _new_panel()
        _quantile_panel(ax, cBF, cBF_lo, cBF_up, lim, title=title, labels=labels, q_mark=None)
        _save(fig, out("fig_cBF_quantile"), keep_existing)

    if cBF is not None and cOR is not None:
        leg = _legend_on(safe_tag, FIG2_TAGS)
        fig, ax = _new_panel()
        _density_panel(ax, cBF, cOR, cOR_lo, cOR_up, title=title, legend=leg, points=True)
        _save(fig, out("fig_scatter_cBF_cOR"), keep_existing)
        fig, ax = _new_panel()
        _density_panel(ax, cBF, cOR, cOR_lo, cOR_up, title=title, legend=leg)
        _save(fig, os.path.join(fig_dir, f"fig_scatter_cBF_cOR__{safe_tag}_2d_Hist.png"), keep_existing)


def generate_paper_figures(out_root: Optional[str] = None, run_name: Optional[str] = None, bins=60,
                           keep_existing: bool = False) -> None:
    """Generate the figures of every run directory under ``out_root``, or only of ``run_name``.

    ``bins``: maximum number of histogram bins (the Rice rule decides below it), or a numpy rule
    such as ``"fd"``. ``keep_existing``: keep
    an existing ``figures_paper/`` and existing files (default: recreate the directory)."""
    _apply_style()
    base_root = out_root or os.path.join(os.path.dirname(__file__), "out")
    if not os.path.isdir(base_root):
        print(f"[make_paper_figures] Output base not found: {base_root}")
        return
    if run_name is None:
        run_dirs = [d for d in glob.glob(os.path.join(base_root, "*")) if os.path.isdir(d)]
        if not run_dirs:
            print(f"[make_paper_figures] No run dirs in {base_root}")
            return
        for rd in run_dirs:
            _generate_for_run_dir(rd, bins=bins, keep_existing=keep_existing)
    else:
        _generate_for_run_dir(os.path.join(base_root, run_name), bins=bins, keep_existing=keep_existing)


def _generate_for_run_dir(run_dir: str, bins=60, keep_existing: bool = False) -> None:
    """Generate the figures of every run listed in ``run_dir/manifest.json``."""
    man = _load_manifest(run_dir)
    runs = man.get("runs", [])
    if not runs:
        print(f"[make_paper_figures] Empty manifest in {run_dir}")
        return
    print(f"[make_paper_figures] Generating figures for {len(runs)} run(s) in {os.path.basename(run_dir)}")
    fig_dir = os.path.join(run_dir, "figures_paper")
    if os.path.exists(fig_dir) and not keep_existing:
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    for r in runs:
        tag = r.get("tag") or r.get("name") or "run"
        pretty = r.get("title") or r.get("pretty") or r.get("name")
        _make_run_figures(run_dir, tag, pretty, bins=bins, keep_existing=keep_existing)


_ALLOWED = {"auto", "fd", "scott", "sturges", "doane", "rice", "sqrt", "stone"}


def parse_bins(s):
    try:
        return int(s)
    except ValueError:
        s = s.lower()
        if s in _ALLOWED:
            return s
        raise argparse.ArgumentTypeError(f"bins must be an integer or one of: {', '.join(sorted(_ALLOWED))}")


def main():
    _apply_style()
    ap = argparse.ArgumentParser(description="Figures of the BF interval from the per-edge CSVs.")
    ap.add_argument("--out-root", type=str, default=None, help="folder that contains the run folders (default: experiments/out)")
    ap.add_argument("--run-name", type=str, default=None, help="one run folder under --out-root (default: all)")
    ap.add_argument("--bins", type=parse_bins, default=60, metavar="INT|RULE",
                    help="histogram bins: a number, or a numpy rule (auto, fd, scott, sturges, doane, rice, sqrt, stone)")
    ap.add_argument("--keep-existing", action="store_true", help="keep the existing figures folder and files")
    args = ap.parse_args()
    generate_paper_figures(out_root=args.out_root, run_name=args.run_name, bins=args.bins,
                           keep_existing=args.keep_existing)


if __name__ == "__main__":
    main()
