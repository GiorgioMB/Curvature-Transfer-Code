#!/usr/bin/env python3
"""Edge map of certify-then-solve (Section 6.3, Figure ``fig:screening-map``).

Draws one registry graph four times on the same vertex positions, one panel per
certificate of the cascade of Algorithm 1 for the threshold query
{kappa_OR < tau} (default tau = 0, the sign query of Table 7):

  (a) BF interval        what kappa_BF and the two degrees decide (Corollary 5.8);
                         reported in Table 7 but not used by the algorithm
  (b) 2-hop interval     the first stage (Proposition 5.9), which contains (a)
  (c) + TLW certificate  the second stage: the 2-hop interval intersected with
                         U^prior, i.e. the sharpened interval of Section 6.1
  (d) + exact solver     the transport problems that remain (orcut)

An edge leaves the cascade at the first certificate that decides it, so the
decided sets are nested. Each panel draws the edges that its own certificate
decides bold, in the colors of the other figures (Okabe-Ito): vermillion, the
color of upper ends, for kappa_OR < tau, which an upper end (or the solver)
decides, and blue, the color of lower ends, for kappa_OR >= tau. Edges decided
by an earlier certificate are thin and gray, and undecided edges are not drawn,
so the drawn part of the graph grows from panel to panel. Below each panel: the
share of edges decided without transport, and in (d) the number of transport
problems.

Consistency. The decisions come from the per-edge store of ``edges.py``, the
arrays behind Tables 3-6 and the screening fractions of ``screening.py``, with
the intervals of ``curvature_transfer.intervals`` widened by ROUNDING_TOL, and
the stage counts and the answer are checked against
``curvature_transfer.screen``, which runs the cascade from scratch (the step
fails otherwise).

Positions. A random geometric graph (registry tag ``rg_n<n>_r<r>``) is drawn at
the points from which it was generated: they are redrawn with the seeded RNG of
``generators.random_geometric``, and the geometric graph they span, restricted
to its largest component as in ``suite.load``, must reproduce the stored edges
and vertex labels exactly. Any other graph gets a deterministic layout
(Kamada-Kawai up to 1500 vertices, otherwise a seeded spring layout), so that
``--graph Yeast`` draws the network of the workshop poster.

Outputs, in ``--out``:
  fig_screening_map__<file_tag>.pdf   the figure (vector; JMLR text width)
  fig_screening_map__<file_tag>.png   preview
  screening_map__<file_tag>.json      per-stage counts, sign split, medians
  screening_map_numbers.tex           the \\res macros quoted in Section 6.3

Usage
  python main.py --steps screening_map figures            # pipeline step -> results/screening_map/
  python experiments/screening_map.py --results results --out results/screening_map
  python experiments/screening_map.py --results results --out results/screening_map --graph Yeast
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # font subsetting chatter on PDF save
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from curvature_transfer import Graph, certify_threshold, screen  # noqa: E402
from curvature_transfer import _io  # noqa: E402
from curvature_transfer import intervals as I  # noqa: E402
import edges as E  # noqa: E402
import suite  # noqa: E402
from make_paper_figures import C_LO, C_UP, _apply_style  # noqa: E402

DEFAULT_GRAPH = "RGG(800, 0.056419)"
STAGES = ("BF interval", "2-hop interval", "+ TLW certificate", "+ exact solver")
N_STAGES = len(STAGES)

# ----------------------------------------------------------------- style
FIG_W = 6.0                  # inches: JMLR's \textwidth, so 8-pt labels print at 8 pt
SIDE, GAP = 0.04, 0.14       # outer margin and gap between panels (inches)
MAP = (FIG_W - 2 * SIDE - (N_STAGES - 1) * GAP) / N_STAGES   # side of each square map
TITLE_H, NUM_H, LEG_H = 0.20, 0.40, 0.22                      # bands above / below the maps
FIG_H = LEG_H + NUM_H + MAP + TITLE_H

C_NEG, C_POS = C_UP, C_LO    # decided here: kappa < tau (by an upper end), kappa >= tau (by a lower end)
C_OLD = "#b4b4b4"            # decided by an earlier certificate
C_TEXT2 = "#4d4d4d"          # secondary text
C_TEXT_OLD = "#8a8a8a"       # legend text of the gray edges
LW_NEW, LW_OLD = 0.85, 0.35  # line widths (pt): decided here, decided earlier (undecided: not drawn)


# ----------------------------------------------------------- the cascade
def stages_from_store(A, tau: float = 0.0) -> np.ndarray:
    """Stage (0..3, as in STAGES) at which certify-then-solve decides each edge.

    Uses the same intervals as ``screening.benchmark_rows``: the BF interval,
    the 2-hop interval of stage one, and the sharpened interval (2-hop
    intersected with U^prior), each widened by ROUNDING_TOL."""
    iv = I.from_arrays(A, A["prior_upper"])
    nested = (iv["bf"], I.two_hop_from_arrays(A), iv["sharpened"])
    decided = []
    for lo, up in nested:
        lo, up = I.widen(lo, up)
        _, _, und = certify_threshold(lo, up, tau)
        decided.append(~und)
    for k in range(len(decided) - 1):  # Proposition 5.9 and the intersection with U^prior
        if np.any(decided[k] & ~decided[k + 1]):
            raise RuntimeError(f"an edge decided by '{STAGES[k]}' is undecided by '{STAGES[k + 1]}'")
    stage = np.full(len(A["c_OR"]), N_STAGES - 1, dtype=np.int64)
    for k in reversed(range(len(decided))):
        stage[decided[k]] = k
    return stage


def check_against_cascade(A, stage, tau, n_jobs, seed):
    """The stage counts and the answer must be those of ``curvature_transfer.screen``."""
    g = Graph.from_edges(A["edges"], int(A["num_nodes"]))
    r = screen(g, tau=tau, n_jobs=n_jobs, seed=seed)
    want = {"n_bf": int((stage == 0).sum()), "n_two_hop": int((stage <= 1).sum()),
            "n_tlw": int((stage == 2).sum()), "n_solved": int((stage == 3).sum())}
    got = {k: int(r[k]) for k in want}
    if got != want:
        raise RuntimeError(f"stage counts {want} differ from curvature_transfer.screen {got}")
    if not np.array_equal(r["answer"], np.asarray(A["c_OR"]) < tau):
        raise RuntimeError("certify-then-solve answer differs from the exact curvatures")


# ------------------------------------------------------------- positions
def rgg_positions(spec, A, seed: int) -> np.ndarray:
    """Generating coordinates of a random geometric graph of the registry,
    checked against the stored edge set and vertex labels."""
    m = re.fullmatch(r"rg_n(\d+)_r([0-9.]+)", spec.tag)
    n0, r = int(m[1]), float(m[2])
    rnd = random.Random(seed)  # the draws of generators.random_geometric
    pts = np.array([(rnd.random(), rnd.random()) for _ in range(n0)])
    iu, ju = np.triu_indices(n0, 1)
    dx, dy = pts[iu, 0] - pts[ju, 0], pts[iu, 1] - pts[ju, 1]
    e = np.stack([iu, ju], axis=1)[dx * dx + dy * dy < r * r]
    # largest component, labeled 0..n'-1 in the order of the original labels (suite.largest_component)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    adj = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n0, n0))
    _, lab = connected_components(adj, directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    remap = -np.ones(n0, np.int64)
    remap[keep] = np.arange(len(keep))
    e = remap[e]
    e = e[(e >= 0).all(axis=1)]
    e = e[np.lexsort((e[:, 1], e[:, 0]))]
    if len(keep) != int(A["num_nodes"]) or not np.array_equal(e, np.asarray(A["edges"])):
        raise RuntimeError(f"{spec.name}: regenerated coordinates do not reproduce the stored graph")
    return pts[keep]


def layout_positions(A, seed: int) -> np.ndarray:
    import networkx as nx

    n = int(A["num_nodes"])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, np.asarray(A["edges"]).tolist()))
    pos = nx.kamada_kawai_layout(G) if n <= 1500 else nx.spring_layout(G, seed=seed, iterations=200)
    return np.array([pos[v] for v in range(n)])


def positions(spec, A, seed: int, layout_seed: int) -> np.ndarray:
    if re.fullmatch(r"rg_n\d+_r[0-9.]+", spec.tag):
        return rgg_positions(spec, A, seed)
    return layout_positions(A, layout_seed)


# --------------------------------------------------------------- summary
def summarize(spec, A, stage, tau):
    c = np.asarray(A["c_OR"], float)
    M = len(c)
    neg = c < tau
    out = {"graph": spec.name, "tag": spec.tag, "tau": tau, "n": int(A["num_nodes"]), "E": M, "stages": []}
    for k, name in enumerate(STAGES):
        at = stage == k
        out["stages"].append({
            "stage": name, "decided_here": int(at.sum()), "decided_cumulative": int((stage <= k).sum()),
            "share_cumulative": float((stage <= k).mean()),
            "below_tau": int((at & neg).sum()), "at_or_above_tau": int((at & ~neg).sum()),
            "median_abs_gap_to_tau": float(np.median(np.abs(c[at] - tau))) if at.any() else float("nan"),
        })
    without = stage < N_STAGES - 1
    out["median_abs_gap_to_tau_without_transport"] = float(np.median(np.abs(c[without] - tau))) if without.any() else float("nan")
    out["work_ratio"] = M / max(1, int((stage == N_STAGES - 1).sum()))
    return out


def _thousands(x: int) -> str:
    return f"{int(x):,}"


def write_macros(s, path):
    """The numbers quoted in Section 6.3, as \\res macros (same namespace as numbers.tex)."""
    st = s["stages"]
    M = s["E"]
    vals = {
        "MapEdges": _thousands(M),
        "MapBF": f"{100 * st[0]['share_cumulative']:.1f}",
        "MapBFPos": f"{100 * st[0]['at_or_above_tau'] / max(1, st[0]['decided_here']):.1f}",
        "MapTwoHop": f"{100 * st[1]['share_cumulative']:.1f}",
        "MapTwoHopNeg": f"{100 * st[1]['below_tau'] / max(1, st[1]['decided_here']):.1f}",
        "MapTLWGain": f"{100 * st[2]['decided_here'] / M:.1f}",
        "MapSharp": f"{100 * st[2]['share_cumulative']:.1f}",
        "MapSolved": _thousands(st[3]["decided_here"]),
        "MapSolvedPct": f"{100 * st[3]['decided_here'] / M:.1f}",
        "MapMedGapSolved": f"{st[3]['median_abs_gap_to_tau']:.2f}",
        "MapMedGapDecided": f"{s['median_abs_gap_to_tau_without_transport']:.2f}",
        "MapWorkRatio": f"{s['work_ratio']:.1f}",
    }
    lines = [f"% written by experiments/screening_map.py: {s['graph']}, tau = {s['tau']:g}"]
    lines += [f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v}}}" for k, v in vals.items()]
    _io.atomic_text("\n".join(lines) + "\n", path)
    return vals


# ------------------------------------------------------------------ figure
def draw(P, edges, stage, below, tau, paths):
    _apply_style()
    segs = P[np.asarray(edges)]
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    pad = 0.012 * float((hi - lo).max())
    tau_txt = "0" if tau == 0 else f"{tau:g}"
    M = len(stage)
    for k in range(N_STAGES):
        x0 = SIDE + k * (MAP + GAP)
        ax = fig.add_axes((x0 / FIG_W, (LEG_H + NUM_H) / FIG_H, MAP / FIG_W, MAP / FIG_H))
        here, before = stage == k, stage < k   # edges decided later are not drawn
        new_neg, new_pos = here & below, here & ~below
        if new_neg.sum() < new_pos.sum():      # the less frequent sign on top, so it stays visible
            new_layers = [(new_pos, C_POS, LW_NEW), (new_neg, C_NEG, LW_NEW)]
        else:
            new_layers = [(new_neg, C_NEG, LW_NEW), (new_pos, C_POS, LW_NEW)]
        layers = [(before, C_OLD, LW_OLD)] + new_layers   # bottom to top
        for mask, colors, lw in layers:
            if mask.any():
                ax.add_collection(LineCollection(segs[mask], colors=colors, linewidths=lw,
                                                 capstyle="round", zorder=1))
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="box", anchor="C")
        ax.axis("off")
        ax.set_title(f"({'abcd'[k]}) {STAGES[k]}", pad=3.0)
        xc = (x0 + MAP / 2) / FIG_W
        y1 = (LEG_H + NUM_H - 0.04) / FIG_H
        y2 = (LEG_H + NUM_H - 0.215) / FIG_H
        if k < N_STAGES - 1:
            big = f"{100 * (stage <= k).mean():.1f}%"
            small = "of the edges decided"
        else:
            big = _thousands(int(here.sum()))
            small = f"transport problems, of {_thousands(M)}"
        fig.text(xc, y1, big, ha="center", va="top", fontsize=11.5)
        fig.text(xc, y2, small, ha="center", va="top", fontsize=7.0, color=C_TEXT2)

    lab_neg = r"$\kappa_{\mathrm{OR}}<" + tau_txt + "$"
    lab_pos = r"$\kappa_{\mathrm{OR}}\geq" + tau_txt + "$"
    h = [Line2D([], [], color=C_NEG, lw=2.0), Line2D([], [], color=C_POS, lw=2.0),
         Line2D([], [], color=C_OLD, lw=1.0), Line2D([], [], color="none")]
    labels = [f"decided here, {lab_neg}", f"decided here, {lab_pos}", "decided by an earlier stage",
              "undecided: not drawn"]
    leg = fig.legend(h, labels, loc="center", bbox_to_anchor=(0.5, (LEG_H / 2) / FIG_H), ncol=4,
                     handlelength=1.8, columnspacing=1.4, fontsize=7.0)
    for t in leg.get_texts()[2:]:
        t.set_color(C_TEXT_OLD)
    for p in paths:
        fig.savefig(p, dpi=300 if p.endswith(".png") else None)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results directory (per-edge store in <results>/edges)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--graph", default=DEFAULT_GRAPH, help="registry name or tag (experiments/suite.py)")
    ap.add_argument("--tau", type=float, default=0.0, help="threshold of the query {kappa_OR < tau}")
    ap.add_argument("--seed", type=int, default=0, help="graph seed (as in the tables)")
    ap.add_argument("--layout-seed", type=int, default=1, help="seed of the spring layout (large non-RGG graphs)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--quick", action="store_true",
                    help="accepted with the common flags of main.py; the graph is small, so it changes nothing")
    a = ap.parse_args()

    spec = suite.specs([a.graph])[0]
    A = E.compute(spec, a.results, a.seed, a.jobs, "paper", a.chunk)
    stage = stages_from_store(A, a.tau)
    check_against_cascade(A, stage, a.tau, a.jobs, a.seed)
    P = positions(spec, A, a.seed, a.layout_seed)

    os.makedirs(a.out, exist_ok=True)
    s = summarize(spec, A, stage, a.tau)
    stem = f"fig_screening_map__{spec.file_tag}"
    draw(P, A["edges"], stage, np.asarray(A["c_OR"]) < a.tau, a.tau,
         [os.path.join(a.out, f"{stem}.pdf"), os.path.join(a.out, f"{stem}.png")])
    _io.atomic_text(json.dumps(s, indent=2) + "\n", os.path.join(a.out, f"screening_map__{spec.file_tag}.json"))
    vals = write_macros(s, os.path.join(a.out, "screening_map_numbers.tex"))
    for row in s["stages"]:
        print(f"[screening_map] {spec.name}: {row['stage']:18s} decided here {row['decided_here']:6,d} "
              f"(<tau {row['below_tau']:,d}, >=tau {row['at_or_above_tau']:,d}); "
              f"cumulative {100 * row['share_cumulative']:5.1f}%", flush=True)
    print("[screening_map] macros: " + ", ".join(f"{k}={v}" for k, v in vals.items()), flush=True)


if __name__ == "__main__":
    main()
