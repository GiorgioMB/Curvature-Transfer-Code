import argparse
import json
import os
import glob
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------
# Style
# ------------------------------

_PAPER_RC = {
    "figure.dpi": 220,
    "savefig.dpi": 300,
    "font.size": 11.5,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": ":",
    "grid.alpha": 0.28,
}

# A small neutral palette (no seaborn dependency)
COL_OBS = "#4C78A8"        # blue
COL_LOWER = "#54A24B"      # green
COL_UPPER = "#E45756"      # red
COL_ENV = "#F58518"        # orange
COL_THETA = "#B279A2"      # purple
COL_QRIB = "#333333"       # almost black


def _apply_style():
    for k, v in _PAPER_RC.items():
        matplotlib.rcParams[k] = v



EXPECTED_COLS = [
    "u","v","deg_i","deg_j","triangle","Xi","sho_max",
    "c_BF","c_OR","c_OR0","Theta_const","Theta_slope","Theta_at_t","env_upper",
    "cOR_lower_from_cBF","cOR_upper_from_cBF","cBF_lower_from_cOR","cBF_upper_from_cOR"
]

def _load_manifest(run_root: str) -> Dict:
    man_path = os.path.join(run_root, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"No manifest.json in {run_root}")
    with open(man_path, "r") as f:
        return json.load(f)

def _edge_csv_for_tag(run_root: str, tag: str) -> Optional[str]:
    # run_experiments writes {base_name}_edges.csv at the same folder as manifest
    base = tag.replace('.', '_')
    cand = os.path.join(run_root, f"{base}_edges.csv")
    if os.path.exists(cand):
        return cand
    # fallback: pick the first *edges.csv that starts with base
    gl = glob.glob(os.path.join(run_root, f"{base}*edges.csv"))
    if gl:
        return gl[0]
    return None

def _read_edges(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        lower_map = {c.lower(): c for c in df.columns}
        new_cols = {}
        for c in missing:
            if c.lower() in lower_map:
                new_cols[lower_map[c.lower()]] = c
        if new_cols:
            df = df.rename(columns=new_cols)
    return df


# ------------------------------
# Plot primitives
# ------------------------------

def _hist_with_bands(ax, obs: np.ndarray,
                     lower: Optional[np.ndarray] = None,
                     upper: Optional[np.ndarray] = None,
                     env: Optional[np.ndarray] = None,
                     theta: Optional[np.ndarray] = None,
                     bins: int = 60,
                     title: str = "",
                     xlabel: str = ""):
    # Observed
    ax.hist(obs, bins=bins, density=True, alpha=0.55, color=COL_OBS, label="observed")
    # Lower/upper bands as step hists (distributions of bounds across edges)
    if lower is not None and len(lower) > 0:
        ax.hist(lower, bins=bins, density=True, histtype="step", linewidth=1.6, color=COL_LOWER, label="pred. lower (dist.)")
    if upper is not None and len(upper) > 0:
        ax.hist(upper, bins=bins, density=True, histtype="step", linewidth=1.6, linestyle="--", color=COL_UPPER, label="pred. upper (dist.)")
    # Envelope distributions
    if env is not None and len(env) > 0:
        ax.hist(env, bins=bins, density=True, histtype="step", linewidth=1.3, linestyle="-.", color=COL_ENV, label="lazy envelope (dist.)")
    if theta is not None and len(theta) > 0:
        ax.hist(theta, bins=bins, density=True, histtype="step", linewidth=1.1, linestyle=":", color=COL_THETA, label=r"$\Theta_\alpha(t)$ (dist.)")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2, handlelength=2.2)


def _binned_quantiles(x: np.ndarray, y: np.ndarray, nbins: int = 28,
                      qs=(0.1, 0.5, 0.9)) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), *(np.array([]) for _ in range(3))
    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        bins = np.array([lo, lo + 1e-6])
    else:
        bins = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    qL = np.full(nbins, np.nan)
    qM = np.full(nbins, np.nan)
    qH = np.full(nbins, np.nan)
    for b in range(nbins):
        vals = y[idx == b]
        if len(vals) > 0:
            qL[b], qM[b], qH[b] = np.quantile(vals, qs)
    return centers, qL, qM, qH


def _scatter_ribbon(ax, bf: np.ndarray, orv: np.ndarray, lower_from_bf: Optional[np.ndarray] = None, nbins: int = 28):
    # Subsample if large for visual clarity
    n = len(bf)
    if n > 25000:
        rng = np.random.default_rng(1234)
        sel = rng.choice(n, size=25000, replace=False)
        bf_s, or_s = bf[sel], orv[sel]
        low_s = lower_from_bf[sel] if lower_from_bf is not None else None
    else:
        bf_s, or_s, low_s = bf, orv, lower_from_bf

    ax.scatter(bf_s, or_s, s=6, alpha=0.28, color=COL_OBS, linewidth=0, label="edges (sample)")
    # Observed quantile ribbon of OR|BF
    xc, qL, qM, qH = _binned_quantiles(bf, orv, nbins=nbins)
    if len(xc) > 0:
        ax.plot(xc, qM, color=COL_QRIB, linewidth=1.7, label="obs. median OR|BF")
        ax.plot(xc, qL, color=COL_QRIB, linewidth=1.1, linestyle="--")
        ax.plot(xc, qH, color=COL_QRIB, linewidth=1.1, linestyle="--")
    # Predicted BF->OR lower (median per bin)
    if low_s is not None:
        _, pL, pM, pH = _binned_quantiles(bf, lower_from_bf, nbins=nbins, qs=(0.5, 0.5, 0.5))
        ax.plot(xc, pM, color=COL_LOWER, linewidth=1.8, label="pred. BF→OR lower (median)")
    ax.set_title("BF vs OR (edgewise)")
    ax.set_xlabel("Balanced Forman curvature")
    ax.set_ylabel("Ollivier–Ricci curvature")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)


# ------------------------------
# Figure orchestration per run
# ------------------------------

def _make_run_figures(run_root: str, tag: str, bins: int = 60):
    csv_path = _edge_csv_for_tag(run_root, tag)
    if csv_path is None or not os.path.exists(csv_path):
        print(f"[make_paper_figures] No edge CSV for '{tag}' under {run_root}; skipping.")
        return
    df = _read_edges(csv_path)

    # Retrieve arrays (robust to missing cols)
    as_np = lambda name: pd.to_numeric(df[name], errors="coerce").to_numpy() if name in df.columns else None

    cOR = as_np("c_OR")
    cBF = as_np("c_BF")
    cOR_lo = as_np("cOR_lower_from_cBF")
    cOR_up = as_np("cOR_upper_from_cBF")
    cBF_lo = as_np("cBF_lower_from_cOR")
    cBF_up = as_np("cBF_upper_from_cOR")
    env_up = as_np("env_upper")
    theta_t = as_np("Theta_at_t")

    fig_dir = os.path.join(run_root, "figures_paper")
    os.makedirs(fig_dir, exist_ok=True)
    safe_tag = tag.replace("/", "-").replace(" ", "_").replace(".", "_")

    # 1) OR histogram with transfer overlays
    if cOR is not None:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        _hist_with_bands(ax, obs=cOR, lower=cOR_lo, upper=cOR_up, env=env_up, theta=theta_t,
                         bins=bins, title=f"{tag} — OR distribution with analytic overlays", xlabel="c_OR")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{safe_tag}__OR_hist_paper.png"))
        plt.close(fig)

    # 2) BF histogram with OR->BF overlays
    if cBF is not None:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        _hist_with_bands(ax, obs=cBF, lower=cBF_lo, upper=cBF_up, env=None, theta=None,
                         bins=bins, title=f"{tag} — BF distribution with analytic overlays", xlabel="c_BF")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{safe_tag}__BF_hist_paper.png"))
        plt.close(fig)

    # 3) Scatter BF vs OR with ribbons
    if (cBF is not None) and (cOR is not None):
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        _scatter_ribbon(ax, cBF, cOR, lower_from_bf=cOR_lo, nbins=28)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{safe_tag}__scatter_BF_OR_paper.png"))
        plt.close(fig)


def generate_paper_figures(out_root: Optional[str] = None, run_name: Optional[str] = None, bins: int = 60):
    _apply_style()
    # Default output root matches run_experiments: experiments/out/<run_name>
    base_root = out_root or os.path.join(os.path.dirname(__file__), "out")
    if not os.path.isdir(base_root):
        print(f"[make_paper_figures] Output base not found: {base_root}")
        return
    if run_name is None:
        # Use all folders
        run_dirs = [d for d in glob.glob(os.path.join(base_root, "*")) if os.path.isdir(d)]
        if not run_dirs:
            print(f"[make_paper_figures] No run dirs in {base_root}")
            return
        for rd in run_dirs:
            _generate_for_run_dir(rd, bins=bins)
    else:
        rd = os.path.join(base_root, run_name)
        _generate_for_run_dir(rd, bins=bins)


def _generate_for_run_dir(run_dir: str, bins: int = 60):
    man = _load_manifest(run_dir)
    runs = man.get("runs", [])
    if not runs:
        print(f"[make_paper_figures] Empty manifest in {run_dir}")
        return
    print(f"[make_paper_figures] Generating figures for {len(runs)} run(s) in {os.path.basename(run_dir)}")
    for r in runs:
        tag = r.get("tag") or r.get("name") or "run"
        _make_run_figures(run_dir, tag, bins=bins)


def main():
    _apply_style()
    ap = argparse.ArgumentParser(description="Generate paper-ready figures from experiment outputs.")
    ap.add_argument("--out-root", type=str, default=None, help="Root folder containing run subfolders (default: experiments/out)")
    ap.add_argument("--run-name", type=str, default=None, help="Specific run folder name to process (default: all under out-root)")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()
    generate_paper_figures(out_root=args.out_root, run_name=args.run_name, bins=args.bins)


if __name__ == "__main__":
    main()
