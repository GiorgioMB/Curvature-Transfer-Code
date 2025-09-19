
import argparse
import json
import os
import glob
import shutil
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# Style
# ==============================

_PAPER_RC = {
    "figure.dpi": 220,
    "savefig.dpi": 320,
    "font.size": 12.0,
    "axes.titlesize": 14,
    "axes.labelsize": 12.5,
    "legend.fontsize": 11.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": ":",
    "grid.alpha": 0.28,
}
# Neutral palette (no seaborn)
COL_OBS   = "#4C78A8"
COL_LOWER = "#54A24B"
COL_UPPER = "#E45756"
COL_ENV   = "#F58518"
COL_THETA = "#B279A2"
COL_QRIB  = "#333333"  # almost black

def _apply_style():
    for k, v in _PAPER_RC.items():
        matplotlib.rcParams[k] = v

# ==============================
# Notation and display names (LaTeX-friendly via mathtext)
# ==============================

DISPLAY = {
    "c_OR":    r"$\mathfrak{c}_{\mathrm{OR}}$",
    "c_BF":    r"$\mathfrak{c}_{\mathrm{BF}}$",
    "c_OR0":   r"$\mathfrak{c}_{\mathrm{OR\text{-}0}}$",
    "Theta_at_t": r"$\Theta_\alpha(\triangle)$",
    "env_upper":  r"transport envelope (upper)",
}

# CSV columns expected from run_experiments / util_curvature
EXPECTED_COLS = [
    "u","v","deg_i","deg_j","triangle","Xi","sho_max",
    "c_BF","c_OR","c_OR0","Theta_const","Theta_slope","Theta_at_t","env_upper",
    "cOR_lower_from_cBF","cOR_upper_from_cBF","cBF_lower_from_cOR","cBF_upper_from_cOR",
]

# ==============================
# I/O helpers
# ==============================

def _load_manifest(run_root: str) -> Dict:
    man_path = os.path.join(run_root, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"No manifest.json in {run_root}")
    with open(man_path, "r") as f:
        return json.load(f)

def _edge_csv_for_tag(run_root: str, tag: str) -> Optional[str]:
    """run_experiments writes {base_name}_edges.csv in the run folder."""
    base = tag.replace('.', '_')
    cand = os.path.join(run_root, f"{base}_edges.csv")
    if os.path.exists(cand):
        return cand
    # fallback: pick the first *edges.csv that starts with base
    gl = glob.glob(os.path.join(run_root, f"{base}*edges.csv"))
    return gl[0] if gl else None

def _read_edges(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    # Be robust to lower-cased headers
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

# ==============================
# Pretty naming for tags (for titles and file names)
# ==============================

def _pretty_from_tag(tag: str) -> str:
    import re
    t = tag
    # ER: er_n{n}_p{p}
    m = re.fullmatch(r"er_n(\d+)_p([0-9.]+)", t)
    if m:
        n, p = m.groups()
        return rf"Erdős-Rényi $G(n,p)$ ($n={int(n)}$, $p={p}$)"
    # WS: ws_n{n}_k{k}_b{beta}
    m = re.fullmatch(r"ws_n(\d+)_k(\d+)_b([0-9.]+)", t)
    if m:
        n, k, b = m.groups()
        return rf"Watts-Strogatz ($n={int(n)}$, $k={int(k)}$, $\beta={b}$)"
    # BA: ba_n{n}_m{m}
    m = re.fullmatch(r"ba_n(\d+)_m(\d+)", t)
    if m:
        n, m2 = m.groups()
        return rf"Barabási-Albert ($n={int(n)}$, $m={int(m2)}$)"
    # RG: rg_n{n}_r{r}
    m = re.fullmatch(r"rg_n(\d+)_r([0-9.]+)", t)
    if m:
        n, r = m.groups()
        return rf"Random Geometric ($n={int(n)}$, $r={r}$)"
    # HRG: hrg_n{n}_R{R}_a{alpha}_T{T}
    m = re.fullmatch(r"hrg_n(\d+)_R([0-9.]+)_a([0-9.]+)_T([0-9.]+)", t)
    if m:
        n, R, a, T = m.groups()
        return rf"Hyperbolic Random Graph ($n={int(n)}$, $R={R}$, $\alpha={a}$, $T={T}$)"
    # Canonical families
    m = re.fullmatch(r"cycle_n(\d+)", t)
    if m:
        (n,) = m.groups()
        return rf"Cycle $C_{{{int(n)}}}$"
    m = re.fullmatch(r"grid_(\d+)x(\d+)", t)
    if m:
        m1, n1 = m.groups()
        return rf"Grid$_{{{int(m1)}\times {int(n1)}}}$"
    # Handle tags that might include _edges suffix
    m = re.fullmatch(r"grid_(\d+)x(\d+)_edges", t)
    if m:
        m1, n1 = m.groups()
        return rf"Grid$_{{{int(m1)}\times {int(n1)}}}$"
    m = re.fullmatch(r"tree_d(\d+)_h(\d+)", t)
    if m:
        d, h = m.groups()
        return rf"${int(d)}$-ary Tree ($d={int(d)}$, $h={int(h)}$)"
    m = re.fullmatch(r"complete_n(\d+)", t)
    if m:
        (n,) = m.groups()
        return rf"Complete Graph $K_{{{int(n)}}}$"
    # Real networks
    m = re.fullmatch(r"real_(.+)", t)
    if m:
        name = m.group(1)
        pretty_map = {
            "karate": "Zachary Karate Club",
            "jazz": "Jazz Musicians",
            "power_grid": "Western US Power Grid",
            "yeast": "Yeast Transcription Network",
        }
        return f"Real network: {pretty_map.get(name, name.replace('_',' '))}"
    # Fallback: return the raw tag
    return t

def _slugify(s: str) -> str:
    import re, unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\-]+", "_", s).strip("_")
    s = re.sub(r"__+", "_", s)
    return s

# ==============================
# Plot primitives
# ==============================

def _hist_with_bands(ax, obs: np.ndarray,
                     lower: Optional[np.ndarray] = None,
                     upper: Optional[np.ndarray] = None,
                     env: Optional[np.ndarray] = None,
                     theta: Optional[np.ndarray] = None,
                     lower_label: Optional[str] = None,
                     upper_label: Optional[str] = None,
                     bins: int = 60,
                     title: str = "",
                     xlabel: str = ""):
    # Observed distribution
    ax.hist(obs, bins=bins, density=True, alpha=0.55, color=COL_OBS, label="observed")
    # Lower/upper transfer distributions (edgewise)
    if lower is not None and len(lower) > 0:
        ax.hist(lower, bins=bins, density=True, histtype="step", linewidth=1.6, color=COL_LOWER,
                label=(lower_label or "lower transfer (dist.)"))
    if upper is not None and len(upper) > 0:
        ax.hist(upper, bins=bins, density=True, histtype="step", linewidth=1.6, linestyle="--", color=COL_UPPER,
                label=(upper_label or "upper transfer (dist.)"))
    # Envelope / Theta distributions
    if env is not None and len(env) > 0:
        ax.hist(env, bins=bins, density=True, histtype="step", linewidth=1.3, linestyle="-.", color=COL_ENV,
                label="transport envelope (dist.)")
    if theta is not None and len(theta) > 0:
        ax.hist(theta, bins=bins, density=True, histtype="step", linewidth=1.1, linestyle=":", color=COL_THETA,
                label=DISPLAY["Theta_at_t"] + " dist.")
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
        # No data: return empty arrays consistently
        return (np.array([]),) * 4

    lo, hi = float(np.min(x)), float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # Degenerate: single bin
        bins = np.array([lo, lo + 1e-6])
    else:
        bins = np.linspace(lo, hi, nbins + 1)

    nb = len(bins) - 1                 # <-- ACTUAL number of bins
    idx = np.clip(np.digitize(x, bins) - 1, 0, nb - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    qL = np.full(nb, np.nan)
    qM = np.full(nb, np.nan)
    qH = np.full(nb, np.nan)
    for b in range(nb):
        vals = y[idx == b]
        if len(vals) > 0:
            qL[b], qM[b], qH[b] = np.quantile(vals, qs)

    return centers, qL, qM, qH


def _scatter_ribbon(ax,
                    bf: np.ndarray, orv: np.ndarray,
                    lower_from_bf: Optional[np.ndarray] = None,
                    nbins: int = 28):
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
        ax.plot(xc, qM, color=COL_QRIB, linewidth=1.7, label=r"median $\mathfrak{c}_{\mathrm{OR}}\mid \mathfrak{c}_{\mathrm{BF}}$")
        ax.plot(xc, qL, color=COL_QRIB, linewidth=1.1, linestyle="--")
        ax.plot(xc, qH, color=COL_QRIB, linewidth=1.1, linestyle="--")
    # Predicted BF->OR lower (median per bin)
    if low_s is not None:
        _, _, pM, _ = _binned_quantiles(bf, lower_from_bf, nbins=nbins, qs=(0.5, 0.5, 0.5))
        ax.plot(xc, pM, color=COL_LOWER, linewidth=1.8, label=r"lower transfer (BF$\rightarrow$OR): median")
    ax.set_title(r"$\mathfrak{c}_{\mathrm{BF}}$ vs $\mathfrak{c}_{\mathrm{OR}}$ (edgewise)")
    ax.set_xlabel(DISPLAY["c_BF"])
    ax.set_ylabel(DISPLAY["c_OR"])
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

# ==============================
# Figure orchestration per run
# ==============================

def _make_run_figures(run_root: str, tag: str, pretty: Optional[str], bins: int = 60, keep_existing: bool = False):
    csv_path = _edge_csv_for_tag(run_root, tag)
    if csv_path is None or not os.path.exists(csv_path):
        print(f"[make_paper_figures] No edge CSV for '{tag}' under {run_root}; skipping.")
        return
    df = _read_edges(csv_path)

    # Retrieve arrays (robust to missing cols)
    as_np = lambda name: pd.to_numeric(df[name], errors="coerce").to_numpy() if name in df.columns else None

    cOR     = as_np("c_OR")
    cBF     = as_np("c_BF")
    cOR_lo  = as_np("cOR_lower_from_cBF")
    cOR_up  = as_np("cOR_upper_from_cBF")
    cBF_lo  = as_np("cBF_lower_from_cOR")
    cBF_up  = as_np("cBF_upper_from_cOR")
    env_up  = as_np("env_upper")
    theta_t = as_np("Theta_at_t")

    fig_dir = os.path.join(run_root, "figures_paper")
    safe_tag = tag.replace("/", "-").replace(" ", "_").replace(".", "_")
    pretty_tag = pretty or _pretty_from_tag(tag)
    pretty_slug = _slugify(pretty_tag)

    # 1) OR histogram with transfer overlays
    if cOR is not None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        title = rf"{pretty_tag} — {DISPLAY['c_OR']} distribution"
        _hist_with_bands(ax, obs=cOR, lower=cOR_lo, upper=cOR_up, env=env_up, theta=theta_t,
                         lower_label=r"BF$\rightarrow$OR lower (dist.)", upper_label=r"BF$\rightarrow$OR upper (dist.)",
                         bins=bins, title=title, xlabel=DISPLAY["c_OR"])
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_cOR_hist__{safe_tag}.png"))
        plt.close(fig)

    # 2) BF histogram with OR->BF overlays
    if cBF is not None:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        title = rf"{pretty_tag} — {DISPLAY['c_BF']} distribution"
        _hist_with_bands(ax, obs=cBF, lower=cBF_lo, upper=cBF_up, env=None, theta=None,
                         lower_label=r"OR$\rightarrow$BF lower (dist.)", upper_label=r"OR$\rightarrow$BF upper (dist.)",
                         bins=bins, title=title, xlabel=DISPLAY["c_BF"])
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"fig_cBF_hist__{safe_tag}.png"))
        plt.close(fig)

    # 3) Scatter BF vs OR with ribbons
    if (cBF is not None) and (cOR is not None):
        fig, ax = plt.subplots(figsize=(6.6, 4.9))
        _scatter_ribbon(ax, cBF, cOR, lower_from_bf=cOR_lo, nbins=28)
        suptitle = rf"{pretty_tag}"
        fig.suptitle(suptitle, y=0.99, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(fig_dir, f"fig_scatter_cBF_cOR__{safe_tag}.png"))
        plt.close(fig)

def generate_paper_figures(out_root: Optional[str] = None, run_name: Optional[str] = None, bins: int = 60, keep_existing: bool = False):
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
        rd = os.path.join(base_root, run_name)
        _generate_for_run_dir(rd, bins=bins, keep_existing=keep_existing)

def _generate_for_run_dir(run_dir: str, bins: int = 60, keep_existing: bool = False):
    man = _load_manifest(run_dir)
    runs = man.get("runs", [])
    if not runs:
        print(f"[make_paper_figures] Empty manifest in {run_dir}")
        return
    print(f"[make_paper_figures] Generating figures for {len(runs)} run(s) in {os.path.basename(run_dir)}")
    
    # Handle directory creation/deletion once per run directory
    fig_dir = os.path.join(run_dir, "figures_paper")
    if os.path.exists(fig_dir):
        if not keep_existing:
            print(f"[make_paper_figures] Removing existing figures directory: {fig_dir}")
            shutil.rmtree(fig_dir)
            os.makedirs(fig_dir, exist_ok=True)
        else:
            print(f"[make_paper_figures] Keeping existing figures directory: {fig_dir}")
    else:
        print(f"[make_paper_figures] Creating new figures directory: {fig_dir}")
        os.makedirs(fig_dir, exist_ok=True)
    
    for r in runs:
        tag = r.get("tag") or r.get("name") or "run"
        pretty = r.get("title") or r.get("pretty")  # optional pretty name in manifest
        _make_run_figures(run_dir, tag, pretty, bins=bins, keep_existing=keep_existing)

def main():
    _apply_style()
    ap = argparse.ArgumentParser(description="Generate paper-ready figures from experiment outputs.")
    ap.add_argument("--out-root", type=str, default=None, help="Root folder containing run subfolders (default: experiments/out)")
    ap.add_argument("--run-name", type=str, default=None, help="Specific run folder name to process (default: all under out-root)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--keep-existing", action="store_true", help="Keep existing figures directory instead of recreating it")
    args = ap.parse_args()
    generate_paper_figures(out_root=args.out_root, run_name=args.run_name, bins=args.bins, keep_existing=args.keep_existing)

if __name__ == "__main__":
    main()
