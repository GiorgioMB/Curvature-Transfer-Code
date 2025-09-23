"""
Experiment runner for graph curvature distributions and envelope coverage

This script orchestrates the generation of random and canonical graphs,
computes edge curvatures and transfer bounds, and saves detailed outputs
(CSV, JSON, plots) for downstream analysis or paper figures.

Features
--------
- Supports a wide range of graph models (Erdos–Renyi, Watts–Strogatz, Barabasi–Albert,
  random geometric, hyperbolic, cycle, grid, tree, complete, and real networks).
- Computes both Balanced Forman and lazy Ollivier–Ricci curvatures, plus tight
  per-edge transfer bounds and envelopes.
- Saves per-edge tables, summary statistics, and histograms for each run.
- Can generate all paper figures in one go (see --auto-figures).

Usage (command line)
--------------------
$ python run_experiments.py --preset tiny
$ python run_experiments.py --er 200 0.03 --ws 200 6 0.1 --jobs 4
$ python run_experiments.py --hrg 1000 7.0 1.0 0.0 --jobs -1 --skip-plots

See argument help for all options.
"""
import os
import argparse
from typing import Dict, Tuple, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import models
from make_paper_figures import generate_paper_figures
from util_curvature import compute_curvatures, write_edge_table, summarize_run


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist (no error if already present)."""
    os.makedirs(path, exist_ok=True)


def _plot_hist(arr: np.ndarray, title: str, path_png: str, bins: int = 50):
    """Save a histogram of arr to a PNG file with a title."""
    plt.figure()
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


def add_preset_args(parser: argparse.ArgumentParser):
    """Add --preset argument for quick experiment suites."""
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["tiny", "small", "paper"],
        help="Run a predefined suite (tiny, small, or paper)"
    )
    

def add_family_args(parser: argparse.ArgumentParser):
    """Add arguments for all supported graph families and experiment controls."""
    # Random graphs
    parser.add_argument("--er", nargs=2, type=float, action="append", metavar=("n","p"),
                        help="Erdos-Renyi G(n,p)")
    parser.add_argument("--ws", nargs=3, type=float, action="append", metavar=("n","k","beta"),
                        help="Watts-Strogatz small-world (n, k, beta) with integer k")
    parser.add_argument("--ba", nargs=2, type=float, action="append", metavar=("n","m"),
                        help="Barabasi-Albert (n, m) with integer m")
    parser.add_argument("--rg", nargs=2, type=float, action="append", metavar=("n","r"),
                        help="Random geometric (n, radius)")
    parser.add_argument(
        "--hrg", nargs=4, type=float, action="append",
        metavar=("n","R","alpha","T"),
        help="Hyperbolic random graph (native model): n, disk radius R, curvature -alpha^2, temperature T"
    )
    # Parallelism controls
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of parallel jobs for both HRG generation and curvature computation. "
                             "Follows scikit-learn conventions: None (auto), 1 (sequential), -1 (all CPUs), "
                             ">1 (exact number), <-1 (all but |jobs|-1 CPUs)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Tile size for HRG; smaller -> more tasks. If omitted, chosen adaptively.")

    # Canonical graphs
    parser.add_argument("--cycle", nargs=1, type=int, action="append", metavar=("n",), help="Cycle C_n")
    parser.add_argument("--grid", nargs=2, type=int, action="append", metavar=("m","n"), help="Grid m x n")
    parser.add_argument("--tree", nargs=2, type=int, action="append", metavar=("d","h"), help="d-ary tree of height h")
    parser.add_argument("--complete", nargs=1, type=int, action="append", metavar=("n",), help="Complete graph K_n")
    # Real networks (optional)
    parser.add_argument("--include-real", action="store_true", help="Include real networks present in experiments/data/*.csv")
    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None, help="Name subfolder under out/")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--auto-figures",
        action="store_true",
        help="Generate paper figures after the runs (on by default for --preset paper)"
    )


def handle_presets(args, seed: int):
    """Fill in default arguments for --preset suites if not already set."""
    if args.preset is None:
        return
    if args.preset == "tiny":
        args.er = args.er or [[150, 0.02]]
        args.ws = args.ws or [[150, 6, 0.1]]
        args.ba = args.ba or [[150, 2]]
        args.rg = args.rg or [[150, 0.12]]
        args.cycle = args.cycle or [[100]]
        args.grid = args.grid or [[10, 12]]
        args.tree = args.tree or [[3, 5]]
        args.complete = args.complete or [[40]]
    elif args.preset == "small":
        args.er = args.er or [[400, 0.02], [400, 0.04]]
        args.ws = args.ws or [[400, 6, 0.1], [400, 8, 0.2]]
        args.ba = args.ba or [[400, 2], [400, 3]]
        args.rg = args.rg or [[400, 0.09]]
        args.cycle = args.cycle or [[200]]
        args.grid = args.grid or [[20, 20]]
        args.tree = args.tree or [[3, 6]]
        args.complete = args.complete or [[60]]
    elif args.preset == "paper":
        # (a) Random models
        args.hrg = args.hrg or [[300, 5.0, 1.0, 0.0], [350, 5.0, 1.0, 0.5]]
        args.er = args.er or [[800, 0.015], [800, 0.03]]
        args.ws = args.ws or [[800, 6, 0.05], [800, 10, 0.2]]
        args.ba = args.ba or [[800, 2], [800, 3], [800, 5]]
        args.rg = args.rg or [[800, 0.08], [800, 0.10]]
        # (b) Canonical families
        args.cycle = args.cycle or [[240]]
        args.grid = args.grid or [[28, 28]]
        args.tree = args.tree or [[3, 7], [4, 6]]
        args.complete = args.complete or [[70]]
        args.include_real = True
        args.skip_plots = True  # generate paper figures instead


def load_real_graphs(data_dir: str) -> List[Tuple[str, int, List[Tuple[int,int]]]]:
    """Load real network edge lists from CSVs in the given directory.

    Each file should have two columns (u, v) per row, with optional # comments.
    Returns a list of (name, n, edges) where n is the number of nodes.
    """
    out = []
    import csv
    for name in ["karate", "jazz", "power_grid", "yeast"]:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            continue
        edges = set()
        with open(path, "r") as f:
            r = csv.reader(f)
            for row in r:
                if not row or row[0].startswith("#"):
                    continue
                u = int(row[0]); v = int(row[1])
                if u == v: 
                    continue
                if u > v: 
                    u,v = v,u
                edges.add((u,v))
        n = 1 + max([max(u,v) for (u,v) in edges]) if edges else 0
        out.append((name, n, sorted(edges)))
    return out


def main():
    """Main experiment loop: parse args, generate graphs, compute curvatures, save outputs."""
    parser = argparse.ArgumentParser(description="Run curvature distribution experiments.")
    add_preset_args(parser)
    add_family_args(parser)
    args = parser.parse_args()
    seed = int(args.seed)

    handle_presets(args, seed)
    if getattr(args, "preset", None) == "paper" and not getattr(args, "auto_figures", False):
        args.auto_figures = True
    runs = []

    # Random families
    if args.er:
        for n, p in args.er:
            runs.append(("er_n{}_p{}".format(int(n), float(p)), *models.erdos_renyi(int(n), float(p), seed=seed)))
    if args.ws:
        for n, k, beta in args.ws:
            runs.append(("ws_n{}_k{}_b{}".format(int(n), int(k), float(beta)), *models.watts_strogatz(int(n), int(k), float(beta), seed=seed)))
    if args.ba:
        for n, m in args.ba:
            runs.append(("ba_n{}_m{}".format(int(n), int(m)), *models.barabasi_albert(int(n), int(m), seed=seed)))
    if args.rg:
        for n, r in args.rg:
            runs.append(("rg_n{}_r{}".format(int(n), float(r)), *models.random_geometric(int(n), float(r), seed=seed)))
    if hasattr(args, "hrg") and args.hrg:
        for n, R, alpha, T in args.hrg:
            runs.append((
                "hrg_n{}_R{}_a{}_T{}".format(int(n), float(R), float(alpha), float(T)),
                *models.make_hyperbolic_random_graph(
                    int(n), float(R),
                    alpha=float(alpha), T=float(T), seed=seed,
                    n_jobs=args.jobs, block_size=args.block_size
                )
            ))
    # Canonical
    if args.cycle:
        for (n,) in args.cycle:
            runs.append(("cycle_n{}".format(int(n)), *models.cycle_graph(int(n))))
    if args.grid:
        for m, n in args.grid:
            runs.append(("grid_{}x{}".format(int(m), int(n)), *models.grid_graph(int(m), int(n))))
    if args.tree:
        for d, h in args.tree:
            runs.append(("tree_d{}_h{}".format(int(d), int(h)), *models.dary_tree(int(d), int(h))))
    if args.complete:
        for (n,) in args.complete:
            runs.append(("complete_n{}".format(int(n)), *models.complete_graph(int(n))))

    # Real graphs
    if args.include_real:
        for name, n, edges in load_real_graphs(os.path.join(os.path.dirname(__file__), "data")):
            runs.append((f"real_{name}", n, edges))

    # Output dir
    run_name = args.run_name or ("preset_" + args.preset if args.preset else "custom")
    out_dir = os.path.join(os.path.dirname(__file__), "out", run_name)
    ensure_dir(out_dir)

    manifest = {"runs": [], "notes": "Distributional histograms and envelope/transfer coverage"}

    for tag, n, edges in runs:
        print(f"[run] {tag}: n={n}, m={len(edges)}")
        curv = compute_curvatures(n, edges, n_jobs=args.jobs)
        # Save edge-level table
        base_name = tag.replace(".", "_")
        if not args.skip_csv:
            write_edge_table(os.path.join(out_dir, f"{base_name}_edges.csv"), curv)
        # Summaries
        summary = summarize_run(curv)
        manifest["runs"].append({"tag": tag, "n": n, "m": len(edges), "summary": summary})

        # Plots
        if not args.skip_plots:
            _plot_hist(curv.base["c_OR"], f"{tag} — c_OR", os.path.join(out_dir, f"{base_name}__hist_cOR.png"), bins=args.bins)
            _plot_hist(curv.base["c_BF"], f"{tag} — c_BF", os.path.join(out_dir, f"{base_name}__hist_cBF.png"), bins=args.bins)
            _plot_hist(curv.theta_at_t - curv.base["c_OR"], f"{tag} — slack Theta(tri) - c_OR", os.path.join(out_dir, f"{base_name}__hist_slack_theta.png"), bins=args.bins)
            _plot_hist(curv.env_upper - curv.base["c_OR"], f"{tag} — slack envelope - c_OR", os.path.join(out_dir, f"{base_name}__hist_slack_env.png"), bins=args.bins)
            
    # Write manifest
    import json
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Optionally build the paper figures once all CSVs are in place
    if getattr(args, "auto_figures", False):
        try:
            base_out = os.path.dirname(out_dir)
            run_name = os.path.basename(out_dir)
            print(f"[run_experiments] Generating paper figures for run '{run_name}'")
            generate_paper_figures(out_root=base_out, run_name=run_name, bins=args.bins)
        except Exception as e:
            print(f"[run_experiments] Paper figure generation failed: {e}")
    print(f"[done] Wrote outputs to {out_dir}")
    


if __name__ == "__main__":
    main()
