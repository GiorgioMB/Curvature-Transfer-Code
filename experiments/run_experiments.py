"""
Experiment runner for graph curvature distributions and envelope coverage

This script orchestrates the generation of random and canonical graphs,
computes edge curvatures and transfer bounds, and saves detailed outputs (CSV, JSON, plots)
for downstream analysis or paper figures.

Features
--------
- Supports a wide range of graph models.
- Computes both Balanced Forman and lazy Ollivier--Ricci curvatures, as well as transfer bounds and envelopes.
- Saves per-edge tables, summary statistics, and histograms for each run.
- Orchestrates exact OT vs. combinatorial bounds runtime benchmarking.
- Orchestrates GNN rewiring experiments across BORF and SDRF variants.
- Can generate all paper figures in one go (see --auto-figures).

Usage (command line)
--------------------
$ python run_experiments.py --preset tiny --only-benchmark --auto-figures
$ python run_experiments.py --er 200 0.03 --ws 200 6 0.1 --jobs 4
$ python run_experiments.py --preset paper --only-gnn
"""
import os
import warnings
import argparse
from typing import Tuple, List
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import json
import time
from datetime import datetime

import models
from make_paper_figures import generate_paper_figures
from util_curvature import compute_curvatures, write_edge_table, summarize_run

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _plot_hist(arr: np.ndarray, title: str, path_png: str, bins: int = 50):
    plt.figure()
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def add_preset_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["tiny", "small", "medium", "paper"],
        help="Run a predefined suite (tiny, small, medium, or paper)"
    )

def add_family_args(parser: argparse.ArgumentParser):
    # Random graphs
    parser.add_argument("--er", nargs=2, type=float, action="append", metavar=("n","p"),
                        help="Erdos-Renyi G(n,p)")
    parser.add_argument("--ws", nargs=3, type=float, action="append", metavar=("n","k","beta"),
                        help="Watts-Strogatz small-world (n, k, beta) with integer k")
    parser.add_argument("--ba", nargs=2, type=float, action="append", metavar=("n","m"),
                        help="Barabasi-Albert (n, m) with integer m")
    parser.add_argument("--rg", nargs=2, type=float, action="append", metavar=("n","r"),
                        help="Random geometric (n, radius)")
    parser.add_argument("--rreg", nargs=2, type=int, action="append", metavar=("n","d"),
                        help="Random d-regular graph (n, d)")
    parser.add_argument("--sbm2", nargs=3, type=float, action="append", metavar=("n","p_in","p_out"),
                        help="2-block homogeneous SBM with equal-size communities; n, p_in, p_out")
    parser.add_argument(
        "--hrg", nargs=4, type=float, action="append",
        metavar=("n","R","alpha","T"),
        help="Hyperbolic random graph (native model): n, disk radius R, curvature -alpha^2, temperature T"
    )

    # Parallelism controls
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of parallel jobs for both HRG generation and curvature computation.")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Tile size for HRG; smaller -> more tasks. If omitted, chosen adaptively.")

    # Canonical graphs
    parser.add_argument("--cycle", nargs=1, type=int, action="append", metavar=("n",), help="Cycle C_n")
    parser.add_argument("--grid", nargs=2, type=int, action="append", metavar=("m","n"), help="Grid m x n")
    parser.add_argument("--torus", nargs=2, type=int, action="append", metavar=("m","n"), help="Toroidal grid C_m x C_n (wraparound)")
    parser.add_argument("--tree", nargs=2, type=int, action="append", metavar=("d","h"), help="d-ary tree of height h")
    parser.add_argument("--complete", nargs=1, type=int, action="append", metavar=("n",), help="Complete graph K_n")

    # Real networks
    parser.add_argument("--include-real", action="store_true", help="Include real networks present in experiments/data/*.csv")

    # Benchmarking controls
    parser.add_argument("--only-benchmark", action="store_true", help="Skip everything and run ONLY the runtime scaling benchmark.")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip the runtime benchmark phase during standard runs.")
    parser.add_argument("--bench-min-n", type=int, default=50, help="Minimum number of nodes for the runtime benchmark.")
    parser.add_argument("--bench-max-n", type=int, default=20000, help="Maximum number of nodes for the runtime benchmark.")
    parser.add_argument("--bench-graphs", type=int, default=100, help="Total number of graph instances evaluated in the benchmark.")
    parser.add_argument("--bench-seed", type=int, default=42, help="Random seed for the benchmark generation.")

    # Ablation controls
    parser.add_argument("--only-ablation", action="store_true", help="Skip everything and run ONLY the SDRF ablation study on minesweeper.")
    parser.add_argument("--no-ablation", action="store_true", help="Skip the SDRF ablation phase during standard runs.")
    parser.add_argument("--max-iters-rewiring", type=int, default=5, help="Maximum iterations for GNN rewiring")
    parser.add_argument("--max-iters-ablation", type=int, default=None, help="Maximum iterations for SDRF ablation grid")
    parser.add_argument("--max-n-ablation", type=int, default=None, help="Maximum candidates (n) for SDRF ablation grid")
    parser.add_argument("--ablation-points", type=int, default=None, help="Number of points in the logspace grids for SDRF ablation")
    
    # GNN Experiment controls
    parser.add_argument("--run-gnn", action="store_true", default=True, help="Execute the GNN topology rewiring experiments.")
    parser.add_argument("--only-gnn", action="store_true", help="Skip graph generation and OT benchmarks, run ONLY GNN experiments.")
    parser.add_argument("--gnn-datasets", nargs="+", type=str, default=None, help="Datasets to evaluate (e.g., ZINC minesweeper)")
    parser.add_argument("--gnn-archs", nargs="+", type=str, default=None, choices=["GCN", "GIN"], help="Architectures to evaluate")
    parser.add_argument("--gnn-trials", type=int, default=None, help="Number of Optuna trials for GNNs")
    parser.add_argument("--gnn-epochs", type=int, default=None, help="Number of epochs to train GNNs")
    parser.add_argument("--gnn-cv", type=int, default=None, help="Number of CV splits for GNNs")
    parser.add_argument("--gnn-reps", type=int, default=None, help="Number of repetitions for GNN evaluations")
    parser.add_argument("--gnn-optim", type=str, default="Adam", help="Optimizer for GNNs")

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None, help="Name subfolder under out/")
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--auto-figures", dest="auto-figures", action="store_true", help="Generate paper figures after the runs (always on for --preset paper)")
    parser.add_argument("--soft-restart", action="store_true", help="Resume in-place: for an existing out/<run-name> folder, skip recomputing any run whose {tag}_edges.csv already exists.")


def handle_presets(args, seed: int):
    """Fill in default arguments for --preset suites if not already set."""
    if args.preset is None:
        return

    if args.preset == "tiny":
        args.bench_max_n = 1000

        args.max_iters_ablation = args.max_iters_ablation or 10
        args.max_n_ablation = args.max_n_ablation or 100
        args.ablation_points = args.ablation_points or 10

        args.er = args.er or [[150, 0.02]]
        args.ws = args.ws or [[150, 6, 0.1]]
        args.ba = args.ba or [[150, 2]]
        args.rg = args.rg or [[150, 0.12]]
        args.cycle = args.cycle or [[100]]
        args.grid = args.grid or [[10, 12]]
        args.tree = args.tree or [[3, 5]]
        args.complete = args.complete or [[40]]
        
        args.gnn_datasets = args.gnn_datasets or ["minesweeper"]
        args.gnn_archs = args.gnn_archs or ["GCN", "GIN", "GAT"]
        args.gnn_reps = args.gnn_reps or 2
        args.gnn_cv = args.gnn_cv or 3
        args.gnn_trials = args.gnn_trials or 5
        args.gnn_epochs = args.gnn_epochs or 50
        args.max_iters_rewiring = args.max_iters_rewiring or 5

        

    elif args.preset == "small":
        args.bench_max_n = 5000

        args.max_n_ablation = args.max_n_ablation or 200
        args.max_iters_ablation = args.max_iters_ablation or 20
        args.ablation_points = args.ablation_points or 20

        args.bench_graphs = 20
        args.er = args.er or [[400, 0.02], [400, 0.04]]
        args.ws = args.ws or [[400, 6, 0.1], [400, 8, 0.2]]
        args.ba = args.ba or [[400, 2], [400, 3]]
        args.rg = args.rg or [[400, 0.09]]
        args.cycle = args.cycle or [[200]]
        args.grid = args.grid or [[20, 20]]
        args.tree = args.tree or [[3, 6]]
        args.complete = args.complete or [[60]]
        
        args.gnn_datasets = args.gnn_datasets or ["ZINC", "Peptides-func", "Peptides-struct", "minesweeper"]
        args.gnn_archs = args.gnn_archs or ["GCN", "GIN"]
        args.gnn_reps = args.gnn_reps or 10
        args.gnn_cv = args.gnn_cv or 5
        args.gnn_trials = args.gnn_trials or 10
        args.gnn_epochs = args.gnn_epochs or 100
        args.max_iters_rewiring = args.max_iters_rewiring or 10

    elif args.preset == "medium":
        args.bench_max_n = 10000

        args.max_n_ablation = args.max_n_ablation or 400
        args.max_iters_ablation = args.max_iters_ablation or 30
        args.ablation_points = args.ablation_points or 30

        args.bench_graphs = 50
        args.hrg = args.hrg or [[800, 5.0, 1.0, 0.0], [800, 5.0, 1.0, 0.5]]
        args.er  = args.er  or [[800, 0.0100125]]
        args.ws  = args.ws  or [[800, 10, 0.05], [800, 10, 0.2]]
        args.ba  = args.ba  or [[800, 2], [800, 5]]
        args.rg  = args.rg  or [[800, 0.056419]]
        args.rreg = args.rreg or [[1000, 8]]
        args.sbm2 = args.sbm2 or [[1000, 0.012018, 0.004006], [1000, 0.004002, 0.012006]]
        
        args.cycle = args.cycle or [[400]]
        args.grid  = args.grid  or [[30, 30]]
        args.torus = args.torus or [[32, 32]]
        args.tree  = args.tree  or [[4, 5]] 
        args.complete = args.complete or [[90]]
        args.skip_plots = True
        
        args.gnn_datasets = args.gnn_datasets or ["ZINC", "Peptides-func", "Peptides-struct", "minesweeper"]
        args.gnn_archs = args.gnn_archs or ["GCN", "GIN", "GAT"]
        args.gnn_reps = args.gnn_reps or 20
        args.gnn_cv = args.gnn_cv or 5
        args.gnn_trials = args.gnn_trials or 30
        args.gnn_epochs = args.gnn_epochs or 200
        args.max_iters_rewiring = args.max_iters_rewiring or 30

    elif args.preset == "paper":
        args.bench_max_n = 15000

        args.max_n_ablation = args.max_n_ablation or 800
        args.max_iters_ablation = args.max_iters_ablation or 40
        args.ablation_points = args.ablation_points or 50

        args.bench_graphs = 100
        args.hrg = args.hrg or [[800, 5.0, 1.0, 0.0], [800, 5.0, 1.0, 0.5]]
        args.er  = args.er  or [[800, 0.0100125], [1600, 0.0050031]]
        args.ws  = args.ws  or [[800, 10, 0.05], [800, 10, 0.2], [1600, 10, 0.05], [1600, 10, 0.2]]
        args.ba  = args.ba  or [[800, 2], [800, 5], [1600, 2], [1600, 5]]
        args.rg  = args.rg  or [[800, 0.056419], [1600, 0.039894]]
        args.rreg = args.rreg or [[1000, 8], [2000, 8]]
        args.sbm2 = args.sbm2 or [[1000, 0.012018, 0.004006], [1000, 0.004002, 0.012006]]
        args.cycle = args.cycle or [[600]]
        args.grid  = args.grid  or [[40, 40]]
        args.torus = args.torus or [[32, 32], [40, 40]]
        args.tree  = args.tree  or [[4, 6]]
        args.complete = args.complete or [[120]]
        args.include_real = True
        args.skip_plots = True
        
        args.gnn_datasets = args.gnn_datasets or ["ZINC", "Peptides-func", "Peptides-struct", "minesweeper"]
        args.gnn_archs = args.gnn_archs or ["GCN", "GIN", "GAT"]
        args.gnn_reps = args.gnn_reps or 100
        args.gnn_cv = args.gnn_cv or 5
        args.gnn_trials = args.gnn_trials or 100
        args.gnn_epochs = args.gnn_epochs or 500
        args.max_iters_rewiring = args.max_iters_rewiring or 60

def load_real_graphs(data_dir: str) -> List[Tuple[str, int, List[Tuple[int,int]]]]:
    out: List[Tuple[str, int, List[Tuple[int,int]]]] = []
    dataset_names = ["karate", "jazz", "power_grid", "yeast", "arxiv"]
    for name in dataset_names:
        base_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(base_path):
            continue
        fh = open(base_path, mode="r", encoding="utf-8", newline="")
        edges = set()
        try:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                first = (row[0] or "").strip()
                if first.startswith("#"):
                    continue
                try:
                    u = int(first)
                    v = int((row[1] or "").strip())
                except (ValueError, IndexError):
                    continue
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        finally:
            fh.close()
        n = 1 + max((max(u, v) for (u, v) in edges), default=-1)
        if n < 0:
            n = 0
        out.append((name, n, sorted(edges)))
    return out

def np_encoder(obj):
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type not serializable: {type(obj)}")

def main():
    parser = argparse.ArgumentParser(description="Run curvature distribution experiments.")
    parser.add_argument("--suppress-warning", action="store_true", help="Suppress all Python runtime warnings.")
    add_preset_args(parser)
    add_family_args(parser)
    args = parser.parse_args()

    if args.suppress_warning:
        warnings.filterwarnings("ignore")

    seed = int(args.seed)
    handle_presets(args, seed)

    if getattr(args, "preset", None) == "paper" and not getattr(args, "auto-figures", False):
        setattr(args, "auto-figures", True)

    run_name = args.run_name or ("preset_" + args.preset if args.preset else "custom")
    out_dir = os.path.join(os.path.dirname(__file__), "out", run_name)
    ensure_dir(out_dir)

    def run_benchmark_phase():
        try:
            import benchmark_run
            bench_csv = os.path.join(out_dir, "benchmark_runtime_scaling.csv")
            print(f"\n[benchmark] Executing OT vs. Bounds runtime scaling -> {bench_csv}")
            benchmark_run.run_benchmark(
                output_csv=bench_csv,
                max_workers=args.jobs,
                min_n=args.bench_min_n,
                n_benchmark=args.bench_max_n,
                num_graphs=args.bench_graphs,
                seed=args.bench_seed,
                auto_figures=getattr(args, "auto-figures", False)
            )
        except Exception as e:
            print(f"[benchmark] Runtime benchmark failed: {e}")

    def run_ablation_phase():
        try:
            import sdrf_ablation
            auto_figures = getattr(args, "auto-figures", getattr(args, "auto-figures", False))
            sdrf_ablation.execute_ablation(
                seed=args.seed,
                out_dir=out_dir,
                preset=args.preset,
                max_iters_limit=args.max_iters_ablation,
                max_n_limit=args.max_n_ablation,
                ablation_points=args.ablation_points,
                auto_figures=auto_figures
            )
        except Exception as e:
            print(f"[ablation] Execution failed: {e}")
    
    # Execute isolated branches
    if getattr(args, "only_benchmark", False):
        run_benchmark_phase()
        return

    if getattr(args, "only_ablation", False):
        run_ablation_phase()
        return

    if not getattr(args, "only_gnn", False):
        prev_summary = {}
        man_path = os.path.join(out_dir, "manifest.json")
        if getattr(args, "soft_restart", False):
            if os.path.exists(man_path):
                try:
                    _prev = json.load(open(man_path, "r"))
                    for r in _prev.get("runs", []):
                        t = r.get("tag")
                        if t:
                            prev_summary[t] = r.get("summary")
                except Exception:
                    prev_summary = {}
                    print(f"[warn] Could not load prior manifest from {man_path}")
            else:
                print(f"[info] No manifest found at {man_path}; rebuilding from existing CSVs.")

        def _csv_base(tag: str) -> str:
            return tag.replace(".", "_")

        def _csv_path_for(tag: str) -> str:
            return os.path.join(out_dir, f"{_csv_base(tag)}_edges.csv")

        def _infer_nm_from_csv(path_csv: str) -> Tuple[int, int]:
            import csv as _csv
            m = 0
            max_id = -1
            with open(path_csv, "r", newline="") as fh:
                rd = _csv.reader(fh)
                _ = next(rd, None)
                for row in rd:
                    if not row or len(row) < 2:
                        continue
                    try:
                        u = int(float(row[0])); v = int(float(row[1]))
                    except Exception:
                        continue
                    m += 1
                    if u > max_id: max_id = u
                    if v > max_id: max_id = v
            n = (max_id + 1) if m > 0 else 0
            return n, m

        to_compute = []
        skipped = []

        def plan_run(tag: str, gen_callable):
            csv_path = _csv_path_for(tag)
            if getattr(args, "soft_restart", False) and os.path.exists(csv_path):
                print(f"[skip] {tag}: found existing CSV -> {os.path.relpath(csv_path, out_dir)}")
                skipped.append((tag, csv_path))
            else:
                to_compute.append((tag, gen_callable))

        if args.er:
            for n, p in args.er:
                tag = "er_n{}_p{}".format(int(n), float(p))
                plan_run(tag, lambda n=int(n), p=float(p): models.erdos_renyi(int(n), float(p), seed=seed))
        if args.ws:
            for n, k, beta in args.ws:
                tag = "ws_n{}_k{}_b{}".format(int(n), int(k), float(beta))
                plan_run(tag, lambda n=int(n), k=int(k), beta=float(beta): models.watts_strogatz(int(n), int(k), float(beta), seed=seed))
        if args.ba:
            for n, m in args.ba:
                tag = "ba_n{}_m{}".format(int(n), int(m))
                plan_run(tag, lambda n=int(n), m=int(m): models.barabasi_albert(int(n), int(m), seed=seed))
        if args.rg:
            for n, r in args.rg:
                tag = "rg_n{}_r{}".format(int(n), float(r))
                plan_run(tag, lambda n=int(n), r=float(r): models.random_geometric(int(n), float(r), seed=seed))
        if args.rreg:
            for n, d in args.rreg:
                tag = "rreg_n{}_d{}".format(int(n), int(d))
                plan_run(tag, lambda n=int(n), d=int(d): models.d_regular_graph(int(n), int(d), seed=seed))
        if args.sbm2:
            for n, p_in, p_out in args.sbm2:
                n = int(n); p_in = float(p_in); p_out = float(p_out)
                a = n // 2; b = n - a
                tag = "sbm2_n{}_pin{}_pout{}".format(n, p_in, p_out)
                plan_run(tag, lambda a=a, b=b, p_in=p_in, p_out=p_out: models.make_sbm_graph([a, b], float(p_in), float(p_out), seed=seed))
        if hasattr(args, "hrg") and args.hrg:
            for n, R, alpha, T in args.hrg:
                tag = "hrg_n{}_R{}_a{}_T{}".format(int(n), float(R), float(alpha), float(T))
                plan_run(tag, lambda n=int(n), R=float(R), alpha=float(alpha), T=float(T):
                         models.make_hyperbolic_random_graph(
                             int(n), float(R), alpha=float(alpha), T=float(T),
                             seed=seed, n_jobs=args.jobs, block_size=args.block_size))

        if args.cycle:
            for (n,) in args.cycle:
                tag = "cycle_n{}".format(int(n))
                plan_run(tag, lambda n=int(n): models.cycle_graph(int(n)))
        if args.grid:
            for m, n in args.grid:
                tag = "grid_{}x{}".format(int(m), int(n))
                plan_run(tag, lambda m=int(m), n=int(n): models.grid_graph(int(m), int(n)))
        if args.torus:
            for m, n in args.torus:
                tag = "torus_{}x{}".format(int(m), int(n))
                plan_run(tag, lambda m=int(m), n=int(n): models.torus_graph(int(m), int(n)))
        if args.tree:
            for d, h in args.tree:
                tag = "tree_d{}_h{}".format(int(d), int(h))
                plan_run(tag, lambda d=int(d), h=int(h): models.dary_tree(int(d), int(h)))
        if args.complete:
            for (n,) in args.complete:
                tag = "complete_n{}".format(int(n))
                plan_run(tag, lambda n=int(n): models.complete_graph(int(n)))

        if args.include_real:
            for name, n, edges in load_real_graphs(os.path.join(os.path.dirname(__file__), "data")):
                tag = f"real_{name}"
                plan_run(tag, lambda n=n, edges=edges: (n, edges))

        manifest = {"runs": [], "notes": "Distributional histograms and envelope/transfer coverage"}

        for tag, gen in to_compute:
            now_iso_string = datetime.now().strftime("%Y-%m-%d--%H:%M:%S%z")
            
            n, edges = gen()
            print(f"{now_iso_string} [run] {tag}: n={n}, m={len(edges)}", end="")
            
            time_start = time.time()
            curv = compute_curvatures(n, edges, n_jobs=args.jobs)
            time_end = time.time()
            total_seconds = int(time_end - time_start)
            h, rem = divmod(total_seconds, 3600)
            m, s = divmod(rem, 60)
            print(f", t:{h:02d}:{m:02d}:{s:02d}s")
            
            base_name = tag.replace(".", "_")
            if not args.skip_csv:
                write_edge_table(os.path.join(out_dir, f"{base_name}_edges.csv"), curv)
                
            summary = summarize_run(curv)
            manifest["runs"].append({"tag": tag, "n": n, "m": len(edges), "summary": summary})
            
            if not args.skip_plots:
                _plot_hist(curv.base["c_OR"], f"{tag}   c_OR", os.path.join(out_dir, f"{base_name}__hist_cOR.png"), bins=args.bins)
                _plot_hist(curv.base["c_BF"], f"{tag}   c_BF", os.path.join(out_dir, f"{base_name}__hist_cBF.png"), bins=args.bins)
                _plot_hist(curv.theta_at_t - curv.base["c_OR"], f"{tag}   slack Theta(tri) - c_OR", os.path.join(out_dir, f"{base_name}__hist_slack_theta.png"), bins=args.bins)
                _plot_hist(curv.env_upper - curv.base["c_OR"], f"{tag}   slack envelope - c_OR", os.path.join(out_dir, f"{base_name}__hist_slack_env.png"), bins=args.bins)
                
        for tag, csv_path in skipped:
            n, m = _infer_nm_from_csv(csv_path)
            manifest["runs"].append({"tag": tag, "n": n, "m": m, "summary": prev_summary.get(tag)})

        if getattr(args, "soft_restart", False):
            seen_bases = { (r.get("tag","")).replace(".", "_") for r in manifest["runs"] }
            for path in glob.glob(os.path.join(out_dir, "*_edges.csv")):
                 base = os.path.basename(path)[:-len("_edges.csv")]
                 if base in seen_bases:
                     continue
                 n, m = _infer_nm_from_csv(path)
                 manifest["runs"].append({"tag": base, "n": n, "m": m, "summary": prev_summary.get(base)})
            if not manifest["runs"]:
                print(f"[info] Soft-restart found no *_edges.csv in {out_dir}.")

        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=np_encoder)

        if getattr(args, "auto-figures", False):
            try:
                base_out = os.path.dirname(out_dir)
                run_name = os.path.basename(out_dir)
                print(f"[run_experiments] Generating paper figures for run '{run_name}'")
                generate_paper_figures(out_root=base_out, run_name=run_name, bins=args.bins)
            except Exception as e:
                print(f"[run_experiments] Paper figure generation failed: {e}")
        if not getattr(args, "no_benchmark", False):
            run_benchmark_phase()
                
        if not getattr(args, "no_ablation", False):
            run_ablation_phase()

    if getattr(args, "run_gnn", False) or getattr(args, "only_gnn", False):
        try:
            import gnn_experiments
            datasets = args.gnn_datasets or ["minesweeper"]
            archs = args.gnn_archs or ["GCN", "GIN"]
            trials = args.gnn_trials or 10
            epochs = args.gnn_epochs or 100
            cv_split = args.gnn_cv or 5
            reps = args.gnn_reps or 3
            optim = args.gnn_optim or "Adam"
            
            print(f"\n{'='*50}\n[gnn_experiments] Starting GNN Evaluations\n{'='*50}")
            for ds in datasets:
                for arch in archs:
                    gnn_experiments.execute_trials(
                        dataset_name=ds,
                        seed=args.seed,
                        n_trials=trials,
                        epochs=epochs,
                        optimizer_name=optim,
                        cv_split=cv_split,
                        arch_name=arch,
                        n_reps=reps,
                        out_dir=out_dir,
                        max_iters_rewiring=args.max_iters_rewiring
                    )
        except Exception as e:
            print(f"[gnn_experiments] GNN evaluation failed: {e}")

    print(f"\n[done] Outputs directed to {out_dir}")

if __name__ == "__main__":
    main()
