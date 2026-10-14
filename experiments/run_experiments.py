#!/usr/bin/env python3
"""Experiment orchestrator.

Phases (each can be run alone):
  distributions  per-edge tables (curvatures, bounds, intervals) for every graph
                 family, a manifest, and the paper figures (make_paper_figures.py)
  bounds         Tables 3-5 and the pending quantities (compare_bounds.py)
  screening      certify-then-solve (screening.py)
  benchmark      runtime scaling (benchmark.py)
  ablation       SDRF ablation on Minesweeper (sdrf_ablation.py; needs torch)
  gnn            rewiring case study (gnn_experiments.py; needs torch)

Examples
  python run_experiments.py --preset tiny
  python run_experiments.py --preset paper --phases distributions bounds screening benchmark
  python run_experiments.py --er 800 0.01 --ws 800 10 0.2 --include-real --phases distributions
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import generators as gen  # noqa: E402
import suite  # noqa: E402
from curvature_transfer import CurvatureEngine, Graph  # noqa: E402

PHASES = ["distributions", "bounds", "screening", "benchmark", "ablation", "gnn"]

PRESETS = {
    "tiny": dict(er=[[150, 0.02]], ws=[[150, 6, 0.1]], ba=[[150, 2]], rg=[[150, 0.12]], cycle=[[100]],
                 grid=[[10, 12]], tree=[[3, 5]], complete=[[40]], bench_max_n=1000, bench_graphs=6,
                 bench_families=["ER"], phases=["distributions", "benchmark"]),
    "small": dict(er=[[400, 0.02], [400, 0.04]], ws=[[400, 6, 0.1], [400, 8, 0.2]], ba=[[400, 2], [400, 3]],
                  rg=[[400, 0.09]], cycle=[[200]], grid=[[20, 20]], tree=[[3, 6]], complete=[[60]],
                  bench_max_n=5000, bench_graphs=20, bench_families=["ER"],
                  phases=["distributions", "bounds", "screening", "benchmark"]),
    "paper": dict(hrg=[[800, 5.0, 1.0, 0.0], [800, 5.0, 1.0, 0.5]], er=[[800, 0.0100125], [1600, 0.0050031]],
                  ws=[[800, 10, 0.05], [800, 10, 0.2], [1600, 10, 0.05], [1600, 10, 0.2]],
                  ba=[[800, 2], [800, 5], [1600, 2], [1600, 5]], rg=[[800, 0.056419], [1600, 0.039894]],
                  rreg=[[1000, 8], [2000, 8]], sbm2=[[1000, 0.012018, 0.004006], [1000, 0.004002, 0.012006]],
                  cycle=[[600]], grid=[[40, 40]], torus=[[32, 32], [40, 40]], tree=[[4, 6]], complete=[[120]],
                  include_real=True, bench_max_n=15000, bench_graphs=40, bench_families=["ER", "BA", "RGG"],
                  phases=["distributions", "bounds", "screening", "benchmark"]),
}


def plan(args):
    """(tag, generator) pairs; tags match the preliminary file names."""
    s = args.seed
    out = []
    for n, p in args.er or []:
        out.append((f"er_n{int(n)}_p{float(p)}", lambda n=int(n), p=float(p): gen.erdos_renyi(n, p, seed=s)))
    for n, k, b in args.ws or []:
        out.append((f"ws_n{int(n)}_k{int(k)}_b{float(b)}", lambda n=int(n), k=int(k), b=float(b): gen.watts_strogatz(n, k, b, seed=s)))
    for n, m in args.ba or []:
        out.append((f"ba_n{int(n)}_m{int(m)}", lambda n=int(n), m=int(m): gen.barabasi_albert(n, m, seed=s)))
    for n, r in args.rg or []:
        out.append((f"rg_n{int(n)}_r{float(r)}", lambda n=int(n), r=float(r): gen.random_geometric(n, r, seed=s)))
    for n, d in args.rreg or []:
        out.append((f"rreg_n{int(n)}_d{int(d)}", lambda n=int(n), d=int(d): gen.d_regular_graph(n, d, seed=s)))
    for n, pin, pout in args.sbm2 or []:
        n = int(n)
        out.append((f"sbm2_n{n}_pin{float(pin)}_pout{float(pout)}",
                    lambda n=n, pin=float(pin), pout=float(pout): gen.make_sbm_graph([n // 2, n - n // 2], pin, pout, seed=s)))
    for n, R, a, T in args.hrg or []:
        out.append((f"hrg_n{int(n)}_R{float(R)}_a{float(a)}_T{float(T)}",
                    lambda n=int(n), R=float(R), a=float(a), T=float(T): suite.hrg(n, R, a, T, s)))
    for (n,) in args.cycle or []:
        out.append((f"cycle_n{int(n)}", lambda n=int(n): gen.cycle_graph(n)))
    for m, n in args.grid or []:
        out.append((f"grid_{int(m)}x{int(n)}", lambda m=int(m), n=int(n): gen.grid_graph(m, n)))
    for m, n in args.torus or []:
        out.append((f"torus_{int(m)}x{int(n)}", lambda m=int(m), n=int(n): gen.torus_graph(m, n)))
    for d, h in args.tree or []:
        out.append((f"tree_d{int(d)}_h{int(h)}", lambda d=int(d), h=int(h): gen.dary_tree(d, h)))
    for (n,) in args.complete or []:
        out.append((f"complete_n{int(n)}", lambda n=int(n): gen.complete_graph(n)))
    if args.include_real:
        for name, n, e in suite.real_graphs():
            out.append((f"real_{name.lower().replace(' ', '_')}", lambda n=n, e=e: (n, e)))
    return out


def summarize(df):
    eps = 1e-9
    q = lambda a: {k: float(v) for k, v in zip(["min", "q05", "q25", "median", "q75", "q95", "max"],
                                               np.quantile(a, [0, .05, .25, .5, .75, .95, 1]))}
    c = df["c_OR"].to_numpy()
    out = {"c_OR": q(c), "c_BF": q(df["c_BF"].to_numpy()),
           "coverage": {
               "OR_in_BF_interval": float(np.mean((c + eps >= df["cOR_lower_from_cBF"]) & (c <= df["cOR_upper_from_cBF"] + eps))),
               "OR_in_sharpened": float(np.mean((c + eps >= df["sharpened_lower"]) & (c <= df["sharpened_upper"] + eps))),
               "BF_in_OR_to_BF_band": float(np.mean((df["c_BF"] + eps >= df["cBF_lower_from_cOR"]) & (df["c_BF"] <= df["cBF_upper_from_cOR"] + eps))),
           },
           "band_widths": {
               "BF_interval_mean": float(np.mean(df["cOR_upper_from_cBF"] - df["cOR_lower_from_cBF"])),
               "sharpened_mean": float(np.mean(df["sharpened_upper"] - df["sharpened_lower"])),
               "OR_to_BF_mean": float(np.mean(df["cBF_upper_from_cOR"] - df["cBF_lower_from_cOR"])),
           }}
    return out


def run_distributions(args, out_dir):
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = {"runs": []}
    if args.soft_restart and os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
    seen = {r["tag"] for r in manifest["runs"]}
    for tag, make in plan(args):
        path = os.path.join(out_dir, f"{tag.replace('.', '_')}_edges.csv")
        if args.soft_restart and os.path.exists(path) and tag in seen:
            print(f"[skip] {tag}")
            continue
        n, edges = make()
        t0 = time.time()
        eng = CurvatureEngine(Graph.from_edges(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n),
                              n_jobs=args.jobs, varpi=args.varpi)
        df = eng.edge_table(exact=True)
        df.to_csv(path, index=False, float_format="%.12g")
        manifest["runs"] = [r for r in manifest["runs"] if r["tag"] != tag]
        manifest["runs"].append({"tag": tag, "n": n, "m": int(len(df)), "summary": summarize(df)})
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        print(f"[distributions] {tag}: n={n} m={len(df)} ({time.time() - t0:.1f}s)", flush=True)
    if args.figures:
        from make_paper_figures import generate_paper_figures
        generate_paper_figures(out_root=os.path.dirname(out_dir), run_name=os.path.basename(out_dir), bins=args.bins)


def _script(name, extra):
    cmd = [sys.executable, os.path.join(HERE, name)] + extra
    print("[run]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--phases", nargs="+", choices=PHASES, default=None)
    for flag, k in [("--er", 2), ("--ba", 2), ("--rg", 2), ("--rreg", 2), ("--grid", 2), ("--torus", 2), ("--tree", 2),
                    ("--ws", 3), ("--sbm2", 3), ("--hrg", 4), ("--cycle", 1), ("--complete", 1)]:
        ap.add_argument(flag, nargs=k, type=float, action="append")
    ap.add_argument("--include-real", action="store_true")
    ap.add_argument("--hrg-jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None, help="Numba threads (-1 = all)")
    ap.add_argument("--varpi", choices=["paper", "legacy"], default="paper")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", default=os.path.join(HERE, "out"))
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--figures", action="store_true", help="render the paper figures after the distributions phase")
    ap.add_argument("--soft-restart", action="store_true", help="skip graphs whose edge table already exists")
    ap.add_argument("--bench-min-n", type=int, default=50)
    ap.add_argument("--bench-max-n", type=int, default=None)
    ap.add_argument("--bench-graphs", type=int, default=None)
    ap.add_argument("--bench-families", nargs="+", default=None, choices=["ER", "BA", "RGG"])
    ap.add_argument("--legacy-dir", default=None, help="preliminary repository, to time its solver too")
    ap.add_argument("--gnn-args", nargs=argparse.REMAINDER, default=[],
                    help="arguments forwarded verbatim to gnn_experiments.py (everything after this flag)")
    args = ap.parse_args()

    p = PRESETS.get(args.preset, {})
    for k, v in p.items():
        if k != "phases" and getattr(args, k, None) in (None, False, []):
            setattr(args, k, v)
    phases = args.phases or p.get("phases") or ["distributions"]
    if args.preset == "paper":
        args.figures = True
    out_dir = os.path.join(args.out_root, args.run_name or (f"preset_{args.preset}" if args.preset else "custom"))
    os.makedirs(out_dir, exist_ok=True)
    jobs = [] if args.jobs is None else ["--jobs", str(args.jobs)]

    if "distributions" in phases:
        run_distributions(args, out_dir)
    if "bounds" in phases:
        _script("compare_bounds.py", ["--out", os.path.join(out_dir, "bounds"), "--varpi", args.varpi] + jobs)
    if "screening" in phases:
        _script("screening.py", ["--out", os.path.join(out_dir, "screening")] + jobs)
    if "benchmark" in phases:
        for fam in args.bench_families or ["ER"]:
            extra = ["--family", fam, "--min-n", str(args.bench_min_n), "--max-n", str(args.bench_max_n or 15000),
                     "--graphs", str(args.bench_graphs or 40), "--out", os.path.join(out_dir, "benchmark")] + jobs
            if args.legacy_dir:
                extra += ["--legacy-dir", args.legacy_dir]
            _script("benchmark.py", extra)
    if "ablation" in phases:
        _script("sdrf_ablation.py", [])
    if "gnn" in phases:
        _script("gnn_experiments.py", args.gnn_args or ["--dataset", "Tree-NeighborsMatch", "--out-dir", out_dir])
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
