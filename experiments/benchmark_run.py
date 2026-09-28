#!/usr/bin/env python3
import os
import sys
import csv
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyg_curvature as pc
import models
from util_curvature import Data, make_edge_index

_SHARED_ENGINE = None

def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _init_benchmark_worker(state_dict, seed):
    set_deterministic_seed(seed)
    global _SHARED_ENGINE
    _SHARED_ENGINE = pc.CurvatureEngine.__new__(pc.CurvatureEngine)
    _SHARED_ENGINE.edges = state_dict["edges"]
    _SHARED_ENGINE.neighbors = state_dict["neighbors"]
    _SHARED_ENGINE.deg = state_dict["deg"]
    _SHARED_ENGINE._cache = {}

def worker_ot(chunk):
    t0 = time.perf_counter()
    for eidx in chunk:
        _SHARED_ENGINE.c_OR_edge(eidx)
    return time.perf_counter() - t0

def worker_ours(chunk):
    t0 = time.perf_counter()
    for eidx in chunk:
        # Transfer moduli evaluation (sharp=False avoids evaluating exact c_OR0 behind the scenes)
        bf = _SHARED_ENGINE.c_BF_edge(eidx)
        _SHARED_ENGINE.varphi_BF_to_OR_edge(eidx, bf, sharp=False)
        _SHARED_ENGINE.psi_BF_to_OR_edge(eidx, bf)
    return time.perf_counter() - t0

def worker_bf(chunk):
    t0 = time.perf_counter()
    for eidx in chunk:
        _SHARED_ENGINE.c_BF_edge(eidx)
    return time.perf_counter() - t0

def worker_prior(chunk):
    t0 = time.perf_counter()
    for eidx in chunk:
        loc = _SHARED_ENGINE._local_for_edge(eidx)
        di, dj, t = float(loc.deg_i), float(loc.deg_j), float(loc.tri)
        if min(di, dj) == 0: continue
        
        # JL lower
        K = 1.0 - 1.0 / di - 1.0 / dj
        zmax, zmin = t / max(di, dj), t / min(di, dj)
        lo_jl = zmax - max(K - zmax, 0.0) - max(K - zmin, 0.0)
        
        # TLW lower
        nUi, nUj = max(0.0, di - 1.0 - t), max(0.0, dj - 1.0 - t)
        mx, my = 1.0 / di, 1.0 / dj
        Lx = nUi * mx
        cc = t * abs(mx - my)
        cross = Lx - my - t * max(my - mx, 0.0)
        lo_tlw = 1.0 - nUi * mx - nUj * my - cc - abs(cross)
    return time.perf_counter() - t0

def generate_runtime_plot(csv_path: str, out_pdf: str, out_png: str):
    ns, ot_times, ours_times, bf_times, prior_times = [], [], [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ns.append(int(row["n"]))
                ot_times.append(float(row["ot_time_sec"]))
                ours_times.append(float(row["ours_time_sec"]))
                bf_times.append(float(row["bf_time_sec"]))
                prior_times.append(float(row["prior_time_sec"]))
            except (ValueError, KeyError):
                continue

    if not ns:
        print("[benchmark_plot] No valid data found in CSV to plot.")
        return

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(ns, ot_times, marker='o', markersize=8, color='#0072B2', 
            linestyle='-', linewidth=2.5, label='Exact Ollivier-Ricci')
    ax.plot(ns, ours_times, marker='s', markersize=8, color='#D55E00', 
            linestyle='--', linewidth=2.5, label='Proposed Moduli')
    ax.plot(ns, bf_times, marker='^', markersize=8, color='#009E73', 
            linestyle=':', linewidth=2.5, label='Balanced Forman')
    ax.plot(ns, prior_times, marker='v', markersize=8, color='#CC79A7', 
            linestyle='-.', linewidth=2.5, label='JL / TLW Bounds')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Number of Nodes (n)')
    ax.set_ylabel('Execution Time (s)')
    
    ax.grid(True, which="both", linestyle="--", alpha=0.5, color='lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='upper left', frameon=False)
    
    plt.tight_layout()
    
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[benchmark] Rendered scaling plot -> {out_pdf}")

def run_benchmark(
    output_csv: str = "data/benchmark_runtime_scaling.csv", 
    max_workers: int = None,
    min_n: int = 50,
    n_benchmark: int = 20000,
    num_graphs: int = 100,
    seed: int = 42,
    auto_figures: bool = False
):
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    out_dir = os.path.dirname(output_csv)
    os.makedirs(out_dir, exist_ok=True)
    set_deterministic_seed(seed)
    
    completed_n = set()
    expected_cols = ["n", "p", "num_edges", "avg_deg", "ot_time_sec", "ours_time_sec", "bf_time_sec", "prior_time_sec"]

    if os.path.exists(output_csv):
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if all(row.get(col) is not None and str(row[col]).strip() != "" for col in expected_cols):
                    try:
                        completed_n.add(int(row["n"]))
                    except ValueError:
                        pass
    else:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(expected_cols)

    raw_n_list = np.geomspace(min_n, n_benchmark, num=num_graphs, dtype=int)
    n_list = sorted(np.unique(raw_n_list).tolist())

    for n in n_list:
        if n in completed_n:
            continue
            
        p = float(0.2 * (50 / n) ** 0.6)
        print(f"--- Benchmarking n={n}, p={p:.5f} ---")
        
        _, edges = models.erdos_renyi(n, p, seed=seed)
        edge_index = make_edge_index(n, edges)
        data = Data(num_nodes=n, edge_index=edge_index)
        
        eng_main = pc.CurvatureEngine(data, n_jobs=1)
        M = len(eng_main.edges)
        
        if M == 0:
            print(f"Skipping n={n}: 0 edges generated.")
            continue
            
        avg_deg = np.mean(eng_main.deg)
        print(f"Structure: {M} undirected edges, Average Degree: {avg_deg:.1f}")
        
        worker_state = eng_main._build_worker_state()
        
        num_chunks = min(M, max_workers * 8)
        chunks = np.array_split(range(M), num_chunks)
        chunks = [c.tolist() for c in chunks if len(c) > 0]

        t_ot_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_benchmark_worker, initargs=(worker_state, seed)) as ex:
            list(ex.map(worker_ot, chunks))
        t_ot_total = time.perf_counter() - t_ot_start
        print(f"Exact OT Time: {t_ot_total:.4f}s")
        
        t_ours_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_benchmark_worker, initargs=(worker_state, seed)) as ex:
            list(ex.map(worker_ours, chunks))
        t_ours_total = time.perf_counter() - t_ours_start
        print(f"Proposed Moduli Time: {t_ours_total:.4f}s")

        t_bf_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_benchmark_worker, initargs=(worker_state, seed)) as ex:
            list(ex.map(worker_bf, chunks))
        t_bf_total = time.perf_counter() - t_bf_start
        print(f"Balanced Forman Time: {t_bf_total:.4f}s")

        t_prior_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_benchmark_worker, initargs=(worker_state, seed)) as ex:
            list(ex.map(worker_prior, chunks))
        t_prior_total = time.perf_counter() - t_prior_start
        print(f"JL / TLW Bounds Time: {t_prior_total:.4f}s")
        
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n, p, M, f"{avg_deg:.2f}", f"{t_ot_total:.4f}", f"{t_ours_total:.4f}", f"{t_bf_total:.4f}", f"{t_prior_total:.4f}"])

    if auto_figures:
        out_pdf = os.path.join(out_dir, "runtime_scaling_loglog.pdf")
        out_png = os.path.join(out_dir, "runtime_scaling_loglog.png")
        generate_runtime_plot(output_csv, out_pdf, out_png)

    return output_csv

if __name__ == "__main__":
    run_benchmark(auto_figures=True)
