import os
import time
import copy
import json
import torch
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from torch_geometric.datasets import HeterophilousGraphDataset

from curvature_rewiring import SDRFRewiring

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def execute_ablation(seed, out_dir, preset, max_iters_limit, max_n_limit=None, ablation_points=None, auto_figures=False):
    set_seed(seed)
    
    print(f"[ablation] Loading 'minesweeper' dataset...")
    dataset = HeterophilousGraphDataset(root="data/Heterophilous", name="minesweeper")
    data = dataset[0]
    num_nodes = data.num_nodes
    
    print(f"[ablation] Base Graph: |V| = {num_nodes}, |E| = {data.edge_index.shape[1] // 2}")
    print(f"[ablation] Building log-space grid up to n={max_n_limit}, iters={max_iters_limit}")
    
    n_vals = np.unique(np.logspace(0, np.log10(max(1, max_n_limit)), num=ablation_points, dtype=int))
    iters_vals = np.unique(np.logspace(0, np.log10(max(1, max_iters_limit)), num=ablation_points, dtype=int))
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sdrf_ablation_results.json")
    
    time_matrix = np.zeros((len(iters_vals), len(n_vals)))
    completed_runs = set()
    
    if os.path.isfile(out_file):
        print(f"[ablation] Discovered existing checkpoint. Soft-restarting from {out_file}...")
        with open(out_file, "r") as f:
            results = json.load(f)
            
        for run in results.get("runs", []):
            run_n = run["max_candidates"]
            run_iters = run["max_iters"]
            completed_runs.add((run_n, run_iters))
            
            i_idx = np.where(iters_vals == run_iters)[0]
            j_idx = np.where(n_vals == run_n)[0]
            if i_idx.size > 0 and j_idx.size > 0:
                time_matrix[i_idx[0], j_idx[0]] = run["time_seconds"]
    else:
        results = {
            "dataset": "minesweeper",
            "preset_used": preset,
            "num_nodes": num_nodes,
            "max_n_limit": int(max_n_limit),
            "max_iters_limit": int(max_iters_limit),
            "runs": []
        }
    
    for i, iters in enumerate(iters_vals):
        for j, n in enumerate(n_vals):
            if (int(n), int(iters)) in completed_runs:
                print(f"[ablation] Skipping computed state: n = {n:5d} | iters = {iters:4d}")
                continue
                
            print(f"[ablation] Evaluating max_candidates (n) = {n:5d} | max_iters = {iters:4d}")
            
            rewirer = SDRFRewiring(metric="bounds", max_iters=int(iters), max_candidates=int(n), n_jobs=-1)
            graph_copy = copy.deepcopy(data)
            
            t0 = time.perf_counter()
            _ = rewirer(graph_copy)
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            time_matrix[i, j] = elapsed
            
            results["runs"].append({
                "max_candidates": int(n),
                "max_iters": int(iters),
                "time_seconds": float(elapsed)
            })

            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)

    print(f"\n[ablation] Logged execution parameters and timing metrics to {out_file}")

    if auto_figures:
        _generate_ablation_plots(n_vals, iters_vals, time_matrix, out_dir)

def _generate_ablation_plots(n_vals, iters_vals, time_matrix, out_dir):
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })
    
    # Heatmap: Absolute Execution Time (T) across Domain (n x I)
    fig_heat, ax_heat = plt.subplots(figsize=(7, 5))
    
    vmin = max(time_matrix.min(), 1e-6)
    vmax = time_matrix.max()
    
    im = ax_heat.pcolormesh(
        n_vals, iters_vals, time_matrix, 
        shading='nearest', cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    
    ax_heat.set_xscale('log')
    ax_heat.set_yscale('log')
    ax_heat.set_xlabel(r'Max Candidates ($n$)')
    ax_heat.set_ylabel(r'Max Iterations')
    ax_heat.set_title('SDRF Runtime Map (Seconds)', pad=15)
    
    cbar = fig_heat.colorbar(im, ax=ax_heat)
    cbar.set_label('Absolute Execution Time (s) [Log Scale]')
    
    fig_heat.tight_layout()
    fig_heat.savefig(os.path.join(out_dir, "sdrf_ablation_heatmap.pdf"), format='pdf', bbox_inches='tight')
    fig_heat.savefig(os.path.join(out_dir, "sdrf_ablation_heatmap.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_heat)
    
    # Conditional expectations: Relative Overhead (Normalized)
    fig_cond, (ax_n, ax_iter) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Scaling vs n, conditioned on max_iters. 
    # Normalize each row by its execution time at the lowest n.
    eps = 1e-9
    norm_matrix_n = time_matrix / np.maximum(time_matrix[:, 0:1], eps)
    
    t_median_n = np.median(norm_matrix_n, axis=0)
    t_min_n = np.min(norm_matrix_n, axis=0)
    t_max_n = np.max(norm_matrix_n, axis=0)
    
    ax_n.plot(n_vals, t_median_n, marker='s', markersize=5, color='C0', label='Median Overhead Growth')
    ax_n.fill_between(n_vals, t_min_n, t_max_n, color='C0', alpha=0.2, label='Min-Max Deviation Bounds')
    
    ax_n.set_xscale('log')
    ax_n.set_yscale('log')
    ax_n.set_xlabel(r'Max Candidates ($n$)')
    ax_n.set_ylabel(r'Relative Time ($T_n / T_{n_0}$)')
    ax_n.set_title(r'Overhead Scaling vs. $n$', pad=15)
    ax_n.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
    ax_n.legend(loc='upper left')
    
    # Scaling vs max_iters, conditioned on n.
    # Normalize each column by its execution time at the lowest max_iters.
    norm_matrix_iter = time_matrix / np.maximum(time_matrix[0:1, :], eps)
    
    t_median_iter = np.median(norm_matrix_iter, axis=1)
    t_min_iter = np.min(norm_matrix_iter, axis=1)
    t_max_iter = np.max(norm_matrix_iter, axis=1)
    
    ax_iter.plot(iters_vals, t_median_iter, marker='o', markersize=5, color='C1', label='Median Overhead Growth')
    ax_iter.fill_between(iters_vals, t_min_iter, t_max_iter, color='C1', alpha=0.2, label='Min-Max Deviation Bounds')
    
    ax_iter.set_xscale('log')
    ax_iter.set_yscale('log')
    ax_iter.set_xlabel(r'Max Iterations')
    ax_iter.set_ylabel(r'Relative Time ($T_I / T_{I_0}$)')
    ax_iter.set_title(r'Overhead Scaling vs. Iterations', pad=15)
    ax_iter.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
    ax_iter.legend(loc='upper left')
    
    fig_cond.tight_layout()
    fig_cond.savefig(os.path.join(out_dir, "sdrf_ablation_relative_scaling.pdf"), format='pdf', bbox_inches='tight')
    fig_cond.savefig(os.path.join(out_dir, "sdrf_ablation_relative_scaling.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_cond)
    
    print("[ablation] Rendered parameter space heatmap and relative conditional scaling projections.")
