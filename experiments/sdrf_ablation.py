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
    
    # Construct log-spaced grids bounds
    n_vals = np.unique(np.logspace(0, np.log10(max(1, max_n_limit)), num=ablation_points, dtype=int))
    iters_vals = np.unique(np.logspace(0, np.log10(max(1, max_iters_limit)), num=ablation_points, dtype=int))
    
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sdrf_ablation_results.json")
    
    time_matrix = np.zeros((len(iters_vals), len(n_vals)))
    completed_runs = set()
    
    # Load state if checkpoint exists
    if os.path.isfile(out_file):
        print(f"[ablation] Discovered existing checkpoint. Soft-restarting from {out_file}...")
        with open(out_file, "r") as f:
            results = json.load(f)
            
        # Reconstruct the time matrix and identify completed grid points
        for run in results.get("runs", []):
            run_n = run["max_candidates"]
            run_iters = run["max_iters"]
            completed_runs.add((run_n, run_iters))
            
            # Map the historical run back to current matrix indices
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

            # Checkpoint partial progress immediately
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
    
    # Heatmap: Execution Time (T) across Domain (n x I)
    fig_heat, ax_heat = plt.subplots(figsize=(7, 5))
    im = ax_heat.pcolormesh(n_vals, iters_vals, time_matrix, shading='nearest', cmap='inferno')
    
    ax_heat.set_xscale('log')
    ax_heat.set_yscale('log')
    ax_heat.set_xlabel(r'Max Candidates ($n$)')
    ax_heat.set_ylabel(r'Max Iterations')
    ax_heat.set_title('SDRF Runtime Map (Seconds)', pad=15)
    
    cbar = fig_heat.colorbar(im, ax=ax_heat)
    cbar.set_label('Execution Time (s)')
    
    fig_heat.tight_layout()
    fig_heat.savefig(os.path.join(out_dir, "sdrf_ablation_heatmap.pdf"), format='pdf', bbox_inches='tight')
    fig_heat.savefig(os.path.join(out_dir, "sdrf_ablation_heatmap.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_heat)
    
    # Conditional expectation 1: Time = f(n) | max_iters
    fig_n, ax_n = plt.subplots(figsize=(6, 4.5))
    for i, iters in enumerate(iters_vals):
        ax_n.plot(n_vals, time_matrix[i, :], marker='s', markersize=5, label=f'iters={iters}')
        
    ax_n.set_xscale('log')
    ax_n.set_yscale('log')
    ax_n.set_xlabel(r'Max Candidates ($n$)')
    ax_n.set_ylabel(r'Execution Time (s)')
    ax_n.set_title(r'Time Scaling vs. Candidates conditioned on $I_{max}$', pad=15)
    ax_n.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
    ax_n.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(out_dir, "sdrf_ablation_time_vs_n.pdf"), format='pdf', bbox_inches='tight')
    fig_n.savefig(os.path.join(out_dir, "sdrf_ablation_time_vs_n.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_n)

    # Conditional expectation 2: Time = f(max_iters) | n
    fig_iter, ax_iter = plt.subplots(figsize=(6, 4.5))
    for j, n in enumerate(n_vals):
        ax_iter.plot(iters_vals, time_matrix[:, j], marker='o', markersize=5, label=f'n={n}')
        
    ax_iter.set_xscale('log')
    ax_iter.set_yscale('log')
    ax_iter.set_xlabel(r'Max Iterations')
    ax_iter.set_ylabel(r'Execution Time (s)')
    ax_iter.set_title(r'Time Scaling vs. Iterations conditioned on $n$', pad=15)
    ax_iter.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
    ax_iter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    fig_iter.tight_layout()
    fig_iter.savefig(os.path.join(out_dir, "sdrf_ablation_time_vs_iters.pdf"), format='pdf', bbox_inches='tight')
    fig_iter.savefig(os.path.join(out_dir, "sdrf_ablation_time_vs_iters.png"), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig_iter)
    
    print("[ablation] Rendered parameter space heatmap and conditional scaling projections.")
