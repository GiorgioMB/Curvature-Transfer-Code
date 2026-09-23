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

def execute_ablation(seed, out_dir, max_iters, auto_figures=False):
    """
    Evaluates the runtime performance of SDRF rewiring as max_candidates (n)
    grows on the minesweeper dataset. Logs times and results to JSON, 
    and optionally generates a paper-ready log-log plot.
    """
    set_seed(seed)
    
    print(f"[ablation] Loading 'minesweeper' dataset...")
    dataset = HeterophilousGraphDataset(root="data/Heterophilous", name="minesweeper")
    data = dataset[0]
    num_nodes = data.num_nodes
    
    print(f"[ablation] Base Graph: |V| = {num_nodes}, |E| = {data.edge_index.shape[1] // 2}")
    
    # Preset definitions mapping to max_candidates (n)
    preset_configs = [
        ("tiny", max(num_nodes // 100, 50)),
        ("small", num_nodes // 4),
        ("medium", num_nodes // 2),
        ("paper", num_nodes - 2)
    ]
    
    results = {
        "dataset": "minesweeper",
        "num_nodes": num_nodes,
        "max_iters": max_iters,
        "runs": {}
    }
    
    x_vals, y_vals = [], []
    
    for preset_name, n in preset_configs:
        print(f"\n[ablation] Running configuration '{preset_name}' | max_candidates (n) = {n} | max_iters = {max_iters}")
        
        # Instantiate SDRF with specific candidate bounds
        rewirer = SDRFRewiring(metric="bounds", max_iters=max_iters, max_candidates=n, n_jobs=-1)
        
        # Deepcopy to prevent mutating original data
        graph_copy = copy.deepcopy(data)
        
        t0 = time.perf_counter()
        _ = rewirer(graph_copy)
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        print(f"[ablation] Completed '{preset_name}' in {elapsed:.4f} seconds.")
        
        results["runs"][preset_name] = {
            "max_candidates": n,
            "time_seconds": elapsed
        }
        x_vals.append(n)
        y_vals.append(elapsed)

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "sdrf_ablation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[ablation] Saved execution logs to {out_file}")

    if auto_figures:
        # Paper-ready styling configuration
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "font.family": "serif",
            "pdf.fonttype": 42, # TrueType rendering for easy Illustrator/Inkscape editing
            "ps.fonttype": 42
        })
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Plot series
        ax.plot(x_vals, y_vals, marker='s', markersize=8, color='#d62728', 
                markeredgecolor='black', markeredgewidth=1.2, linestyle='-', linewidth=2.0)
        
        # Log-Log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Labels and Title
        ax.set_xlabel(r'Max Candidates ($n$)')
        ax.set_ylabel(r'Execution Time (Seconds)')
        ax.set_title('SDRF Runtime Scaling by Candidate Bound', pad=15)
        
        # Grid layout (major + subtle minor grids)
        ax.grid(True, which="major", linestyle="-", alpha=0.3, color='gray')
        ax.grid(True, which="minor", linestyle="--", alpha=0.1, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Annotate points with dataset thresholds
        for i, (name, _) in enumerate(preset_configs):
            ax.annotate(f"{name}\n({x_vals[i]})", 
                        (x_vals[i], y_vals[i]), 
                        textcoords="offset points", 
                        xytext=(0, 12), 
                        ha='center', 
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9))
        
        plt.tight_layout()
        
        pdf_path = os.path.join(out_dir, "sdrf_ablation_scaling.pdf")
        png_path = os.path.join(out_dir, "sdrf_ablation_scaling.png")
        
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[ablation] Rendered paper-ready plot -> {pdf_path}")
