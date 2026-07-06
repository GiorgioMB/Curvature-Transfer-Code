#!/usr/bin/env python3
import os
import sys
import csv
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data
import random
import torch
import pyg_curvature as pc


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# Global reference for the worker processes to prevent PyTorch/PyG object pickling
_SHARED_ENGINE = None

def set_deterministic_seed(seed: int = 42):
    """Fixes the PRNG state for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _init_benchmark_worker(state_dict):
    """
    Initializes a lightweight engine shell in the worker process.
    """
    set_deterministic_seed(42)
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

def worker_bounds(chunk):
    t0 = time.perf_counter()
    for eidx in chunk:
        _SHARED_ENGINE.c_BF_edge(eidx)
        _SHARED_ENGINE.lazy_transport_envelope(eidx)
    return time.perf_counter() - t0

def run_benchmark(
    output_csv="data/benchmark_runtime_scaling.csv", 
    max_workers=None,
    min_n=50,
    max_n=20000,
    num_graphs=100
):
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    completed_n = set()
    expected_cols = ["n", "p", "num_edges", "avg_deg", "ot_time_sec", "bounds_time_sec"]

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
            writer.writerow(["n", "p", "num_edges", "avg_deg", "ot_time_sec", "bounds_time_sec"])

    raw_n_list = np.geomspace(min_n, max_n, num=num_graphs, dtype=int)
    n_list = sorted(np.unique(raw_n_list).tolist())

    for n in n_list:
        if n in completed_n:
            continue
        
        # Dynamically scale p to ensure average degree grows at a rate of O(n^0.4)
        p = float(0.2 * (50 / n) ** 0.6)
        print(f"--- Benchmarking n={n}, p={p} ---")
        
        # O(n^2 * p) allocation; processed exclusively in the master thread
        edge_index = erdos_renyi_graph(num_nodes=n, edge_prob=p, directed=False)
        data = Data(num_nodes=n, edge_index=edge_index)
        
        # Single master engine derivation
        eng_main = pc.CurvatureEngine(data, n_jobs=1)
        M = len(eng_main.edges)
        avg_deg = np.mean(eng_main.deg)
        print(f"Structure: {M} undirected edges, Average Degree: {avg_deg:.1f}")
        
        # Extract minimalist serialized state to distribute across the pool
        worker_state = eng_main._build_worker_state()
        
        # Granular workload distribution
        chunks = np.array_split(range(M), max_workers * 8)
        chunks = [c.tolist() for c in chunks if len(c) > 0]

        # Benchmark exact OT
        t_ot_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_benchmark_worker, initargs=(worker_state,)) as ex:
            list(ex.map(worker_ot, chunks))
        t_ot_total = time.perf_counter() - t_ot_start
        print(f"Exact OT Time: {t_ot_total:.4f}s")
        
        # Benchmark Combinatorial Bounds Stack
        t_bounds_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_benchmark_worker, initargs=(worker_state,)) as ex:
            list(ex.map(worker_bounds, chunks))
        t_bounds_total = time.perf_counter() - t_bounds_start
        print(f"Bounds Time: {t_bounds_total:.4f}s")
        
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([n, p, M, f"{avg_deg:.2f}", f"{t_ot_total:.4f}", f"{t_bounds_total:.4f}"])

if __name__ == "__main__":
    set_deterministic_seed(42) ##Modify this and line 36 to use another seed
    run_benchmark()
