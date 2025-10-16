"""
Utility helpers for computing and analyzing graph curvatures.

This module wraps the CurvatureEngine from pyg_curvature.py and provides:
- A light-weight Data container compatible with CurvatureEngine
- Convenience functions to compute curvatures and transfer bounds
- CSV/JSON export helpers
- Summary statistics over curvatures and envelopes

What you get
- Balanced Forman curvature (fast, combinatorial)
- Lazy and non-lazy Ollivier--Ricci curvature (exact, via optimal transport)
- Per-edge bounds that translate between BF and OR
- Per-edge envelopes that upper-bound lazy OR using only local structure

Quick example
-------------
>>> # Undirected triangle on nodes {0,1,2}
>>> num_nodes = 3
>>> edges_undirected = [(0,1), (1,2), (0,2)]
>>> result = analyze_graph(num_nodes, edges_undirected)
>>> # Detailed per-edge arrays:
>>> result.base.keys()
... dict_keys(['edges','deg_i','deg_j','triangle','Xi','sho_max','C4','c_BF','c_OR','c_OR0','Theta_Const','Theta_Slope'])
>>> # Transfer bounds:
>>> result.or_from_bf.keys(), result.bf_from_or.keys()
... (dict_keys(['c_BF','c_OR_lower_from_c_BF','c_OR_upper_from_c_BF']),
     dict_keys(['c_OR','c_BF_lower_from_c_OR','c_BF_upper_from_c_OR']))

Outputs
- write_edge_table(): saves a CSV with one row per undirected edge
- summarize_run(): returns JSON-serializable summary metrics
- analyze_graph(): high-level function to run everything and optionally write files
"""

import os
import math
import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# Import the curvature engine from the project root
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyg_curvature as pc


@dataclass
class CurvatureResult:
    """Container for all outputs from a curvature analysis run.

    Fields
    - base: raw per-edge arrays from CurvatureEngine.compute_all()
    - or_from_bf: dict with lower/upper bounds on OR given BF
    - bf_from_or: dict with lower/upper bounds on BF given OR
    - env_upper: per-edge upper envelope for lazy OR (from lazy_transport_envelope)
    - theta_at_t: linear envelope Theta evaluated at t = triangle count per edge
    """
    base: Dict[str, np.ndarray]
    # Transfer bounds
    or_from_bf: Dict[str, np.ndarray]
    bf_from_or: Dict[str, np.ndarray]
    # Envelopes per edge (lazy transport)
    env_upper: np.ndarray
    theta_at_t: np.ndarray


class Data:
    """Minimal PyG-like Data object with num_nodes and edge_index.

    This class mirrors the attributes that CurvatureEngine expects from
    torch_geometric.data.Data, so you can use these utilities without pulling
    in the full PyG dependency in small scripts.

    Attributes
    - num_nodes: number of graph nodes (int)
    - edge_index: torch.LongTensor of shape (2, E) for directed edges
                  (for each undirected (u, v) we include both (u, v) and (v, u))
    """
    def __init__(
            self, 
            num_nodes: int, 
            edge_index: torch.Tensor
            ):
        self.num_nodes = int(num_nodes)
        self.edge_index = edge_index


def make_edge_index(
        num_nodes: int, 
        edges_undirected: List[Tuple[int,int]]
        ) -> torch.Tensor:
    """
    Build a directed PyG-style edge_index from undirected pairs.

    Parameters
    - num_nodes: number of nodes (not used directly, included for compatibility)
    - edges_undirected: list of (u, v) with u and v as integer node ids

    Returns
    - torch.LongTensor of shape (2, 2M) where each undirected edge contributes
      two directed edges (u, v) and (v, u).
    """
    rows = []
    cols = []
    for u, v in edges_undirected:
        rows.extend([u, v])
        cols.extend([v, u])
    return torch.tensor([rows, cols], dtype=torch.long)


def compute_curvatures(
        num_nodes: int, 
        edges_undirected: List[Tuple[int,int]], 
        n_jobs=None
        ) -> CurvatureResult:
    """
    Compute curvatures, envelopes, and transfer bounds for a graph.
    
    Parameters
    - num_nodes : int Number of nodes in the graph
    - edges_undirected : List[Tuple[int,int]] List of undirected edges as (u,v) pairs
    - n_jobs : int or None, optional Number of parallel jobs. Follows scikit-learn conventions:
        - None (default): auto mode -- uses parallelism for graphs with >=512 edges
        - 1: sequential execution
        - -1: use all available CPUs
        - > 1: use exactly n_jobs CPUs
        - < -1: use (n_cpus + 1 + n_jobs) CPUs
        
    Returns
    - CurvatureResult object containing base curvatures, transfer bounds, and envelopes.
    """
    edge_index = make_edge_index(num_nodes, edges_undirected)
    eng = pc.CurvatureEngine(Data(num_nodes, edge_index), n_jobs=n_jobs)

    base = eng.compute_all()
    # Compute transfer bounds in both directions
    or_from_bf = eng.bounds_from_BF(base["c_BF"])
    bf_from_or = eng.bounds_from_OR(base["c_OR"])

    # Compute lazy transport envelope and linear Theta evaluated at t = triangles
    M = len(base["edges"])
    env_upper = np.zeros(M, dtype=float)
    for eidx in range(M):
        env = eng.lazy_transport_envelope(eidx)
        env_upper[eidx] = float(env["cOR_upper"])  # per-edge upper bound on c_OR

    theta_at_t = base["Theta_Const"] + base["Theta_Slope"] * base["triangle"]
    return CurvatureResult(base=base, or_from_bf=or_from_bf, bf_from_or=bf_from_or, env_upper=env_upper, theta_at_t=theta_at_t)


def summarize_distribution(
        arr: np.ndarray
        ) -> Dict[str, float]:
    """
    Summary statistics for a numeric array.

    Returns a dict with count, mean, std, min/max, and several quantiles
    useful for quick inspection and dashboards.
    """
    q = np.quantile(arr, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).tolist()
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "min": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "median": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "max": float(q[6]),
    }


def write_edge_table(
        path_csv: str, 
        curv: CurvatureResult
        ) -> None:
    """
    Write a per-edge CSV with base metrics, envelopes, and transfer bands.

    Columns
    - u, v: endpoints of the undirected edge with u < v (engine order)
    - deg_i, deg_j: endpoint degrees
    - triangle: number of common neighbors of (u, v)
    - Xi, sho_max, C4: 4-cycle related counts and the C4 correction term
    - c_BF, c_OR, c_OR0: Balanced Forman, lazy OR, non-lazy OR
    - Theta_const, Theta_slope, Theta_at_t: linear envelope params and value at t=triangle
    - env_upper: lazy transport upper envelope for c_OR
    - cOR_lower_from_cBF, cOR_upper_from_cBF: transfer bounds OR | BF
    - cBF_lower_from_cOR, cBF_upper_from_cOR: transfer bounds BF | OR
    """
    base = curv.base
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        # header describes all per-edge quantities for downstream analysis
        w.writerow(["u","v","deg_i","deg_j","triangle","Xi","sho_max","c_BF","c_OR","c_OR0","Theta_const","Theta_slope","Theta_at_t","env_upper",
                    "cOR_lower_from_cBF","cOR_upper_from_cBF","cBF_lower_from_cOR","cBF_upper_from_cOR"])
        for eidx, (u,v) in enumerate(base["edges"]):
            w.writerow([
                int(u), int(v),
                int(base["deg_i"][eidx]), int(base["deg_j"][eidx]),
                int(base["triangle"][eidx]), int(base["Xi"][eidx]), float(base["sho_max"][eidx]),
                float(base["c_BF"][eidx]), float(base["c_OR"][eidx]), float(base["c_OR0"][eidx]),
                float(base["Theta_Const"][eidx]), float(base["Theta_Slope"][eidx]), float(curv.theta_at_t[eidx]),
                float(curv.env_upper[eidx]),
                float(curv.or_from_bf["c_OR_lower_from_c_BF"][eidx]), float(curv.or_from_bf["c_OR_upper_from_c_BF"][eidx]),
                float(curv.bf_from_or["c_BF_lower_from_c_OR"][eidx]), float(curv.bf_from_or["c_BF_upper_from_c_OR"][eidx])
            ])


def summarize_run(
        curv: CurvatureResult
        ) -> Dict[str, Dict[str, float]]:
    """
    Summarize distributions and coverage of bounds/envelopes.

    Returns a nested dict with:
    - c_OR, c_BF: distribution summaries
    - env_slack: env_upper - c_OR distribution
    - theta_slack: Theta_at_t - c_OR distribution
    - triangle, Xi: structure distributions
    - coverage: fractions of edges where true curvatures lie within the bands
    - band_widths: average widths of transfer bands
    """
    base = curv.base
    out = {}
    out["c_OR"] = summarize_distribution(base["c_OR"])
    out["c_BF"] = summarize_distribution(base["c_BF"])
    # envelope slacks quantify how loose each envelope is on average
    env_slack = curv.env_upper - base["c_OR"]
    theta_slack = curv.theta_at_t - base["c_OR"]
    out["env_slack"] = summarize_distribution(env_slack)
    out["theta_slack"] = summarize_distribution(theta_slack)
    out["triangle"] = summarize_distribution(base["triangle"])
    out["Xi"] = summarize_distribution(base["Xi"])
    # coverage/fraction within bounds: 1.0 means the band fully covers the truth
    eps = 1e-9
    out["coverage"] = {
        "OR_in_BF_to_OR_band": float(np.mean((base["c_OR"] + eps >= curv.or_from_bf["c_OR_lower_from_c_BF"]) & (base["c_OR"] <= curv.or_from_bf["c_OR_upper_from_c_BF"] + eps))),
        "BF_in_OR_to_BF_band": float(np.mean((base["c_BF"] + eps >= curv.bf_from_or["c_BF_lower_from_c_OR"]) & (base["c_BF"] <= curv.bf_from_or["c_BF_upper_from_c_OR"] + eps))),
        "OR_below_env_upper": float(np.mean(base["c_OR"] <= curv.env_upper + eps)),
        "OR_below_theta": float(np.mean(base["c_OR"] <= curv.theta_at_t + eps)),
    }
    # widths of transfer bands give a sense of tightness
    out["band_widths"] = {
        "BF_to_OR_width_mean": float(np.mean(curv.or_from_bf["c_OR_upper_from_c_BF"] - curv.or_from_bf["c_OR_lower_from_c_BF"])),
        "OR_to_BF_width_mean": float(np.mean(curv.bf_from_or["c_BF_upper_from_c_OR"] - curv.bf_from_or["c_BF_lower_from_c_OR"])),
    }
    return out


def analyze_graph(
        num_nodes: int,
        edges_undirected: List[Tuple[int,int]],
        output_csv: str = None,
        output_summary: str = None,
        n_jobs=None) -> CurvatureResult:
    """
    Complete graph curvature analysis with optional file outputs.
    
    Parameters
    - num_nodes : int Number of nodes in the graph
    - edges_undirected : List[Tuple[int,int]] List of undirected edges as (u,v) pairs
    - output_csv : str, optional Path to write detailed per-edge results CSV
    - output_summary : str, optional Path to write summary statistics JSON
    - n_jobs : int or None, optional Number of parallel jobs (see CurvatureEngine for semantics)

    Returns
    - CurvatureResult object containing all computed curvature data

    Example
    >>> num_nodes = 4
    >>> edges_undirected = [(0,1),(1,2),(2,3),(3,0)]  # a 4-cycle
    >>> res = analyze_graph(num_nodes, edges_undirected,
    ...                    output_csv="edges.csv", output_summary="summary.json")
    """
    # Compute curvatures and bounds
    curv = compute_curvatures(num_nodes, edges_undirected, n_jobs=n_jobs)
    
    # Write outputs if requested
    if output_csv:
        write_edge_table(output_csv, curv)
        
    if output_summary:
        summary = summarize_run(curv)
        with open(output_summary, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return curv
