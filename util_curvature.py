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
    base: Dict[str, np.ndarray]
    # Transfer bounds
    or_from_bf: Dict[str, np.ndarray]
    bf_from_or: Dict[str, np.ndarray]
    # Envelopes per edge (lazy transport)
    env_upper: np.ndarray
    theta_at_t: np.ndarray


class Data:
    """Minimal PyG-like Data with num_nodes and edge_index."""
    def __init__(self, num_nodes: int, edge_index: torch.Tensor):
        self.num_nodes = int(num_nodes)
        self.edge_index = edge_index


def make_edge_index(num_nodes: int, edges_undirected: List[Tuple[int,int]]) -> torch.Tensor:
    rows = []
    cols = []
    for u, v in edges_undirected:
        rows.extend([u, v])
        cols.extend([v, u])
    return torch.tensor([rows, cols], dtype=torch.long)


def compute_curvatures(num_nodes: int, edges_undirected: List[Tuple[int,int]]) -> CurvatureResult:
    edge_index = make_edge_index(num_nodes, edges_undirected)
    eng = pc.CurvatureEngine(Data(num_nodes, edge_index))

    base = eng.compute_all()

    # transfer bounds
    or_from_bf = eng.bounds_from_BF()
    bf_from_or = eng.bounds_from_OR()

    # lazy transport envelope and Theta_alpha(triangle)
    M = len(base["edges"])
    env_upper = np.zeros(M, dtype=float)
    for eidx in range(M):
        env = eng.lazy_transport_envelope(eidx)
        env_upper[eidx] = float(env["cOR_upper"])

    theta_at_t = base["Theta_Const"] + base["Theta_Slope"] * base["triangle"]
    return CurvatureResult(base=base, or_from_bf=or_from_bf, bf_from_or=bf_from_or, env_upper=env_upper, theta_at_t=theta_at_t)


def summarize_distribution(arr: np.ndarray) -> Dict[str, float]:
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


def write_edge_table(path_csv: str, curv: CurvatureResult) -> None:
    base = curv.base
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
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


def summarize_run(curv: CurvatureResult) -> Dict[str, Dict[str, float]]:
    base = curv.base
    out = {}
    out["c_OR"] = summarize_distribution(base["c_OR"])
    out["c_BF"] = summarize_distribution(base["c_BF"])
    # envelope slacks
    env_slack = curv.env_upper - base["c_OR"]
    theta_slack = curv.theta_at_t - base["c_OR"]
    out["env_slack"] = summarize_distribution(env_slack)
    out["theta_slack"] = summarize_distribution(theta_slack)
    out["triangle"] = summarize_distribution(base["triangle"])
    out["Xi"] = summarize_distribution(base["Xi"])
    # coverage/fraction within bounds
    eps = 1e-9
    out["coverage"] = {
        "OR_in_BF_to_OR_band": float(np.mean((base["c_OR"] + eps >= curv.or_from_bf["c_OR_lower_from_c_BF"]) & (base["c_OR"] <= curv.or_from_bf["c_OR_upper_from_c_BF"] + eps))),
        "BF_in_OR_to_BF_band": float(np.mean((base["c_BF"] + eps >= curv.bf_from_or["c_BF_lower_from_c_OR"]) & (base["c_BF"] <= curv.bf_from_or["c_BF_upper_from_c_OR"] + eps))),
        "OR_below_env_upper": float(np.mean(base["c_OR"] <= curv.env_upper + eps)),
        "OR_below_theta": float(np.mean(base["c_OR"] <= curv.theta_at_t + eps)),
    }
    # widths of transfer bands
    out["band_widths"] = {
        "BF_to_OR_width_mean": float(np.mean(curv.or_from_bf["c_OR_upper_from_c_BF"] - curv.or_from_bf["c_OR_lower_from_c_BF"])),
        "OR_to_BF_width_mean": float(np.mean(curv.bf_from_or["c_BF_upper_from_c_OR"] - curv.bf_from_or["c_BF_lower_from_c_OR"])),
    }
    return out