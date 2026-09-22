"""
Curvature-based Graph Rewiring Transforms

This module provides PyTorch Geometric compatible implementations of Stochastic 
Discrete Ricci Flow (SDRF) and Batched Ollivier-Ricci Flow (BORF). These 
implementations have been rewritten and heavily optimized from their original 
sources to integrate directly with our CurvatureEngine, allowing the algorithms 
to be driven by exact Ollivier-Ricci curvature, Balanced Forman curvature, or 
their theoretical transfer bounds.

References:
- SDRF: Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X., & Bronstein, 
  M. M. (2021). Understanding over-squashing and bottlenecks on graphs via 
  network topology and curvature. arXiv preprint arXiv:2111.14522.
- BORF: Nguyen, K., Nguyen, D., & Ho, N. (2022). Revisiting Over-smoothing and 
  Over-squashing Using Ollivier-Ricci Curvature. arXiv preprint arXiv:2211.15779.
"""

import os
import torch
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

import pyg_curvature as pc

def _fast_pyg_edge_index(edges: set, device: torch.device) -> torch.Tensor:
    """Otimized conversion from an edge set to a PyG-compatible directed edge_index."""
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    
    # Vectorized conversion rather than Python loops
    edges_arr = np.array(list(edges), dtype=np.int64)
    rows = np.concatenate([edges_arr[:, 0], edges_arr[:, 1]])
    cols = np.concatenate([edges_arr[:, 1], edges_arr[:, 0]])
    
    # Sort for canonical PyG format using numpy (torch.lexsort is unavailable in older PT)
    order = np.lexsort((cols, rows))
    
    edge_index = torch.from_numpy(np.stack([rows[order], cols[order]])).to(device)
    return edge_index

def _evaluate_metric_all(eng: pc.CurvatureEngine, metric: str, n_jobs: int = 1) -> np.ndarray:
    """Evaluates the requested metric for all edges."""
    M = len(eng.edges)
    if M == 0:
        return np.array([], dtype=float)

    if metric == "c_OR_lower_from_c_BF":
        bf_vals = _evaluate_metric_all(eng, "c_BF", n_jobs)
        return eng.varphi_BF_to_OR(bf_vals, sharp=False)
    if metric == "c_OR_upper_from_c_BF":
        bf_vals = _evaluate_metric_all(eng, "c_BF", n_jobs)
        return eng.psi_BF_to_OR(bf_vals)

    # Target isolated functions based on metric
    def _eval_func(i: int) -> float:
        if metric == "c_BF": return eng.c_BF_edge(i)
        if metric == "c_OR": return eng.c_OR_edge(i)
        raise ValueError(f"Unsupported metric identifier: '{metric}'. "
                         "Allowed: c_BF, c_OR, c_OR_lower_from_c_BF, c_OR_upper_from_c_BF")

    if n_jobs == 1:
        return np.array([_eval_func(i) for i in range(M)], dtype=float)
    else:
        workers = os.cpu_count() if n_jobs in (None, -1) else n_jobs
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return np.array(list(ex.map(_eval_func, range(M))), dtype=float)

def _evaluate_metric_single(eng: pc.CurvatureEngine, metric: str, eidx: int) -> float:
    """Evaluates a metric for a single edge."""
    if metric == "c_OR_lower_from_c_BF":
        return float(eng.varphi_BF_to_OR(eng.c_BF_edge(eidx), sharp=False)[eidx])
    if metric == "c_OR_upper_from_c_BF":
        return float(eng.psi_BF_to_OR(eng.c_BF_edge(eidx))[eidx])
        
    if metric == "c_BF": return eng.c_BF_edge(eidx)
    if metric == "c_OR": return eng.c_OR_edge(eidx)
    
    raise ValueError(f"Unsupported metric identifier: '{metric}'. "
                     "Allowed: c_BF, c_OR, c_OR_lower_from_c_BF, c_OR_upper_from_c_BF")


class SDRFRewiring(BaseTransform):
    """
    Stochastic Discrete Ricci Flow (SDRF) 
    (Optimized for CurvatureEngine)
    """
    def __init__(self, metric="c_OR_lower_from_c_BF", max_iters=10, 
                 max_candidates=3, remove_edges=True, n_jobs=None):
        self.metric = metric
        self.max_iters = max_iters
        self.max_candidates = max_candidates
        self.remove_edges = remove_edges
        self.n_jobs = n_jobs

    def __call__(self, data: Data) -> Data:
        device = data.edge_index.device
        num_nodes = data.num_nodes
        
        eng_init = pc.CurvatureEngine(data, n_jobs=self.n_jobs)
        current_edges = set(tuple(sorted(e)) for e in eng_init.edges)
        
        for _ in range(self.max_iters):
            if not current_edges:
                break
                
            edge_index = _fast_pyg_edge_index(current_edges, device)
            eng = pc.CurvatureEngine(Data(num_nodes=num_nodes, edge_index=edge_index), n_jobs=self.n_jobs)
            
            # Fast target evaluation
            vals = _evaluate_metric_all(eng, self.metric, n_jobs=self.n_jobs)
            edges_list = list(eng.edges)
            
            min_idx, max_idx = np.argmin(vals), np.argmax(vals)
            e_min, e_max = tuple(edges_list[min_idx]), tuple(edges_list[max_idx])
            
            u, v = e_min
            Nu, Nv = eng.neighbors[u], eng.neighbors[v]
            
            # Find candidate edges closing triangles or 4-cycles around e_min
            candidates = set()
            
            # Triangle candidates: connect u to N(v) or v to N(u)
            for y in Nv:
                if y != u and y not in Nu:
                    candidates.add(tuple(sorted((u, y))))
            for x in Nu:
                if x != v and x not in Nv:
                    candidates.add(tuple(sorted((v, x))))
                    
            # 4-cycle candidates: connect N(u) to N(v)
            for x in Nu:
                if x == v: continue
                for y in Nv:
                    if y == u or y == x: continue
                    if y not in eng.neighbors[x]:
                        candidates.add(tuple(sorted((x, y))))
            
            candidates = list(candidates)
            
            if not candidates:
                # Fallback: find all non-edges safely
                non_edges = [(a, b) for a in range(num_nodes) for b in range(a + 1, num_nodes) if b not in eng.neighbors[a]]
                if non_edges:
                    candidates = [random.choice(non_edges)]
                else:
                    break  # Graph is completely fully connected
            
            if len(candidates) > self.max_candidates:
                candidates = random.sample(candidates, self.max_candidates)
            
            best_candidate = None
            best_improvement = -float('inf')
            
            # Evaluate candidates isolated to e_min's local index
            for cand in candidates:
                cand_edges = current_edges | {cand}
                cand_idx = _fast_pyg_edge_index(cand_edges, device)
                cand_eng = pc.CurvatureEngine(Data(num_nodes=num_nodes, edge_index=cand_idx), n_jobs=1)
                
                try:
                    # Binary search equivalent list lookup (eng.edges is strictly sorted u < v)
                    new_idx = cand_eng.edges.index(e_min)
                    score = _evaluate_metric_single(cand_eng, self.metric, new_idx)
                except ValueError:
                    score = -float('inf')
                    
                if score > best_improvement:
                    best_improvement = score
                    best_candidate = cand
            
            if best_candidate:
                current_edges.add(best_candidate)
                
            if self.remove_edges and len(current_edges) > 1:
                # Discard only if we do not break strict connectivity assumptions
                if eng.deg[e_max[0]] > 1 and eng.deg[e_max[1]] > 1:
                    current_edges.discard(e_max)
                    
        data.edge_index = _fast_pyg_edge_index(current_edges, device)
        return data


class BORFRewiring(BaseTransform):
    """
    Batched Ollivier-Ricci Flow (BORF) 
    (Optimized for CurvatureEngine)
    """
    def __init__(self, metric="c_OR", max_iters=3, batch_add=3, batch_remove=3, n_jobs=None):
        self.metric = metric
        self.max_iters = max_iters
        self.batch_add = batch_add
        self.batch_remove = batch_remove
        self.n_jobs = n_jobs

    def __call__(self, data: Data) -> Data:
        device = data.edge_index.device
        num_nodes = data.num_nodes
        
        eng_init = pc.CurvatureEngine(data, n_jobs=self.n_jobs)
        current_edges = set(tuple(sorted(e)) for e in eng_init.edges)
        
        for _ in range(self.max_iters):
            if not current_edges:
                break
                
            edge_index = _fast_pyg_edge_index(current_edges, device)
            eng = pc.CurvatureEngine(Data(num_nodes=num_nodes, edge_index=edge_index), n_jobs=self.n_jobs)
            
            vals = _evaluate_metric_all(eng, self.metric, n_jobs=self.n_jobs)
            edges_list = eng.edges
            
            sorted_indices = np.argsort(vals)
            
            # Edge Additions (Heuristic common neighbor maximization)
            add_candidates = set()
            lowest_indices = sorted_indices[:self.batch_add * 3]  
            
            for idx in lowest_indices:
                if len(add_candidates) >= self.batch_add:
                    break
                    
                u, v = edges_list[idx]
                Nu, Nv = eng.neighbors[u], eng.neighbors[v]
                
                best_cand = None
                best_overlap = -1
                
                # Triangle candidates
                for y in Nv:
                    if y != u and y not in Nu:
                        overlap = len(eng.neighbors[u] & eng.neighbors[y])
                        if overlap > best_overlap:
                            best_overlap, best_cand = overlap, tuple(sorted((u, y)))
                for x in Nu:
                    if x != v and x not in Nv:
                        overlap = len(eng.neighbors[v] & eng.neighbors[x])
                        if overlap > best_overlap:
                            best_overlap, best_cand = overlap, tuple(sorted((v, x)))
                            
                # 4-cycle candidates
                for x in Nu:
                    if x == v: continue
                    for y in Nv:
                        if y == u or y == x: continue
                        if y not in eng.neighbors[x]:
                            overlap = len(eng.neighbors[x] & eng.neighbors[y])
                            if overlap > best_overlap:
                                best_overlap, best_cand = overlap, tuple(sorted((x, y)))
                
                if best_cand and best_cand not in current_edges:
                    add_candidates.add(best_cand)
            
            current_edges.update(add_candidates)
            
            # Edge Removals
            if self.batch_remove > 0:
                highest_indices = sorted_indices[::-1]
                removed_count = 0
                
                for idx in highest_indices:
                    if removed_count >= self.batch_remove:
                        break
                        
                    u, v = edges_list[idx]
                    if eng.deg[u] > 1 and eng.deg[v] > 1:
                        e_max = tuple(sorted((u, v)))
                        if e_max in current_edges:
                            current_edges.remove(e_max)
                            eng.deg[u] -= 1
                            eng.deg[v] -= 1
                            removed_count += 1
                            
        data.edge_index = _fast_pyg_edge_index(current_edges, device)
        return data
