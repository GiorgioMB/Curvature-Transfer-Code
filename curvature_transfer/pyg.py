"""PyTorch Geometric adapters (requires torch and torch_geometric).

The transforms replace ``data.edge_index`` with the rewired topology (both
directions, sorted by source then target) and set ``data.edge_attr`` to a
single float feature that is 1 for added edges and 0 for original ones. The
GNNs of the experiments use ``edge_dim=1`` and consume exactly this feature,
as in the preliminary code; any original edge features are dropped.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform

from . import rewiring
from .engine import CurvatureEngine
from .graph import Graph

__all__ = ["SDRFRewiring", "BORFRewiring", "FoSRRewiring", "engine_from_data"]


def engine_from_data(data, **kw) -> CurvatureEngine:
    return CurvatureEngine(Graph.from_edge_index(data.edge_index, int(data.num_nodes)), **kw)


def _write_back(data, original: np.ndarray, new: np.ndarray):
    device = data.edge_index.device
    n = int(data.num_nodes)
    key = lambda e: e[:, 0] * (n + 1) + e[:, 1]
    is_added = ~np.isin(key(new), key(original)) if len(new) else np.zeros(0, bool)
    src = np.concatenate([new[:, 0], new[:, 1]])
    dst = np.concatenate([new[:, 1], new[:, 0]])
    flag = np.concatenate([is_added, is_added]).astype(np.float32)
    order = np.lexsort((dst, src))
    data.edge_index = torch.from_numpy(np.stack([src[order], dst[order]])).long().to(device)
    data.edge_attr = torch.from_numpy(flag[order]).unsqueeze(1).to(device)
    return data


class _Rewire(BaseTransform):
    def forward(self, data):
        g = Graph.from_edge_index(data.edge_index, int(data.num_nodes))
        return _write_back(data, g.edges, self._rewire(g))

    def _rewire(self, g: Graph) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SDRFRewiring(_Rewire):
    def __init__(self, metric="bounds", max_iters=10, max_candidates=3, remove_edges=True, n_jobs=None, seed=None):
        super().__init__()
        self.metric, self.max_iters, self.max_candidates = metric, max_iters, max_candidates
        self.remove_edges, self.n_jobs, self.seed = remove_edges, n_jobs, seed

    def _rewire(self, g):
        import random
        rng = None if self.seed is None else random.Random(self.seed)
        return rewiring.sdrf(g.num_nodes, g.edges, self.metric, self.max_iters, self.max_candidates,
                             self.remove_edges, rng=rng, n_jobs=self.n_jobs)


class BORFRewiring(_Rewire):
    def __init__(self, metric="bounds", max_iters=3, batch_add=3, batch_remove=3, n_jobs=None):
        super().__init__()
        self.metric, self.max_iters = metric, max_iters
        self.batch_add, self.batch_remove, self.n_jobs = batch_add, batch_remove, n_jobs

    def _rewire(self, g):
        return rewiring.borf(g.num_nodes, g.edges, self.metric, self.max_iters,
                             self.batch_add, self.batch_remove, n_jobs=self.n_jobs)


class FoSRRewiring(_Rewire):
    def __init__(self, max_iters=10):
        super().__init__()
        self.max_iters = max_iters

    def _rewire(self, g):
        return rewiring.fosr(g.num_nodes, g.edges, self.max_iters)
