"""Undirected simple graphs in CSR form.

Every curvature in this package is a function of an *undirected* edge, so the
graph is canonicalized once: self-loops and duplicate edges are removed, each
undirected edge is stored once as ``(u, v)`` with ``u < v`` (in lexicographic
order), and adjacency lists are sorted. All kernels read the CSR arrays
``indptr``/``indices`` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

__all__ = ["Graph"]


@dataclass(frozen=True)
class Graph:
    """Immutable undirected simple graph.

    Attributes
    ----------
    num_nodes : int
    edges : (M, 2) int64 array of undirected edges with ``u < v``, sorted.
    indptr, indices : CSR adjacency with sorted neighbor lists.
    deg : (num_nodes,) int64 degrees.
    """

    num_nodes: int
    edges: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    deg: np.ndarray

    # ------------------------------------------------------------------ build
    @classmethod
    def from_edges(cls, edges, num_nodes: Optional[int] = None) -> "Graph":
        """Build from any iterable/array of (u, v) pairs (directed or not)."""
        arr = np.asarray(list(edges) if not isinstance(edges, np.ndarray) else edges, dtype=np.int64)
        if arr.size == 0:
            arr = arr.reshape(0, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            if arr.ndim == 2 and arr.shape[0] == 2:
                arr = arr.T
            else:
                raise ValueError(f"edges must have shape (M, 2) or (2, M), got {arr.shape}")
        if arr.size and arr.min() < 0:
            raise ValueError("node ids must be nonnegative")
        n = int(arr.max()) + 1 if arr.size else 0
        if num_nodes is None:
            num_nodes = n
        elif num_nodes < n:
            raise ValueError(f"num_nodes={num_nodes} but edges reference node {n - 1}")
        u = np.minimum(arr[:, 0], arr[:, 1])
        v = np.maximum(arr[:, 0], arr[:, 1])
        keep = u != v
        und = np.unique(np.stack([u[keep], v[keep]], axis=1), axis=0) if keep.any() else np.empty((0, 2), np.int64)
        return cls._from_canonical(int(num_nodes), und)

    @classmethod
    def from_edge_index(cls, edge_index, num_nodes: Optional[int] = None) -> "Graph":
        """Build from a PyG-style ``(2, E)`` edge_index (torch tensor or array)."""
        ei = edge_index
        if hasattr(ei, "detach"):
            ei = ei.detach().cpu().numpy()
        ei = np.asarray(ei, dtype=np.int64)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        return cls.from_edges(ei.T, num_nodes)

    @classmethod
    def from_networkx(cls, G) -> Tuple["Graph", np.ndarray]:
        """Build from a networkx graph. Returns ``(graph, nodes)`` where
        ``nodes[k]`` is the original label of internal node ``k``."""
        nodes = np.array(list(G.nodes()), dtype=object)
        pos = {x: k for k, x in enumerate(nodes)}
        e = [(pos[a], pos[b]) for a, b in G.edges()]
        return cls.from_edges(np.array(e, dtype=np.int64).reshape(-1, 2), len(nodes)), nodes

    @classmethod
    def _from_canonical(cls, n: int, und: np.ndarray) -> "Graph":
        und = np.ascontiguousarray(und, dtype=np.int64)
        src = np.concatenate([und[:, 0], und[:, 1]])
        dst = np.concatenate([und[:, 1], und[:, 0]])
        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]
        deg = np.bincount(src, minlength=n).astype(np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(deg, out=indptr[1:])
        return cls(n, und, indptr, np.ascontiguousarray(dst, dtype=np.int64), deg)

    # --------------------------------------------------------------- queries
    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    def neighbors(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def has_edge(self, u: int, v: int) -> bool:
        nb = self.neighbors(u)
        k = np.searchsorted(nb, v)
        return bool(k < nb.size and nb[k] == v)

    def edge_id(self, u: int, v: int) -> int:
        """Index of undirected edge {u, v} in ``self.edges`` (or -1)."""
        a, b = (u, v) if u < v else (v, u)
        lo = np.searchsorted(self.edges[:, 0], a, "left")
        hi = np.searchsorted(self.edges[:, 0], a, "right")
        k = lo + np.searchsorted(self.edges[lo:hi, 1], b)
        return int(k) if k < hi and self.edges[k, 1] == b else -1

    def edge_ids(self, pairs: np.ndarray) -> np.ndarray:
        """Vectorized ``edge_id`` for an (K, 2) array; -1 where absent."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        a = np.minimum(pairs[:, 0], pairs[:, 1]); b = np.maximum(pairs[:, 0], pairs[:, 1])
        key_e = self.edges[:, 0] * (self.num_nodes + 1) + self.edges[:, 1]
        key = a * (self.num_nodes + 1) + b
        k = np.searchsorted(key_e, key)
        k = np.minimum(k, max(len(key_e) - 1, 0))
        ok = (len(key_e) > 0) & (key_e[k] == key) if len(key_e) else np.zeros(len(key), bool)
        return np.where(ok, k, -1)

    def directed_to_undirected(self, values, edge_index, agg: str = "mean") -> np.ndarray:
        """Aggregate per-directed-edge values onto ``self.edges`` order."""
        ei = edge_index.detach().cpu().numpy() if hasattr(edge_index, "detach") else np.asarray(edge_index)
        vals = np.asarray(values, dtype=float)
        ids = self.edge_ids(ei.T)
        ok = ids >= 0
        out = np.full(self.num_edges, np.nan)
        if agg == "mean":
            s = np.bincount(ids[ok], weights=vals[ok], minlength=self.num_edges)
            c = np.bincount(ids[ok], minlength=self.num_edges)
            out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        elif agg in ("min", "max"):
            fill = np.inf if agg == "min" else -np.inf
            out = np.full(self.num_edges, fill)
            (np.minimum if agg == "min" else np.maximum).at(out, ids[ok], vals[ok])
            out[~np.isfinite(out)] = np.nan
        elif agg == "first":
            first = {}
            for k in np.nonzero(ok)[0]:
                first.setdefault(int(ids[k]), vals[k])
            for e, x in first.items():
                out[e] = x
        else:
            raise ValueError(f"unknown agg={agg!r}")
        return out

    def to_edge_index(self) -> np.ndarray:
        """Directed (2, 2M) edge index, sorted by (source, target)."""
        src = np.repeat(np.arange(self.num_nodes, dtype=np.int64), self.deg)
        return np.stack([src, self.indices])

    def with_edges(self, add: Iterable = (), remove: Iterable = ()) -> "Graph":
        """Return a new graph with edges added/removed (both as (u, v) pairs)."""
        e = self.edges
        rem = np.asarray(list(remove), dtype=np.int64).reshape(-1, 2)
        if rem.size:
            ids = self.edge_ids(rem)
            mask = np.ones(len(e), bool); mask[ids[ids >= 0]] = False
            e = e[mask]
        addv = np.asarray(list(add), dtype=np.int64).reshape(-1, 2)
        if addv.size:
            e = np.concatenate([e, addv])
        return Graph.from_edges(e, self.num_nodes)
