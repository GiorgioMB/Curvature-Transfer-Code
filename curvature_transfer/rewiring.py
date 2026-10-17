"""Curvature-based rewiring on plain edge arrays (no torch required).

The algorithms follow the preliminary code, so results stay comparable:

* SDRF (greedy variant, Topping et al. 2022): each iteration finds the edge
  with the lowest *addition score*, proposes support edges around it, scores
  each proposal by the *removal-type* score of that edge after the proposal is
  added, adds the best one, and removes the edge with the highest removal
  score (if both endpoints keep degree >= 1). Candidate scoring only needs the
  curvature of one edge, so it is evaluated on the 2-hop subgraph around it
  instead of on the whole graph.
* BORF (Nguyen et al. 2023): batches of additions around the lowest-scored
  edges (choosing the candidate with the largest common neighborhood) and
  batches of removals among the highest-scored edges.
* FoSR (Karhadkar et al. 2023), simplified: each iteration recomputes the
  Fiedler vector and adds the non-edge maximizing (x_u - x_v)^2.

Oracles (``metric``):
  "c_OR"       exact lazy OR curvature for both scores (BORF / OR-SDRF);
  "c_BF"       Balanced Forman curvature for both (SDRF);
  "bounds"     the BF interval (Corollary 5.6): addition by the upper end
               psi(c_BF), removal by the lower end Lambda(c_BF) (S-BORF/S-SDRF);
  "sharpened"  the sharpened interval (Proposition 5.7);
  "jl_bounds"  Jost--Liu: addition by tri/deg_max, removal by the lifted JL
               lower bound (JL-BORF).
"""
from __future__ import annotations

import random as _random
import time as _time
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np

from . import _io
from .engine import CurvatureEngine
from .graph import Graph

__all__ = ["METRICS", "scores", "local_edge_score", "sdrf", "borf", "fosr"]

METRICS = ("c_OR", "c_BF", "bounds", "sharpened", "jl_bounds")
Edge = Tuple[int, int]


def scores(graph: Graph, metric: str, n_jobs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(addition score, removal score) for every edge of ``graph``.
    Low addition score = bottleneck; high removal score = redundant edge."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    eng = CurvatureEngine(graph, n_jobs=n_jobs)
    if metric == "c_OR":
        c = eng.exact(("c_OR",))["c_OR"]
        return c, c
    if metric == "c_BF":
        c = eng.bf()
        return c, c
    b = eng.bounds()
    if metric == "bounds":
        return b["bf_upper"], b["bf_lower"]
    if metric == "sharpened":
        return b["sharpened_upper"], b["sharpened_lower"]
    return b["jl_upper"], b["jl_lower_lazy"]  # jl_bounds


def _local_graph(adj: Dict[int, Set[int]], u: int, v: int) -> Tuple[Graph, int]:
    """2-hop subgraph sufficient for every curvature of edge (u, v): all
    edges incident to S = N(u) u N(v). Returns (graph, edge id of (u, v))."""
    S = adj[u] | adj[v]
    loc: Dict[int, int] = {}
    pairs = []
    for x in S:
        for y in adj[x]:
            pairs.append((loc.setdefault(x, len(loc)), loc.setdefault(y, len(loc))))
    g = Graph.from_edges(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), len(loc))
    return g, g.edge_id(loc[u], loc[v])


def local_edge_score(adj: Dict[int, Set[int]], u: int, v: int, metric: str, kind: str = "remove") -> float:
    """Score of the single edge (u, v) in the graph given by ``adj``."""
    g, eid = _local_graph(adj, u, v)
    add, rem = scores(g, metric, n_jobs=1)
    return float((rem if kind == "remove" else add)[eid])


def _adjacency(num_nodes: int, edges: Iterable[Edge]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = {k: set() for k in range(num_nodes)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _canon(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _sdrf_candidates(adj, u, v) -> list:
    Nu, Nv = adj[u], adj[v]
    cand = set()
    for y in Nv:
        if y != u and y not in Nu:
            cand.add(_canon(u, y))
    for x in Nu:
        if x != v and x not in Nv:
            cand.add(_canon(v, x))
    for x in Nu:
        if x == v:
            continue
        for y in Nv:
            if y != u and y != x and y not in adj[x]:
                cand.add(_canon(x, y))
    return sorted(cand)


class _IterCheckpoint:
    """Per-iteration checkpoint of a rewiring loop.

    After every iteration the current edge set, the iteration counter, the RNG
    state, and the accumulated *compute* time (checkpoint I/O excluded) are
    written atomically to ``path``. A restarted call with the same inputs
    (checked through a fingerprint) continues from the last finished iteration
    and produces the same result as an uninterrupted run.
    """

    def __init__(self, path: Optional[str], key: str):
        self.path, self.key = path, key

    def load(self):
        if not self.path:
            return None
        st = _io.read_pickle(self.path)
        return st if isinstance(st, dict) and st.get("key") == self.key else None

    def save(self, it: int, current: Set[Edge], rng_state, elapsed: float):
        if self.path:
            _io.atomic_pickle({"key": self.key, "it": it, "edges": sorted(current),
                               "rng": rng_state, "elapsed": elapsed}, self.path)


def _start(kind, num_nodes, edges, params, checkpoint, rng=None):
    current: Set[Edge] = {_canon(int(a), int(b)) for a, b in np.asarray(edges).reshape(-1, 2) if a != b}
    key = (_io.fingerprint(kind, int(num_nodes), np.array(sorted(current), dtype=np.int64).reshape(-1, 2), params)
           if checkpoint else "")
    ck = _IterCheckpoint(checkpoint, key)
    st = ck.load()
    it0, elapsed = 0, 0.0
    if st is not None:
        current = {tuple(e) for e in st["edges"]}
        it0, elapsed = int(st["it"]), float(st["elapsed"])
        if rng is not None and st.get("rng") is not None:
            rng.setstate(st["rng"])
    return current, ck, it0, elapsed


def _random_non_edge(adj, n, rng, tries=1000):
    for _ in range(tries):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b and b not in adj[a]:
            return _canon(a, b)
    return None


def sdrf(num_nodes: int, edges: np.ndarray, metric: str = "bounds", max_iters: int = 10,
         max_candidates: int = 3, remove_edges: bool = True, rng=None,
         n_jobs: Optional[int] = None, checkpoint: Optional[str] = None,
         info: Optional[dict] = None) -> np.ndarray:
    """Greedy SDRF. Returns the rewired undirected edge array (u < v, sorted).

    ``rng``: a ``random.Random`` (default: the global ``random`` module, so
    that seeding with ``random.seed`` reproduces the preliminary code).
    ``checkpoint``: optional file; the loop state is saved there after every
    iteration and a restarted call resumes from it. ``info`` (a dict) receives
    ``compute_time`` (seconds, all iterations, including resumed ones)."""
    rng = _random if rng is None else rng
    current, ck, it0, elapsed = _start("sdrf", num_nodes, edges, [metric, max_iters, max_candidates, remove_edges],
                                       checkpoint, rng)
    for it in range(it0, max_iters):
        t0 = _time.perf_counter()
        if not current:
            break
        g = Graph.from_edges(np.array(sorted(current), dtype=np.int64), num_nodes)
        add_s, rem_s = scores(g, metric, n_jobs)
        e_min = tuple(int(x) for x in g.edges[int(np.argmin(add_s))])
        e_max = tuple(int(x) for x in g.edges[int(np.argmax(rem_s))])
        adj = _adjacency(num_nodes, current)
        cands = _sdrf_candidates(adj, *e_min)
        if not cands:
            ne = _random_non_edge(adj, num_nodes, rng)
            if ne is None:
                break
            cands = [ne]
        if len(cands) > max_candidates:
            cands = rng.sample(cands, max_candidates)
        best, best_score = None, -np.inf
        for c in cands:
            adj[c[0]].add(c[1]); adj[c[1]].add(c[0])
            sc = local_edge_score(adj, e_min[0], e_min[1], metric, "remove")
            adj[c[0]].discard(c[1]); adj[c[1]].discard(c[0])
            if sc > best_score:
                best, best_score = c, sc
        if best is not None:
            current.add(best)
        if remove_edges and len(current) > 1 and g.deg[e_max[0]] > 1 and g.deg[e_max[1]] > 1:
            current.discard(e_max)
        elapsed += _time.perf_counter() - t0
        ck.save(it + 1, current, rng.getstate(), elapsed)
    if info is not None:
        info["compute_time"] = elapsed
    return np.array(sorted(current), dtype=np.int64).reshape(-1, 2)


def borf(num_nodes: int, edges: np.ndarray, metric: str = "bounds", max_iters: int = 3,
         batch_add: int = 3, batch_remove: int = 3, n_jobs: Optional[int] = None,
         checkpoint: Optional[str] = None, info: Optional[dict] = None) -> np.ndarray:
    """Batched BORF-style rewiring. Returns the rewired undirected edge array.
    ``checkpoint`` / ``info``: as in :func:`sdrf`."""
    current, ck, it0, elapsed = _start("borf", num_nodes, edges, [metric, max_iters, batch_add, batch_remove], checkpoint)
    for it in range(it0, max_iters):
        t0 = _time.perf_counter()
        if not current:
            break
        g = Graph.from_edges(np.array(sorted(current), dtype=np.int64), num_nodes)
        add_s, rem_s = scores(g, metric, n_jobs)
        lowest = np.argsort(add_s, kind="stable")[: batch_add * 3]
        highest = np.argsort(-rem_s, kind="stable")
        adj = _adjacency(num_nodes, current)
        added: Set[Edge] = set()
        for idx in lowest:
            if len(added) >= batch_add:
                break
            u, v = (int(x) for x in g.edges[idx])
            best, best_overlap = None, -1
            for c in _sdrf_candidates(adj, u, v):
                ov = len(adj[c[0]] & adj[c[1]])
                if ov > best_overlap:
                    best, best_overlap = c, ov
            if best is not None and best not in current:
                added.add(best)
        current |= added
        deg = g.deg.copy()
        removed = 0
        for idx in highest:
            if removed >= batch_remove:
                break
            u, v = (int(x) for x in g.edges[idx])
            if deg[u] > 1 and deg[v] > 1 and (u, v) in current:
                current.discard((u, v))
                deg[u] -= 1; deg[v] -= 1
                removed += 1
        elapsed += _time.perf_counter() - t0
        ck.save(it + 1, current, None, elapsed)
    if info is not None:
        info["compute_time"] = elapsed
    return np.array(sorted(current), dtype=np.int64).reshape(-1, 2)


def fosr(num_nodes: int, edges: np.ndarray, max_iters: int = 10,
         checkpoint: Optional[str] = None, info: Optional[dict] = None) -> np.ndarray:
    """Spectral rewiring: add the non-edge maximizing (x_u - x_v)^2 for the
    current Fiedler vector x, ``max_iters`` times. ``checkpoint`` / ``info``:
    as in :func:`sdrf`."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    current, ck, it0, elapsed = _start("fosr", num_nodes, edges, [max_iters], checkpoint)
    n = num_nodes
    for it in range(it0, max_iters):
        t0 = _time.perf_counter()
        if not current or n < 3:
            break
        e = np.array(sorted(current), dtype=np.int64)
        A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
        try:
            vals, vecs = eigsh(L.astype(float), k=2, which="SA")
        except Exception:
            break
        x = vecs[:, np.argsort(vals)[1]]
        order = np.argsort(x)
        adj = _adjacency(n, current)
        added = None
        # scan pairs from the two ends of the spectrum outward
        for a in range(n):
            for b in range(n - 1, a, -1):
                u, v = int(order[a]), int(order[b])
                if v not in adj[u]:
                    added = _canon(u, v)
                    break
            if added:
                break
        if added is None:
            break
        current.add(added)
        elapsed += _time.perf_counter() - t0
        ck.save(it + 1, current, None, elapsed)
    if info is not None:
        info["compute_time"] = elapsed
    return np.array(sorted(current), dtype=np.int64).reshape(-1, 2)
