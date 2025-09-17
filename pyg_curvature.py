# -*- coding: utf-8 -*-
"""
pyg_curvature.py

Balanced Forman (c_BF) and Ollivier-Ricci (c_OR) curvatures for edges of an
undirected, simple graph stored as a PyTorch Geometric Data object, together
with the envelopes/transfer moduli developed in the provided notes:

- Prop. (Lazy -> Non-Lazy transfer): Eq. (3.1 -- master-beta in the latex) and (3.2 -- sharper-piecewise)
- Prop. (Lazy transport envelope): Eq. (3.7 -- OR-master-corrected-sharp)
- Thm. (BF -> OR Lower Transfer): Eq. (4.2 -- psi-BF-to-OR-lazy)
- Thm. (BF -> OR Upper Transfer): Eq. (4.8 -- psi-upper-final) via Psi_alpha
- Prop. (Monotone coverage envelope Theta_alpha): Eq. (4.12 -- Theta-alpha-def)
- Thm. (OR -> BF Lower Transfer): Eq. (4.14 -- OR-to-BF-lower)
- Thm. (OR -> BF Upper Transfer): Eq. (4.15 -- psi-OR-to-BF-lazy)

Conventions used:
- Graph: simple, undirected, connected with unit-length edges.
- Lazy measures: alpha_u = 1/(deg(u)+1); neighbor mass w_u^(alpha) = 1/(deg(u)+1).
- Distances are graph shortest-path lengths (unweighted). On the supports used
  here they are in {0,1,2,3}. We compute them exactly by BFS limited to depth 3.

Implementation details
----------------------
1) We deduplicate undirected edges (store u<v). All adjacency is kept as
   Python lists/sets for fast neighborhood operations.
2) The Wasserstein-1 distance between the (small) discrete supports is solved
   exactly as a transportation problem (Earth Mover's Distance) via a
   transportation-simplex (MODI) method. Supplies/demands are uniform, but the
   code is general.
3) All quantities are per-edge. Vectorized helpers return arrays aligned to the
   deduplicated undirected edge list.
4) Per-edge evaluation of the transfer moduli and envelopes when the threshold
   (zeta or theta) is itself the observed curvature value at that edge. This
   provides analytic image bounds c_OR(e) in terms of c_BF(e) and vice versa:
    -   bounds_from_BF()  -> lower/upper bounds on c_OR given c_BF
    -   bounds_from_OR()  -> lower/upper bounds on c_BF given c_OR
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set, Any, Deque, Iterator, cast

import math
import numpy as np
import numpy.typing as npt

try:
    import torch
    from torch_geometric.data import Data
except Exception as exc:
    raise RuntimeError("This module requires torch and torch_geometric.") from exc


# ------------------------------
# Utility helpers
# ------------------------------

def _as_undirected_unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a (2, M) LongTensor of undirected edges with u<v, deduplicated."""
    assert edge_index.shape[0] == 2 and edge_index.dtype == torch.long
    ei = edge_index.detach().cpu()
    u = ei[0].numpy().astype(np.int64)
    v = ei[1].numpy().astype(np.int64)
    mask = u < v
    u1 = u[mask]
    v1 = v[mask]
    mask2 = u > v
    u2 = v[mask2]
    v2 = u[mask2]
    u = np.concatenate([u1, u2], axis=0)
    v = np.concatenate([v1, v2], axis=0)
    if u.size == 0:
        return torch.empty((2,0), dtype=torch.long)
    pairs = np.stack([u, v], axis=1)
    pairs = np.unique(pairs, axis=0)
    return torch.from_numpy(pairs.T).long()


def _build_neighbors(num_nodes: int, undirected_edges: torch.Tensor) -> List[Set[int]]:
    """Adjacency sets for each node from undirected, deduplicated edges (u<v)."""
    neigh = [set() for _ in range(num_nodes)]
    u = undirected_edges[0].tolist()
    v = undirected_edges[1].tolist()
    for a,b in zip(u, v):
        neigh[a].add(b)
        neigh[b].add(a)
    return neigh


def _bfs_dist_limited(source: int, targets: Set[int], neighbors: List[Set[int]], max_depth: int = 3) -> Dict[int, int]:
    """
    BFS from source up to max_depth; return distances to targets discovered.
    Distances on the supports we use are always <= 3 (support of W1 is at most 3), but
    this routine is correct in general for the specified max_depth.
    """
    if source in targets:
        return {source: 0}
    if max_depth <= 0:
        return {}
    found: Dict[int, int] = {}
    visited = {source}
    frontier = [source]
    dist = 0
    while frontier and dist < max_depth:
        dist += 1
        nxt = []
        for x in frontier:
            for y in neighbors[x]:
                if y in visited:
                    continue
                visited.add(y)
                if y in targets and y not in found:
                    found[y] = dist
                nxt.append(y)
        frontier = nxt
        if len(found) == len(targets):
            break
    return found


def _pairwise_distances_between_sets(A: Sequence[int], B: Sequence[int], neighbors: List[Set[int]], default_far: int = 10) -> np.ndarray:
    """Compute pairwise shortest-path distances between A and B by BFS with horizon 3.

    Returns:
        C: np.ndarray of shape (len(A), len(B)), with integer distances.
           Any undetected pair gets default_far (should not happen for our supports).
    """
    setB = set(B)
    m, n = len(A), len(B)
    C = np.full((m, n), default_far, dtype=float)
    idxB = {b:j for j,b in enumerate(B)}
    for i, a in enumerate(A):
        dd = _bfs_dist_limited(a, setB, neighbors, max_depth=3)
        # Fill distances
        for b, d in dd.items():
            C[i, idxB[b]] = float(d)
        # Also fill trivial cases if missing
        if a in idxB:
            C[i, idxB[a]] = 0.0
    return C


# ------------------------------
# Transportation (EMD / W1) solver on small supports
# ------------------------------

def _northwest_corner(supply: np.ndarray, demand: np.ndarray) -> Tuple[np.ndarray, Set[Tuple[int,int]]]:
    """Initial basic feasible solution via Northwest Corner with degeneracy handling.

    Returns:
        X: (m,n) flows
        basis: set of (i,j) positions considered basic (includes zero-flows to keep m+n-1).
    """
    S = supply.copy()
    D = demand.copy()
    m, n = len(S), len(D)
    X = np.zeros((m, n), dtype=float)
    basis: Set[Tuple[int,int]] = set()

    i = 0
    j = 0
    tol = 1e-15
    while i < m and j < n:
        x = min(S[i], D[j])
        X[i, j] += x
        basis.add((i, j))
        S[i] -= x
        D[j] -= x
        if S[i] <= tol and D[j] <= tol:
            # Both exhausted: advance one dimension and add a degenerate basis to keep count.
            S[i] = 0.0
            D[j] = 0.0
            if i < m-1 and j < n-1:
                # Add a degenerate basic variable at (i, j+1) with zero flow
                basis.add((i, j+1))
                j += 1
            elif j < n-1:
                j += 1
            elif i < m-1:
                i += 1
            else:
                break
        elif S[i] <= tol:
            S[i] = 0.0
            i += 1
        elif D[j] <= tol:
            D[j] = 0.0
            j += 1
        else:
            # Should not happen; x exhausted one side
            pass

    # Ensure we have at least m+n-1 basic positions
    # Add supplemental zero-flow bases if needed by scanning grid
    need = (m + n - 1) - len(basis)
    if need > 0:
        for ii in range(m):
            if need == 0: break
            for jj in range(n):
                if (ii, jj) not in basis:
                    basis.add((ii, jj))
                    need -= 1
                    if need == 0: break

    return X, basis


def _compute_potentials(cost: np.ndarray, basis: Set[Tuple[int,int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute u (rows) and v (cols) potentials solving u_i + v_j = c_ij for (i,j) in basis.

    We fix u_0 = 0 and propagate.
    """
    m, n = cost.shape
    u = np.full(m, np.nan, dtype=float)
    v = np.full(n, np.nan, dtype=float)
    u[0] = 0.0

    rows_to_cols = {i: [] for i in range(m)}
    cols_to_rows = {j: [] for j in range(n)}
    for (i,j) in basis:
        rows_to_cols[i].append(j)
        cols_to_rows[j].append(i)

    from collections import deque
    q = deque()
    q.append(('r', 0))
    seen_r = set([0])
    seen_c = set()
    while q:
        typ, idx = q.popleft()
        if typ == 'r':
            i = idx
            for j in rows_to_cols[i]:
                if math.isnan(v[j]):
                    v[j] = cost[i, j] - u[i]
                if j not in seen_c:
                    seen_c.add(j)
                    q.append(('c', j))
        else:
            j = idx
            for i in cols_to_rows[j]:
                if math.isnan(u[i]):
                    u[i] = cost[i, j] - v[j]
                if i not in seen_r:
                    seen_r.add(i)
                    q.append(('r', i))

    u[np.isnan(u)] = 0.0
    v[np.isnan(v)] = 0.0
    return u, v


def _find_cycle(basis: Set[Tuple[int,int]], enter: Tuple[int,int], m: int, n: int) -> List[Tuple[int,int,int]]:
    """Find the unique alternating cycle created by adding 'enter' into the basis.

    Returns a list of (i,j,sign) with +1 for entering cell and alternating +/- 1 along the cycle.
    """
    from collections import deque

    (i0, j0) = enter
    rows_to_cols: Dict[int, Set[int]] = {i: set() for i in range(m)}
    cols_to_rows: Dict[int, Set[int]] = {j: set() for j in range(n)}
    for (i,j) in basis:
        rows_to_cols[i].add(j)
        cols_to_rows[j].add(i)

    start = ('r', i0)
    goal = ('c', j0)
    q: Deque[Any] = deque([start])  # allow mixed node types
    parent: Dict[Any, Optional[Any]] = {start: None}
    visited: Set[Any] = {start}

    def neighbors(node):
        typ, idx = node
        if typ == 'r':
            for jj in rows_to_cols[idx]:
                node2 = ('c', jj)
                if node2 not in visited:
                    yield node2
        else:
            for ii in cols_to_rows[idx]:
                node2 = ('r', ii)
                if node2 not in visited:
                    yield node2

    found = False
    while q:
        node = q.popleft()
        if node == goal:
            found = True
            break
        for nb in neighbors(node):
            visited.add(nb)
            parent[nb] = node
            q.append(nb)

    if not found:
        row_cols = [j for (i,j) in basis if i == i0 and j != j0]
        col_rows = [i for (i,j) in basis if j == j0 and i != i0]
        if row_cols and col_rows:
            j1 = row_cols[0]
            i1 = col_rows[0]
            cycle = [(i0,j0, +1), (i0,j1, -1), (i1,j1, +1), (i1,j0, -1)]
            return cycle
        raise RuntimeError("Failed to find cycle for MODI. Basis may be inconsistent.")

    nodes: List[Any] = []
    cur: Optional[Any] = goal
    while cur is not None:
        nodes.append(cur)
        cur = parent.get(cur, None)
    nodes = list(reversed(nodes))

    cycle: List[Tuple[int,int,int]] = [(i0, j0, +1)]
    for k in range(len(nodes) - 1):
        a = nodes[k]; b = nodes[k+1]
        if a[0] == 'r' and b[0] == 'c':
            i = a[1]; j = b[1]
        elif a[0] == 'c' and b[0] == 'r':
            i = b[1]; j = a[1]
        else:
            continue
        sign = -1 if (k % 2 == 0) else +1
        cycle.append((i, j, sign))

    return cycle


def _transportation_simplex(cost: np.ndarray, supply: np.ndarray, demand: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> Tuple[np.ndarray, float]:
    """Solve the balanced transportation problem exactly (up to tol) via MODI.

    Minimize sum_{i,j} cost[i,j] * X[i,j]  subject to row/col sums = supply/demand.

    Returns:
        X: optimal flow matrix
        value: minimal cost
    """
    m, n = cost.shape
    assert supply.shape == (m,) and demand.shape == (n,)
    if not np.isclose(supply.sum(), demand.sum(), atol=1e-9):
        raise ValueError("Supply and demand must sum to the same total.")

    # Initial BFS via NW corner
    X, basis = _northwest_corner(supply, demand)

    for it in range(max_iter):
        # Potentials
        u, v = _compute_potentials(cost, basis)

        # Reduced costs
        rc = np.full_like(cost, np.nan, dtype=float)
        for i in range(m):
            for j in range(n):
                if (i,j) not in basis:
                    rc[i,j] = cost[i,j] - (u[i] + v[j])

        # Optimality check
        min_rc = np.nanmin(rc)
        if min_rc >= -tol or np.isnan(min_rc):
            # Optimal
            break

        # Entering cell: most negative reduced cost
        enter_idx = cast(Tuple[int,int], np.unravel_index(np.nanargmin(rc), rc.shape))  # type: ignore[arg-type]
        # Find cycle
        cycle = _find_cycle(basis, enter_idx, m, n)

        # Determine theta (minimum flow on '-' cells)
        theta = math.inf
        minus_cells = []
        for (i,j,sgn) in cycle:
            if sgn < 0:
                minus_cells.append((i,j))
                theta = min(theta, X[i,j])
        if not np.isfinite(theta):
            theta = 0.0

        # Update flows
        for (i,j,sgn) in cycle:
            X[i,j] = X[i,j] + theta if sgn > 0 else X[i,j] - theta
            if abs(X[i,j]) < tol:
                X[i,j] = 0.0

        # Update basis: add entering cell; remove one zero-flow among minus cells if needed
        basis.add(enter_idx)
        zeros = [(i,j) for (i,j) in minus_cells if X[i,j] <= tol]
        # Keep basis size roughly m+n-1 by removing one zero that does not break feasibility
        if zeros:
            # Prefer removing a zero that is not the entering cell (it never is among minus)
            basis.remove(zeros[0])

    value = float((X * cost).sum())
    return X, value


def wasserstein1_uniform(cost: np.ndarray, n_left: int, n_right: int) -> float:
    """Compute W1 for uniform marginals on supports of sizes n_left and n_right.

    Returns the minimal expected cost for transporting from the uniform on left
    to the uniform on right given pairwise costs.
    """
    m, n = cost.shape
    assert m == n_left and n == n_right
    if m == 0 or n == 0:
        return 0.0
    supply = np.full(m, 1.0/m, dtype=float)
    demand = np.full(n, 1.0/n, dtype=float)
    _, val = _transportation_simplex(cost, supply, demand)
    return float(val)


# ------------------------------
# Core engine
# ------------------------------

@dataclass
class EdgeLocal:
    i: int
    j: int
    deg_i: int
    deg_j: int
    Ni: Set[int]
    Nj: Set[int]
    C: Set[int]             # common neighbors
    Ui: Set[int]            # unique neighbors of i
    Uj: Set[int]            # unique neighbors of j
    tri: int                # triangle(i,j) = |C|
    Xi: int                 # Xi_ij
    varpi_max: int          # varpi_max(i,j)
    sho_max: int            # sho_max(i,j) = varpi_max * max{deg_i,deg_j}


class CurvatureEngine:
    """Compute curvatures and envelopes on a PyG graph."""

    def __init__(self, data: "Data"):
        if not hasattr(data, "num_nodes") or not hasattr(data, "edge_index"):
            raise ValueError("Expected a PyG Data object with num_nodes and edge_index.")
        if data.num_nodes is None or data.edge_index is None:
            raise ValueError("Data object has None for num_nodes or edge_index.")
        self.num_nodes = int(data.num_nodes)
        self._original_edge_index = data.edge_index
        self._cache: Dict[str, Any] = {}
        # Deduplicate to undirected unique edges (u<v)
        self.undirected_edges = _as_undirected_unique_edges(data.edge_index, self.num_nodes)
        self.neighbors = _build_neighbors(self.num_nodes, self.undirected_edges)
        self.deg = np.array([len(self.neighbors[u]) for u in range(self.num_nodes)], dtype=int)
        # Pre-build edge list
        self.edges: List[Tuple[int,int]] = list(zip(self.undirected_edges[0].tolist(), self.undirected_edges[1].tolist()))
        self._TAU = 1e-12  # numerical tolerance for premise checks

    def _as_edgewise(self, x: npt.ArrayLike, name: str) -> np.ndarray:
        """Coerce a scalar or 1-D array to a length-M float array."""
        M = len(self.edges)
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.full(M, float(arr), dtype=float)
        if arr.ndim == 1 and arr.shape[0] == M:
            return arr
        raise ValueError(f"{name} must be scalar or length-{M} vector, got shape {arr.shape}.")

    # ---------- Neighborhood-derived primitives ----------

    def _local_for_edge(self, idx: int) -> EdgeLocal:
        i, j = self.edges[idx]
        Ni = self.neighbors[i]
        Nj = self.neighbors[j]
        Ci = Ni & Nj
        Ui = Ni - (Nj | {j})
        Uj = Nj - (Ni | {i})
        tri = len(Ci)

        # Xi_ij
        Xi_left  = sum(len(self.neighbors[k] & Uj) for k in Ui)
        Xi_right = sum(len(self.neighbors[w] & Ui) for w in Uj)
        Xi = Xi_left + Xi_right


        # varpi_max
        setNj_noi = Nj - {i}
        setNi_noj = Ni - {j}
        varpi1 = 0
        for k in setNi_noj:
            varpi1 = max(varpi1, len(self.neighbors[k] & setNj_noi))
        varpi2 = 0
        for w in setNj_noi:
            varpi2 = max(varpi2, len(self.neighbors[w] & setNi_noj))
        varpi_max = max(varpi1, varpi2)

        deg_i = self.deg[i]
        deg_j = self.deg[j]
        sho_max = max(1, len(Ui) * len(Uj))


        return EdgeLocal(i=i, j=j, deg_i=deg_i, deg_j=deg_j, Ni=Ni, Nj=Nj, C=Ci, Ui=Ui, Uj=Uj,
                         tri=tri, Xi=Xi, varpi_max=varpi_max, sho_max=sho_max)

    # ---------- Distances ----------
    def _values_to_undirected(
        self,
        values: np.ndarray,
        edge_index: Optional[torch.Tensor] = None,
        agg: str = "mean",
    ) -> np.ndarray:
        """
        Map per-directed-edge values onto the engine's undirected u<v edge order.
        If `values` is already aligned to undirected edges (len == len(self.edges)), it
        is returned unchanged.

        Parameters
        ----------
        values : (E,) or (M,) array
            If length == number of columns in `edge_index` (directed), the function
            aggregates the two orientations to a single undirected edge using `agg`.
            If length == len(self.edges) (undirected), it is returned as-is.
        edge_index : torch.LongTensor (2, E), optional
            Directed edges. If None, uses the original edge_index provided to the engine.
        agg : {"mean","sum","min","max","first"}
            Aggregation to collapse (u->v, v->u) into (min(u,v), max(u,v)).

        Returns
        -------
        np.ndarray of shape (M,)
            Values aligned to the engine's undirected edges order (u<v).
        """
        M = len(self.edges)
        if values.shape[0] == M:
            return values  # already undirected order

        if edge_index is None:
            edge_index = self._original_edge_index
        if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be a (2, E) LongTensor for directed edges.")

        E = edge_index.shape[1]
        if values.shape[0] != E:
            raise ValueError(f"values has length {values.shape[0]} but directed edge_index has {E} columns.")

        # Build buckets for undirected pairs
        u = edge_index[0].detach().cpu().numpy().astype(np.int64)
        v = edge_index[1].detach().cpu().numpy().astype(np.int64)
        buckets: Dict[Tuple[int,int], List[float]] = {}
        for k in range(E):
            a = int(u[k]); b = int(v[k])
            if a == b:
                # ignore self-loops if present; they don't appear in undirected u<v list
                continue
            p = (a, b) if a < b else (b, a)
            buckets.setdefault(p, []).append(float(values[k]))

        # Aggregator
        def _agg(xs: List[float]) -> float:
            if not xs:
                return float("nan")
            if agg == "mean":
                return float(sum(xs) / len(xs))
            elif agg == "sum":
                return float(sum(xs))
            elif agg == "min":
                return float(min(xs))
            elif agg == "max":
                return float(max(xs))
            elif agg == "first":
                return float(xs[0])
            else:
                raise ValueError(f"Unknown agg='{agg}'")

        # Emit in engine's undirected order
        out = np.zeros(M, dtype=float)
        for idx, (i, j) in enumerate(self.edges):
            xs = buckets.get((i, j), [])
            out[idx] = _agg(xs)
        return out

    def _cost_matrix_lazy_supports(self, i: int, j: int) -> Tuple[np.ndarray, List[int], List[int]]:
        """Cost matrix between supports of lazy measures m_i, m_j: {i} U N(i) vs {j} U N(j)."""
        left = [i] + sorted(self.neighbors[i])
        right = [j] + sorted(self.neighbors[j])
        C = _pairwise_distances_between_sets(left, right, self.neighbors, default_far=10)
        return C, left, right

    def _cost_matrix_nonlazy_supports(self, i: int, j: int) -> Tuple[np.ndarray, List[int], List[int]]:
        """Cost matrix between non-lazy neighbor laws nu_i and nu_j: N(i) vs N(j)."""
        left = sorted(self.neighbors[i])
        right = sorted(self.neighbors[j])
        C = _pairwise_distances_between_sets(left, right, self.neighbors, default_far=10)
        return C, left, right

    # ---------- Curvatures ----------

    @staticmethod
    def _C4_edge(Xi: int, sho_max: int) -> float:
        if Xi <= 0 or sho_max <= 0:
            return 0.0
        return float(Xi) / float(sho_max)

    def c_BF_edge(self, idx: int) -> float:
        """Balanced Forman curvature on edge idx with the convention c_BF=0 if min degree == 1."""
        loc = self._local_for_edge(idx)
        a, b = loc.deg_i, loc.deg_j
        if min(a, b) == 1:
            return 0.0
        tri = loc.tri
        Xi = loc.Xi
        C4 = self._C4_edge(Xi, loc.sho_max)
        val = (2.0/a + 2.0/b - 2.0
               + 2.0*tri/max(a,b)
               + tri/min(a,b)
               + C4)
        return float(val)

    def c_OR_edge(self, idx: int) -> float:
        """Lazy Ollivier-Ricci curvature on edge idx with alpha_u = 1/(deg(u)+1)."""
        i, j = self.edges[idx]
        C, left, right = self._cost_matrix_lazy_supports(i, j)
        W1 = wasserstein1_uniform(C, len(left), len(right))
        # dist_G(i,j) = 1 for edges
        return float(1.0 - W1)

    def c_OR0_edge(self, idx: int) -> float:
        """Non-lazy Ollivier-Ricci curvature using neighbor laws nu_i, nu_j (no Dirac masses)."""
        i, j = self.edges[idx]
        C, left, right = self._cost_matrix_nonlazy_supports(i, j)
        if len(left) == 0 and len(right) == 0:
            return 1.0  # deg-zero (should not happen on connected graphs)
        if len(left) == 0 or len(right) == 0:
            # If one side has no neighbors, treat W1 on empty vs non-empty as 1-step move
            # but for simple connected graphs this does not happen
            return 0.0
        W1 = wasserstein1_uniform(C, len(left), len(right))
        # Normalization by dist(i,j) is immaterial for edges (distance=1)
        return float(1.0 - W1)

    def _get_c_OR0_all(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute (and cache) non-lazy OR curvatures for all undirected edges.
        Use this to avoid re-running the transport for sign sharpening repeatedly.

        Parameters
        ----------
        force_recompute : bool
            If True, recompute even if a cached array exists.

        Returns
        -------
        np.ndarray of shape (M,)
            c_OR0 for each undirected edge in self.edges order (u<v).
        """
        key = "_c_OR0_all"
        if (not force_recompute) and (key in self._cache):
            return self._cache[key]
        arr = np.array([self.c_OR0_edge(eidx) for eidx in range(len(self.edges))], dtype=float)
        self._cache[key] = arr
        return arr
    # ---------- Edgewise parameters (Def. edgewise comparison parameters) ----------

    @staticmethod
    def _S(i_deg: int, j_deg: int) -> float:
        return 2.0/i_deg + 2.0/j_deg - 2.0

    @staticmethod
    def _T(i_deg: int, j_deg: int) -> float:
        return 2.0/max(i_deg, j_deg) + 1.0/min(i_deg, j_deg)

    @staticmethod
    def _K(i_deg: int, j_deg: int) -> float:
        return 1.0 - 1.0/min(i_deg, j_deg) - 1.0/max(i_deg, j_deg)

    @staticmethod
    def _Zmax(tri: int, i_deg: int, j_deg: int) -> float:
        return tri / float(max(i_deg, j_deg))

    @staticmethod
    def _Zmin(tri: int, i_deg: int, j_deg: int) -> float:
        return tri / float(min(i_deg, j_deg))

    # ---------- Lazy parameters for envelopes ----------

    @staticmethod
    def _alpha(deg: int) -> float:
        # alpha = 1 / (deg + 1)
        return 1.0 / (deg + 1.0)

    @staticmethod
    def _w_alpha(deg: int) -> float:
        # w^(alpha) = (1-alpha)/deg = 1/(deg+1)
        return 1.0 / (deg + 1.0)

    @staticmethod
    def _Sigma_alpha(deg_i: int, deg_j: int) -> float:
        # Sigma_alpha = rho_i/(1-alpha_i) + rho_j/(1-alpha_j) = (rho_i+1) + (rho_j+1)
        return float(deg_i + deg_j + 2)

    @staticmethod
    def _z_i_j(deg_i: int, deg_j: int) -> Tuple[float, float]:
        # z_i = min{ alpha_i, (1-alpha_j)/rho_j } = min{ 1/(rho_i+1), 1/(rho_j+1) } = 1/(rho_max+1)
        a_i = 1.0/(deg_i+1.0)
        a_j = 1.0/(deg_j+1.0)
        z_i = min(a_i, a_j)
        z_j = min(a_i, a_j)
        return z_i, z_j

    @staticmethod
    def _r_terms(deg_i: int, deg_j: int) -> Tuple[float, float, float, float]:
        ai = 1.0/(deg_i+1.0); aj = 1.0/(deg_j+1.0)
        bj = 1.0/(deg_j+1.0); bi = 1.0/(deg_i+1.0)
        ri = max(0.0, ai - bj)
        rj = max(0.0, aj - bi)
        bri = max(0.0, bj - ai)
        brj = max(0.0, bi - aj)
        return ri, bri, rj, brj

    # ---------- Proposition: Lazy transport envelope (upper bound for c_OR) ----------

    def lazy_transport_envelope(self, idx: int) -> Dict[str, float]:
        """Compute an explicit upper bound from Prop. 'Lazy Transport Envelope' (Eq. OR-master-corrected-sharp).

        This uses the sharp per-edge Xi_ij and triangle(i,j) when available to form
        the tightest bound per that proposition. The slacks are chosen by the
        upper envelopes in Eqs. (mUU-envelope-sharp) and (mTriangle-envelope-sharp).
        """
        loc = self._local_for_edge(idx)
        i_deg, j_deg = loc.deg_i, loc.deg_j
        tri = loc.tri
        Xi = loc.Xi

        ai = self._alpha(i_deg); aj = self._alpha(j_deg)
        wi = self._w_alpha(i_deg); wj = self._w_alpha(j_deg)
        w_wedge = min(wi, wj)
        Sigma = self._Sigma_alpha(i_deg, j_deg)
        zi, zj = self._z_i_j(i_deg, j_deg)
        ri, bri, rj, brj = self._r_terms(i_deg, j_deg)
        UU_cap = min((i_deg-1-tri)*wi, (j_deg-1-tri)*wj)
        UU_cov = Xi / Sigma if Sigma > 0 else 0.0
        mUU = max(0.0, min(UU_cap, UU_cov))

        mTri = min(tri*abs(wi-wj), (i_deg-1-tri)*wi + (j_deg-1-tri)*wj)

        cOR_upper = (-1.0 + 2.0*(zi+zj) + (ri+bri+rj+brj) + 2.0*tri*w_wedge + mUU + mTri)
        return {
            "cOR_upper": float(cOR_upper),
            "z_i": zi, "z_j": zj,
            "r_i": ri, "rbar_i": bri, "r_j": rj, "rbar_j": brj,
            "mUU": mUU, "mTri": mTri,
            "wi": wi, "wj": wj, "Sigma": Sigma,
        }

    # ---------- Proposition: Monotone coverage envelope Theta_alpha (upper bound affine in triangles) ----------

    @staticmethod
    def _const_slope(deg_i: int, deg_j: int) -> Tuple[float, float]:
        Sigma = float(deg_i + deg_j + 2)
        ai = 1.0/(deg_i+1.0); aj = 1.0/(deg_j+1.0)
        wi = 1.0/(deg_i+1.0); wj = 1.0/(deg_j+1.0)
        # z_i=z_j=min(ai,aj)
        z_i = min(ai, aj); z_j = z_i
        # r-terms
        diff = abs(ai - aj)
        endpoint_residual = 2.0 * diff
        Const = -1.0 + 2.0*(z_i+z_j) + endpoint_residual + (deg_i + deg_j - 2.0)/Sigma
        Slope = wi + wj - 2.0/Sigma
        # Slope is strictly positive as shown in the note
        return Const, Slope

    def Theta_alpha(self, idx: int, t: Optional[float] = None) -> Tuple[float, float, float]:
        """Return Theta_alpha(t) = Const + Slope * t, and (Const,Slope). If t is None, replace t by triangle(i,j)."""
        loc = self._local_for_edge(idx)
        Const, Slope = self._const_slope(loc.deg_i, loc.deg_j)
        if t is None:
            t = float(loc.tri)
        return Const + Slope * t, Const, Slope

    # ---------- Theorem: BF -> OR Lower transfer modulus ----------

    def varphi_BF_to_OR(self, zeta) -> np.ndarray:
        """Edgewise lower bound on c_OR from c_BF >= zeta (Theorem BF->OR lower).

        Returns an array of shape (num_edges,)."""
        zetas = self._as_edgewise(zeta, "zeta")
        M = len(self.edges)
        out = np.zeros(M, dtype=float)
        for eidx in range(M):
            loc = self._local_for_edge(eidx)
            S = self._S(loc.deg_i, loc.deg_j)
            T = self._T(loc.deg_i, loc.deg_j)
            K = self._K(loc.deg_i, loc.deg_j)
            C4 = self._C4_edge(loc.Xi, loc.sho_max)
            # Zeta forcing lower bound on triangles:
            #Zscr = max(0.0, (zeta - S - C4) / T) if T > 0 else 0.0
            #Zbar_max = Zscr / float(max(loc.deg_i, loc.deg_j)) if max(loc.deg_i, loc.deg_j) > 0 else 0.0
            #Zbar_min = Zscr / float(min(loc.deg_i, loc.deg_j)) if min(loc.deg_i, loc.deg_j) > 0 else 0.0
            tri_bar = loc.tri
            Zbar_max = tri_bar / float(max(loc.deg_i, loc.deg_j)) if max(loc.deg_i, loc.deg_j) > 0 else 0.0
            Zbar_min = tri_bar / float(min(loc.deg_i, loc.deg_j)) if min(loc.deg_i, loc.deg_j) > 0 else 0.0
            # Jost-Liu lower bound g(i,j) with triangles replaced by Zscr
            g = -max(0.0, K - Zbar_max) - max(0.0, K - Zbar_min) + Zbar_max
            # Lazy modulus: choose alpha_* according to sign of the non-lazy curvature
            cOR0 = self.c_OR0_edge(eidx)
            ai = self._alpha(loc.deg_i); aj = self._alpha(loc.deg_j)
            a_min, a_max = min(ai, aj), max(ai, aj)
            Delta = abs(ai - aj)
            a_star = a_min if cOR0 >= 0.0 else a_max
            out[eidx] = (1.0 - a_star) * g - Delta
        return out

    # ---------- Theorem: BF -> OR Upper transfer modulus (psi via Psi_alpha) ----------

    def psi_BF_to_OR(self, zeta) -> np.ndarray:
        """
        Edgewise upper bound on c_OR from c_BF <= zeta (Theorem BF->OR upper).
        Accepts either:
            - scalar zeta: same threshold on all edges, or
            - array-like zeta: per-edge thresholds; if aligned to directed edges,
              it will be aggregated to undirected order via `agg='mean'`.
        """
        # Normalize zeta -> length-M vector in undirected edge order
        arr = np.asarray(zeta, dtype=float)
        if arr.ndim == 0:
            zetas = self._as_edgewise(float(arr), "zeta")
        else:
            zetas = self._values_to_undirected(
                np.ravel(arr), edge_index=self._original_edge_index, agg="mean"
            )
        out = []
        for eidx in range(len(self.edges)):
            zeta_e = float(zetas[eidx])
            loc = self._local_for_edge(eidx)
            i_deg, j_deg = loc.deg_i, loc.deg_j
            tri = loc.tri

            S = self._S(i_deg, j_deg)
            T = self._T(i_deg, j_deg)
            # Budget b(zeta)
            b = max(0.0, zeta_e - S)

            # Ingredients
            Sigma = self._Sigma_alpha(i_deg, j_deg)
            wi = self._w_alpha(i_deg); wj = self._w_alpha(j_deg)
            w_wedge = min(wi, wj)

            # sho_max^* and eta_alpha
            rho_max = float(max(i_deg, j_deg))
            sho_max_star = rho_max * (rho_max - 1.0)
            eta_alpha = sho_max_star / Sigma if Sigma > 0 else 0.0

            def A_u(t: float, deg_u: int, w_u: float) -> float:
                return (deg_u - 1.0 - t) * w_u

            def A_min(t: float) -> float:
                return min(A_u(t, i_deg, wi), A_u(t, j_deg, wj))

            def B_alpha(t: float) -> float:
                return eta_alpha * (b - T * t)

            def C_alpha(t: float) -> float:
                return min(t * abs(wi - wj), A_u(t, i_deg, wi) + A_u(t, j_deg, wj))

            # The envelope Psi_alpha(t)
            zi, zj = self._z_i_j(i_deg, j_deg)
            ri, bri, rj, brj = self._r_terms(i_deg, j_deg)
            def Psi(t: float) -> float:
                return (-1.0 + 2.0*(zi+zj) + (ri+bri+rj+brj) + 2.0*t*w_wedge
                        + max(0.0, min(A_min(t), B_alpha(t))) + C_alpha(t))

            # Triangle feasible range
            t_max = min(i_deg, j_deg) - 1.0
            if T > 0:
                t_max = min(t_max, b / T)  # budget ceiling

            # Candidate breakpoints
            cand = [0.0]

            # Intersections t at which A_i or A_j equals B_alpha
            denom_i = (eta_alpha * T - wi)
            denom_j = (eta_alpha * T - wj)
            if abs(denom_i) > 1e-15:
                ti = (eta_alpha * b - wi * (i_deg - 1.0)) / denom_i
                cand.append(ti)
            if abs(denom_j) > 1e-15:
                tj = (eta_alpha * b - wj * (j_deg - 1.0)) / denom_j
                cand.append(tj)

            # Swap point where A_i = A_j (if wi != wj)
            if abs(wi - wj) > 1e-15:
                t_swap = (wj * (j_deg - 1.0) - wi * (i_deg - 1.0)) / (wj - wi)
                cand.append(t_swap)

            # Demand-supply intersection for C_alpha
            denom = wi + wj + abs(wi - wj)
            if denom > 1e-15:
                t_scap = (wi * (i_deg - 1.0) + wj * (j_deg - 1.0)) / denom
                cand.append(t_scap)

            cand.append(t_max)

            # Clamp candidates and evaluate
            best = -math.inf
            for t in cand:
                if not np.isfinite(t):
                    continue
                t_clamped = max(0.0, min(t_max, t))
                best = max(best, Psi(t_clamped))

            out.append(best)
        return np.array(out, dtype=float)

    # ---------- Theorem: OR -> BF lower transfer modulus ----------

    def varphi_OR_to_BF(self, theta, robust: bool = False) -> np.ndarray:
        """
        Edgewise *lower* bound on c_BF from c_OR >= theta (Theorem OR->BF lower).
        Accepts either:
            - scalar theta: same threshold on all edges, or
            - array-like theta: per-edge thresholds; if aligned to directed edges,
              it will be aggregated to undirected order via `agg='mean'`.
        If `robust=False` (default): paper-literal inversion using Theta_alpha.
        If `robust=True`: apply premise checks identical to the previous
            `varphi_OR_to_BF_per_edge` (leaf guard + envelope check, with fallback).
        """
        arr = np.asarray(theta, dtype=float)
        if arr.ndim == 0:
            thetas = self._as_edgewise(float(arr), "theta")
        else:
            thetas = self._values_to_undirected(
                np.ravel(arr), edge_index=self._original_edge_index, agg="mean"
            )
        out = []
        for eidx in range(len(self.edges)):
            theta_e = float(thetas[eidx])
            loc = self._local_for_edge(eidx)
            i_deg, j_deg = loc.deg_i, loc.deg_j
            S = self._S(i_deg, j_deg)
            T = self._T(i_deg, j_deg)
            if not robust:
                Const, Slope = self._const_slope(i_deg, j_deg)
                t_min = 0.0 if Slope <= 0 else max(0.0, (theta_e - Const) / Slope)
                t_min = min(t_min, min(i_deg, j_deg) - 1.0)
                out.append(S + T * t_min)
            else:
                # Robust variant: identical to previous per-edge function
                tri = float(loc.tri)
                if min(i_deg, j_deg) <= 1:
                    out.append(0.0); continue
                Const, Slope = self._const_slope(i_deg, j_deg)
                envelope_ok = (Slope > self._TAU) and (theta_e <= Const + Slope * tri + self._TAU)
                if not envelope_ok:
                    out.append(S + T * tri); continue
                t_min = max(0.0, (theta_e - Const) / Slope)
                t_min = min(t_min, min(i_deg, j_deg) - 1.0)
                t_min = min(t_min, tri)
                out.append(S + T * t_min)            
        return np.array(out, dtype=float)

    # ---------- Theorem: OR -> BF upper transfer modulus ----------
    @staticmethod
    def _g_nonlazy(di: int, dj: int, t: float) -> float:
        rho_min = float(min(di, dj))
        rho_max = float(max(di, dj))
        K = max(0.0, 1.0 - 1.0/rho_min - 1.0/rho_max)
        zmax = t / rho_max
        zmin = t / rho_min
        return -max(0.0, K - zmax) - max(0.0, K - zmin) + zmax

    def _u_max_from_s0(self, di: int, dj: int, s0: float, tol: float = 1e-12) -> float:
        # Bracket feasible triangles
        t_lo = 0.0
        t_hi = float(min(di, dj) - 1)
        # Quick exits
        g_lo = self._g_nonlazy(di, dj, t_lo)      # = -2K
        if s0 <= g_lo + tol:
            return 0.0
        g_hi = self._g_nonlazy(di, dj, t_hi)
        if s0 >= g_hi - tol:
            return t_hi
        # Monotone bisection: largest t with g(t) <= s0
        for _ in range(50):
            t_mid = 0.5 * (t_lo + t_hi)
            if self._g_nonlazy(di, dj, t_mid) <= s0 + tol:
                t_lo = t_mid
            else:
                t_hi = t_mid
        return t_lo


    def psi_OR_to_BF(self, theta, use_sign_sharpening: bool = True) -> np.ndarray:
        """Edgewise upper bound on c_BF from c_OR <= theta (Theorem OR->BF upper).

        If use_sign_sharpening is True, we compute c_OR-0 to choose alpha_* piecewise,
        otherwise we use the sign-agnostic beta = alpha_min variant.
        """
        thetas = self._as_edgewise(theta, "theta")
        M = len(self.edges)
        out = np.zeros(M, dtype=float)
        for eidx in range(M):
            theta_e = float(thetas[eidx])
            loc = self._local_for_edge(eidx)
            i_deg, j_deg = loc.deg_i, loc.deg_j
            S = self._S(i_deg, j_deg)
            T = self._T(i_deg, j_deg)
            K = max(0.0, self._K(i_deg, j_deg))  # [K]_+
            # Choose alpha_* as in Prop. (sharper) depending on sign of c_OR-0
            ai = self._alpha(i_deg); aj = self._alpha(j_deg)
            Delta = abs(ai - aj)
            if use_sign_sharpening:
                cOR0 = self.c_OR0_edge(eidx)
                a_star = min(ai, aj) if cOR0 >= 0.0 else max(ai, aj)
            else:
                a_star = min(ai, aj)  # sign-agnostic (safe)
            # s0 proxy
            denom = (1.0 - a_star)
            s0 = float('inf') if denom <= 0 else (theta_e + Delta) / denom

            u_max = self._u_max_from_s0(i_deg, j_deg, s0)

            C4 = self._C4_edge(loc.Xi, loc.sho_max)
            out[eidx] = S + T * u_max + C4
        return out

    def bounds_from_BF(self, c_BF: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Given (or computing) per-edge c_BF, return per-edge lower/upper bounds on c_OR.
        """
        if c_BF is None:
            base = self.compute_all()
            c_BF = base["c_BF"]
        else:
            c_BF = self._values_to_undirected(c_BF, edge_index=self._original_edge_index, agg="mean")
        lower = self.varphi_BF_to_OR(c_BF)
        upper = self.psi_BF_to_OR(c_BF)
        return {"c_BF": c_BF, "c_OR_lower_from_c_BF": lower, "c_OR_upper_from_c_BF": upper}

    def bounds_from_OR(self, c_OR: Optional[np.ndarray] = None, use_sign_sharpening: bool = True, reuse_cOR0: bool = True) -> Dict[str, np.ndarray]:
        """
        Given (or computing) per-edge c_OR, return per-edge lower/upper bounds on c_BF.
        """
        if c_OR is None:
            base = self.compute_all()
            c_OR = base["c_OR"]
            c_OR0 = self._get_c_OR0_all() if reuse_cOR0 else None
        else:
            c_OR = self._values_to_undirected(c_OR, edge_index=self._original_edge_index, agg="mean")
            c_OR0 = self._get_c_OR0_all() if (use_sign_sharpening and reuse_cOR0) else None
        lower = self.varphi_OR_to_BF(c_OR)
        upper = self.psi_OR_to_BF(c_OR, use_sign_sharpening=use_sign_sharpening)
        return {"c_OR": c_OR, "c_BF_lower_from_c_OR": lower, "c_BF_upper_from_c_OR": upper}

 

    def compute_all(self) -> Dict[str, np.ndarray]:
        """Compute base curvatures and per-edge structural terms.

        Returns a dict with fields:
            - 'edges': (M,2) array of undirected edges (u,v) with u<v
            - 'deg_i', 'deg_j': degrees at endpoints
            - 'triangle': |N(i)∩N(j)|
            - 'Xi': Xi_ij
            - 'sho_max': sho_max(i,j) = varpi_max(i,j)*max{deg_i,deg_j}
            - 'C4': per-edge 4-cycle coefficient Xi/sho_max (0 if Xi=0)
            - 'c_BF': Balanced Forman curvature
            - 'c_OR': Lazy Ollivier-Ricci curvature
            - 'c_OR0': Non-lazy curvature (neighbors only)
            - 'Theta_Const', 'Theta_Slope'
        """
        M = len(self.edges)
        deg_i = np.zeros(M, dtype=float)
        deg_j = np.zeros(M, dtype=float)
        tri = np.zeros(M, dtype=float)
        Xi = np.zeros(M, dtype=float)
        sho = np.zeros(M, dtype=float)
        C4 = np.zeros(M, dtype=float)
        cBF = np.zeros(M, dtype=float)
        cOR = np.zeros(M, dtype=float)
        cOR0 = np.zeros(M, dtype=float)
        ThC = np.zeros(M, dtype=float)
        ThS = np.zeros(M, dtype=float)

        for eidx in range(M):
            loc = self._local_for_edge(eidx)
            deg_i[eidx] = loc.deg_i
            deg_j[eidx] = loc.deg_j
            tri[eidx] = loc.tri
            Xi[eidx] = loc.Xi
            sho[eidx] = loc.sho_max
            C4[eidx] = self._C4_edge(loc.Xi, loc.sho_max)
            cBF[eidx] = self.c_BF_edge(eidx)
            cOR[eidx] = self.c_OR_edge(eidx)
            cOR0[eidx] = self.c_OR0_edge(eidx)
            _, cst, slp = self.Theta_alpha(eidx, t=None)
            ThC[eidx] = cst
            ThS[eidx] = slp

        return {
            "edges": np.array(self.edges, dtype=int),
            "deg_i": deg_i, "deg_j": deg_j,
            "triangle": tri,
            "Xi": Xi,
            "sho_max": sho,
            "C4": C4,
            "c_BF": cBF,
            "c_OR": cOR,
            "c_OR0": cOR0,
            "Theta_Const": ThC,
            "Theta_Slope": ThS,
        }


# ------------------------------
# Minimal self-test
# ------------------------------

def _demo_square_graph() -> None:
    """Run a quick demo on a 4-cycle (square) graph including analytic transfers."""
    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0,1,1,2,2,3,3,0],
                               [1,0,2,1,3,2,0,3]], dtype=torch.long)
    data = Data(num_nodes=4, edge_index=edge_index)
    eng = CurvatureEngine(data)
    base = eng.compute_all()
    print("Edges u<v:\n", base["edges"])
    print("Degrees i,j:\n", np.c_[base["deg_i"], base["deg_j"]])
    print("Triangles per edge:", base["triangle"])
    print("Xi per edge:", base["Xi"])
    print("sho_max per edge:", base["sho_max"])
    print("C4 per edge:", np.round(base["C4"], 6))
    print("c_BF per edge:", np.round(base["c_BF"], 6))
    print("c_OR per edge:", np.round(base["c_OR"], 6))
    print("c_OR0 per edge:", np.round(base["c_OR0"], 6))

    # Per-edge analytic transfer (self-applied) examples
    bf_bounds = eng.bounds_from_BF(base["c_BF"])
    or_bounds = eng.bounds_from_OR(base["c_OR"])
    print("Per-edge OR lower from c_BF:", np.round(bf_bounds["c_OR_lower_from_c_BF"], 6))
    print("Per-edge OR upper from c_BF:", np.round(bf_bounds["c_OR_upper_from_c_BF"], 6))
    print("Per-edge BF lower from c_OR:", np.round(or_bounds["c_BF_lower_from_c_OR"], 6))
    print("Per-edge BF upper from c_OR:", np.round(or_bounds["c_BF_upper_from_c_OR"], 6))


if __name__ == "__main__":
    _demo_square_graph()
