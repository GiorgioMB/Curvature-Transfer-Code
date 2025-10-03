"""
Graph model generators, this module provides several classic random graph
generators plus a fast, parallel implementation of the native hyperbolic 
random graph (HRG) model. Everything returns undirected simple graphs as 
a sorted list of edges (u, v) with u < v.

Included models
- Erdos--Renyi G(n, p)
- Watts--Strogatz small-world model
- Barabasi--Albert preferential attachment
- Random geometric (unit square)
- Random d-regular
- Stochastic Block Model (SBM): homogeneous assortative / disassortative
- Cycle, grid, d-ary tree, complete graph, toroidal 
- Native HRG (Krioukov et al.), both a simple reference version and a scalable parallel 
  version using shared memory and tiled computation.

Main concepts for the hyperbolic model
- Each node is placed in a hyperbolic disk of radius R. Angles are uniform; the
  radial coordinate is sampled so that cosh(alpha * r) is uniform in [1, cosh(alpha * R)].
- The hyperbolic distance d between two nodes can be computed via cosh(alpha d)
  from the radii and the angle difference (law of cosines in the hyperbolic plane).
- Edges are independent given positions: p(d) = 1 / (1 + exp((d - R) / (2T))).
  Temperature T = 0 gives a hard threshold at d <= R.
- The parallel generator precomputes cosh/sinh of the radii and cos/sin of the
  angles, stores them in shared memory, and assigns disjoint tiles of the upper
  triangle to worker processes. This avoids duplicates and minimizes overhead.

Usage (example)
>>> n, edges = erdos_renyi(100, 0.05, seed=1)
>>> n, edges = make_hyperbolic_random_graph(10000, R=10.0, alpha=1.0, T=0.5, seed=123, n_jobs=-1)
"""
import random
from typing import List, Tuple
import os
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
try:
    import numexpr as _ne
except Exception:
    _ne = None
    print(f"[WARNING] numexpr not available; falling back to numpy for hyperbolic random graph generation.")

# Global references for worker processes. We attach shared-memory views here.
_global = {"thetas": None, "cah": None, "sah": None}  # cosh(alpha r), sinh(alpha r)


def _add_undirected(
        edges: set,
        u: int,
        v: int
        ) -> None:
    """
    Insert an undirected edge (u, v) in canonical form (u < v); skip self-loops.
    """
    if u == v: 
        return
    if u > v:
        u, v = v, u
    edges.add((u, v))


def _init_hrg_shm(
        meta: dict
        ) -> None:
    """
    Initializer for HRG worker processes: attach to shared memory arrays.

    Parameters
    - meta: dict mapping key -> (name, shape, dtype_str) for each shared array
      where keys are: 'cah', 'sah', 'cos_t', 'sin_t'.

    Behavior
    - Opens SharedMemory blocks by name and creates NumPy views for each.
    - Stores views in the module-level _global dict so workers can read them.
    """
    # meta maps key -> (name, shape, dtype_str)
    ca_name, ca_shape, ca_dtype = meta["cah"]
    sa_name, sa_shape, sa_dtype = meta["sah"]
    ct_name, ct_shape, ct_dtype = meta["cos_t"]
    st_name, st_shape, st_dtype = meta["sin_t"]

    shm_ca = shared_memory.SharedMemory(name=ca_name)
    shm_sa = shared_memory.SharedMemory(name=sa_name)
    shm_ct = shared_memory.SharedMemory(name=ct_name)
    shm_st = shared_memory.SharedMemory(name=st_name)

    _global["cah_shm"]   = shm_ca
    _global["sah_shm"]   = shm_sa
    _global["cos_t_shm"] = shm_ct
    _global["sin_t_shm"] = shm_st

    _global["cah"]   = np.ndarray(ca_shape, dtype=np.dtype(ca_dtype), buffer=shm_ca.buf)
    _global["sah"]   = np.ndarray(sa_shape, dtype=np.dtype(sa_dtype), buffer=shm_sa.buf)
    _global["cos_t"] = np.ndarray(ct_shape, dtype=np.dtype(ct_dtype), buffer=shm_ct.buf)
    _global["sin_t"] = np.ndarray(st_shape, dtype=np.dtype(st_dtype), buffer=shm_st.buf)


def _tile_worker(
        i0: int, 
        i1: int, 
        j0: int, 
        j1: int, 
        is_diag: bool, 
        cR: float, 
        R: float, 
        alpha: float, 
        T: float, 
        seed: int
        ) -> np.ndarray:
    """
    Compute edges inside a tile [i0:i1] x [j0:j1] of the adjacency upper triangle.

    Method
    - Use the identity cos(Delta theta) = cos(theta_i)cos(theta_j) + sin(theta_i)sin(theta_j).
    - Compute cosh(alpha d) = cosh(alpha r_i)cosh(alpha r_j)
      - sinh(alpha r_i)sinh(alpha r_j) * cos(Delta theta).
    - For T = 0: add an edge if cosh(alpha d) <= cosh(alpha R) (hard threshold).
    - For T > 0: compute d = arccosh(max(1, cosh(alpha d))) / alpha and connect
      with probability p(d) = 1 / (1 + exp((d - R)/(2T))).

    Parameters
    - i0, i1, j0, j1: integer tile bounds
    - is_diag: True if this tile is on the diagonal; we then keep only the
      strict upper triangle within the tile to avoid duplicates
    - cR: cosh(alpha * R)
    - R, alpha, T: HRG parameters
    - seed: 64-bit seed to initialize an independent RNG stream per tile

    Returns
    - 2D int64 array of shape (m, 2) with rows [u, v] (u < v), or empty array.
    """
    cah = _global["cah"];   sah = _global["sah"]
    cos_t = _global["cos_t"]; sin_t = _global["sin_t"]

    ci = cah[i0:i1]; si = sah[i0:i1]
    cj = cah[j0:j1]; sj = sah[j0:j1]

    cosi = cos_t[i0:i1]; sini = sin_t[i0:i1]
    cosj = cos_t[j0:j1]; sinj = sin_t[j0:j1]

    # cos(Delta theta) via two outer products (gemm-friendly)
    ococ  = np.outer(cosi, cosj)
    osisj = np.outer(sini, sinj)
    # Outer products for radial hyperbolic terms
    ci_cj = np.outer(ci, cj)
    si_sj = np.outer(si, sj)

    # cosh(alpha d) = ci*cj - si*sj * cos(Delta theta)
    if _ne is not None:
        cosh_ad = _ne.evaluate("ci_cj - si_sj * (ococ + osisj)")
    else:
        cosh_ad = ci_cj - si_sj * (ococ + osisj)

    if T <= 0.0:
        # Hard threshold in 'cosh space'
        mask = (cosh_ad <= cR)
        if is_diag:
            mask = np.triu(mask, k=1)
        ii, jj = np.nonzero(mask)
        if ii.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        u = i0 + ii
        v = j0 + jj
        return np.stack((u, v), axis=1).astype(np.int64)

    # T > 0: logistic kernel; compute d and then p(d)
    # d = arccosh(max(1, cosh_ad))/alpha
    # (Use a fresh array for stability; ufuncs may reuse buffers)
    d = np.arccosh(np.maximum(1.0, cosh_ad)) / alpha

    # p = 1 / (1 + exp((d - R)/(2T)))
    p = 1.0 / (1.0 + np.exp((d - R) / (2.0 * T)))

    # Independent RNG stream per tile
    rng = np.random.default_rng(
        np.uint64((seed ^ 0x9E3779B97F4A7C15) + i0 * 13091204281 + j0 * 334214459)
    )
    mask = rng.random(size=p.shape) < p
    if is_diag:
        mask = np.triu(mask, k=1)

    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    u = i0 + ii
    v = j0 + jj
    return np.stack((u, v), axis=1).astype(np.int64)


def erdos_renyi(
        n: int, 
        p: float, 
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Erdos--Renyi G(n, p): connect each pair independently with probability p.

    Returns (n, sorted list of undirected edges (u, v) with u < v).
    """
    rnd = random.Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if rnd.random() < p:
                edges.add((u, v))
    return n, sorted(edges)


def watts_strogatz(
        n: int, 
        k: int, 
        beta: float, 
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """Watts--Strogatz small-world model.

    Start with a ring where each node connects to k/2 neighbors on each side,
    then rewire each ring edge with probability beta to a random endpoint.

    Returns (n, sorted list of edges).
    """
    assert k % 2 == 2 or k % 2 == 0
    rnd = random.Random(seed)
    edges = set()
    # initial ring
    half = k // 2
    for u in range(n):
        for d in range(1, half+1):
            v = (u + d) % n
            _add_undirected(edges, u, v)
    # rewire one orientation of each original edge
    for u in range(n):
        for d in range(1, half+1):
            v = (u + d) % n
            if rnd.random() < beta:
                # remove old edge and add a new one to a random node w != u, avoiding duplicates/self-loops
                try:
                    edges.remove(tuple(sorted((u, v))))
                except KeyError:
                    pass
                while True:
                    w = rnd.randrange(n)
                    if w != u and (min(u,w), max(u,w)) not in edges:
                        _add_undirected(edges, u, w)
                        break
    return n, sorted(edges)


def barabasi_albert(
        n: int, 
        m: int, 
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """Barabasi--Albert preferential attachment.

    Start with a clique of size m+1. Each new node attaches to m existing
    nodes chosen with probability proportional to degree.
    Returns (n, sorted edges).
    """
    assert m >= 1 and n >= m+1
    rnd = random.Random(seed)
    edges = set()
    # initial clique of size m+1
    for u in range(m+1):
        for v in range(u+1, m+1):
            _add_undirected(edges, u, v)
    # degree list for preferential selection
    deg = [0]*n
    for u,v in edges:
        deg[u]+=1; deg[v]+=1
    # list of nodes with multiplicity = degree (for sampling)
    mult = []
    for u in range(m+1):
        mult.extend([u]*deg[u])
    for new in range(m+1, n):
        targets = set()
        while len(targets) < m:
            if mult:
                t = mult[rnd.randrange(len(mult))]
            else:
                t = rnd.randrange(new)  # fallback uniform if mult empty
            if t != new:
                targets.add(t)
        for t in targets:
            _add_undirected(edges, new, t)
            deg[new]+=1; deg[t]+=1
            mult.append(t)
            mult.append(new)
    return n, sorted(edges)


def random_geometric(
        n: int, 
        r: float, 
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Random geometric graph in the unit square: connect if Euclidean distance < r.

    Returns (n, sorted edges).
    """
    rnd = random.Random(seed)
    pts = [(rnd.random(), rnd.random()) for _ in range(n)]
    edges = set()
    r2 = r*r
    for u in range(n):
        x1,y1 = pts[u]
        for v in range(u+1, n):
            x2,y2 = pts[v]
            dx = x1-x2; dy=y1-y2
            if dx*dx + dy*dy < r2:
                _add_undirected(edges, u, v)
    return n, sorted(edges)

def d_regular_graph(
        n: int,
        d: int,
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Random d-regular simple graph on n nodes using a configuration-model pairing
    with rejection (restart on dead-ends). Requires 0 <= d < n and n*d even.
    Returns (n, sorted edges).
    """
    if d < 0 or d >= n:
        raise ValueError("d must satisfy 0 <= d < n")
    if (n * d) % 2 != 0:
        raise ValueError("n * d must be even for a d-regular graph")
    if d == 0:
        return n, []
    rnd = random.Random(seed)
    max_tries = 128
    for _ in range(max_tries):
        # Multiset of 'stubs' (one per half-edge)
        stubs = [u for u in range(n) for _ in range(d)]
        rnd.shuffle(stubs)
        edges = set()
        ok = True
        while stubs:
            u = stubs.pop()
            # find a partner v that avoids loops/multi-edges
            found = False
            for k in range(len(stubs)):
                v = stubs[k]
                if v == u:
                    continue
                e = (u, v) if u < v else (v, u)
                if e in edges:
                    continue
                # pair u with v
                stubs.pop(k)
                edges.add(e)
                found = True
                break
            if not found:
                ok = False
                break
        if ok and len(edges) == (n * d) // 2:
            return n, sorted(edges)
    raise RuntimeError("Failed to generate a simple d-regular graph after multiple attempts")


def cycle_graph(
        n: int
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Cycle on n nodes: edges (0,1), (1,2), ..., (n-1,0).
    """
    edges = set()
    for u in range(n):
        _add_undirected(edges, u, (u+1)%n)
    return n, sorted(edges)


def grid_graph(
        m: int, 
        n: int
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    m x n rectangular grid with 4-neighbor connectivity.
    """
    edges = set()
    def id(i,j): return i*n + j
    for i in range(m):
        for j in range(n):
            if i+1 < m: _add_undirected(edges, id(i,j), id(i+1,j))
            if j+1 < n: _add_undirected(edges, id(i,j), id(i,j+1))
    return m*n, sorted(edges)

def torus_graph(
        m: int,
        n: int
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    m x n toroidal grid (C_m x C_n) with 4-neighbor connectivity and wraparound.
    Returns (m*n, sorted edges).
    """
    edges = set()
    def id(i,j): return i*n + j
    for i in range(m):
        for j in range(n):
            _add_undirected(edges, id(i,j), id((i+1) % m, j))
            _add_undirected(edges, id(i,j), id(i, (j+1) % n))
    return m*n, sorted(edges)

def dary_tree(
        d: int, 
        h: int
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Rooted d-ary tree of height h (root at level 0).

    Returns (n, sorted edges). For d <= 1, this degenerates to a path of length h.
    """
    if h < 0: 
        return 0, []
    if d < 1:
        return 1, []
    # number of nodes: (d^(h+1)-1)/(d-1)
    n = (d**(h+1)-1)//(d-1) if d > 1 else h+1
    edges = set()
    # parent index p has children indices from c=d*p+1 to d*p+d for a BFS labeling
    for p in range((d**h-1)//(d-1) if d>1 else h):
        for j in range(1, d+1):
            c = d*p + j
            if c < n:
                _add_undirected(edges, p, c)
    return n, sorted(edges)


def complete_graph(
        n: int
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Complete graph K_n: all pairs connected.
    """
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            _add_undirected(edges, u, v)
    return n, sorted(edges)


def deprecated_make_hyperbolic_random_graph(
        n: int, 
        R: float, 
        alpha: float = 1.0, 
        T: float = 0.0, 
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Reference (single-process) HRG generator for clarity and testing.

    Native hyperbolic model (Krioukov et al. 2010). Nodes are distributed in a
    hyperbolic disk of radius R with curvature -alpha^2. Connection probability:
        p(d) = 1 / (1 + exp((d - R) / (2T)))
    with T = 0 giving a hard threshold at d <= R.

    Returns (n, sorted edges). Slower than the parallel version but easy to read.
    """
    rng = np.random.default_rng(seed)

    # angles and radii
    thetas = [2.0*math.pi*rng.random() for _ in range(n)]
    cR = math.cosh(alpha*R)
    rs, cosh_r, sinh_r = [], [], []
    for _ in range(n):
        u = rng.random()
        x = 1.0 + u * (cR - 1.0)        # in [1, cosh(αR)]
        r = math.acosh(max(1.0, x)) / alpha
        rs.append(r)
        cosh_r.append(math.cosh(r))
        sinh_r.append(math.sinh(r))

    # edges
    edges = set()
    for i in range(n):
        ti, ci, si = thetas[i], cosh_r[i], sinh_r[i]
        for j in range(i+1, n):
            dj, cj, sj = thetas[j], cosh_r[j], sinh_r[j]
            dtheta = abs(ti - dj)
            if dtheta > math.pi:
                dtheta = 2.0*math.pi - dtheta
            cosh_d = ci*cj - si*sj*math.cos(dtheta)
            d = math.acosh(max(1.0, cosh_d))
            if T <= 0.0:
                if d <= R:
                    edges.add((i, j))
            else:
                p = 1.0 / (1.0 + math.exp((d - R)/(2.0*T)))
                if rng.random() < p:
                    edges.add((i, j))
    return n, sorted(edges)



def make_hyperbolic_random_graph(
    n: int, 
    R: float, 
    alpha: float = 1.0, 
    T: float = 0.0, 
    seed: int = 0,
    n_jobs: int | None = None, 
    block_size: int | None = None
    ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Parallel generator for the native HRG (Krioukov et al.).

    Positions
    - theta ~ Uniform[0, 2π)
    - cosh(alpha * r) ~ Uniform[1, cosh(alpha * R)]  (so r is concentrated near R)

    Connection probability
    - p(d) = 1 / (1 + exp((d - R) / (2T))).
      T = 0 gives a hard threshold at d <= R.

    Parallelism strategy
    - Precompute arrays: cos(theta), sin(theta), cosh(alpha r), sinh(alpha r).
    - Place these arrays in OS shared memory blocks so worker processes can read
      them without copying. Workers attach in _init_hrg_shm.
    - Partition the upper triangle of the adjacency matrix into tiles. Each
      worker computes edges for its tiles; diagonal tiles use a strict upper
      triangle mask to avoid self-loops and duplicates.

    Parameters
    - n : int Number of nodes
    - R : float Disk radius parameter
    - alpha : float, default=1.0 Curvature parameter (curvature = -alpha^2)
    - T : float, default=0.0 Temperature parameter (T=0 gives hard threshold)
    - seed : int, default=0 Random seed
    - n_jobs : int or None, default=None Number of parallel jobs. Follows scikit-learn conventions:
        - None: use all available CPU cores
        - 1: sequential execution
        - -1: use all available CPU cores
        - > 1: use exactly n_jobs processes
        - < -1: use (n_cpus + 1 + n_jobs) processes
    - block_size : int or None, default=None Tile size for parallelization. If None, 
        chosen adaptively. Larger tiles reduce overhead; smaller tiles increase load balance.
        
    Returns
    - tuple of (n, sorted list of edges (u, v) with u < v)
    """
    if n <= 1:
        return n, []

    # Validate and resolve n_jobs following scikit-learn conventions
    n_cpus = os.cpu_count() or 1
    
    if n_jobs is None:
        # Use all available CPUs
        resolved_n_jobs = n_cpus
    elif n_jobs == 1:
        # Sequential execution - we'll handle this by using a single process
        resolved_n_jobs = 1
    elif n_jobs == -1:
        # Use all available CPUs
        resolved_n_jobs = n_cpus
    elif n_jobs > 1:
        # Use exactly n_jobs processes
        resolved_n_jobs = min(n_jobs, n_cpus)  # Don't exceed available CPUs
    elif n_jobs < -1:
        # Use (n_cpus + 1 + n_jobs) processes
        resolved_n_jobs = max(1, n_cpus + 1 + n_jobs)
    else:  # n_jobs == 0
        raise ValueError("n_jobs cannot be 0")

    # Choose block_size to produce plenty of tiles (~O(resolved_n_jobs)) but not too tiny
    if block_size is None:
        target_tasks = max(8 * resolved_n_jobs, 32)
        per_axis = int(math.ceil(math.sqrt(2 * target_tasks)))
        block_size = max(128, int(math.ceil(n / per_axis)))

    rng = np.random.default_rng(seed)

    # --- Draw positions
    thetas = rng.random(n) * (2.0 * np.pi)
    cos_t = np.cos(thetas).astype(np.float64)
    sin_t = np.sin(thetas).astype(np.float64)

    cR = np.cosh(alpha * R)

    # Radial: x = cosh(α r) ∈ [1, cosh(α R)]
    u = rng.random(n)
    x = 1.0 + u * (cR - 1.0)       # x = cosh(α r)
    ar = np.arccosh(x)             # ar = α r
    # cah = cosh(α r) = x, sah = sinh(α r)
    cah = x.astype(np.float64, copy=False)
    sah = np.sinh(ar).astype(np.float64)

    # --- Put arrays in shared memory (cos_t, sin_t, cah, sah)
    shm_ca = shared_memory.SharedMemory(create=True, size=cah.nbytes)
    shm_sa = shared_memory.SharedMemory(create=True, size=sah.nbytes)
    shm_ct = shared_memory.SharedMemory(create=True, size=cos_t.nbytes)
    shm_st = shared_memory.SharedMemory(create=True, size=sin_t.nbytes)

    try:
        # Copy data into shared memory buffers
        ca_view = np.ndarray(cah.shape, dtype=cah.dtype, buffer=shm_ca.buf); ca_view[:] = cah
        sa_view = np.ndarray(sah.shape, dtype=sah.dtype, buffer=shm_sa.buf); sa_view[:] = sah
        ct_view = np.ndarray(cos_t.shape, dtype=cos_t.dtype, buffer=shm_ct.buf); ct_view[:] = cos_t
        st_view = np.ndarray(sin_t.shape, dtype=sin_t.dtype, buffer=shm_st.buf); st_view[:] = sin_t

        meta = {
            "cah":   (shm_ca.name, cah.shape,   str(cah.dtype)),
            "sah":   (shm_sa.name, sah.shape,   str(sah.dtype)),
            "cos_t": (shm_ct.name, cos_t.shape, str(cos_t.dtype)),
            "sin_t": (shm_st.name, sin_t.shape, str(sin_t.dtype)),
        }

        # --- Enumerate tiles covering the upper triangle (avoid duplicates)
        tiles = []
        for i0 in range(0, n, block_size):
            i1 = min(n, i0 + block_size)
            # diagonal tile
            tiles.append((i0, i1, i0, i1, True))
            # off-diagonal tiles to the right
            for j0 in range(i1, n, block_size):
                j1 = min(n, j0 + block_size)
                tiles.append((i0, i1, j0, j1, False))

        edges_parts = []
        
        # Handle sequential execution case without multiprocessing overhead
        if resolved_n_jobs == 1:
            # Initialize global state for sequential execution
            _global["cah"] = cah
            _global["sah"] = sah
            _global["cos_t"] = cos_t
            _global["sin_t"] = sin_t
            
            for (i0, i1, j0, j1, is_diag) in tiles:
                edges_part = _tile_worker(i0, i1, j0, j1, is_diag, cR, R, alpha, T, seed)
                edges_parts.append(edges_part)
        else:
            # Parallel execution using a process pool
            with ProcessPoolExecutor(
                max_workers=resolved_n_jobs, initializer=_init_hrg_shm, initargs=(meta,)
            ) as ex:
                futs = [
                    ex.submit(_tile_worker, i0, i1, j0, j1, is_diag, cR, R, alpha, T, seed)
                    for (i0, i1, j0, j1, is_diag) in tiles
                ]
                for f in as_completed(futs):
                    edges_parts.append(f.result())

        # --- Merge and sort edges (u < v)
        if edges_parts:
            E = np.vstack(edges_parts) if len(edges_parts) > 1 else edges_parts[0]
            if E.size == 0:
                edges = []
            else:
                order = np.lexsort((E[:, 1], E[:, 0]))  # primary: u, secondary: v
                E = E[order]
                edges = list(map(tuple, E.tolist()))
        else:
            edges = []

    finally:
        # Detach in parent and free shared memory blocks to avoid leaks
        for shm in (shm_ca, shm_sa, shm_ct, shm_st):
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    return n, edges


def stochastic_block_model(
        sizes: List[int],
        P: List[List[float]],
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Undirected Stochastic Block Model (SBM).

    Parameters
    - sizes: list of community sizes [n1, n2, ..., nk], each > 0
    - P: k x k symmetric matrix of connection probabilities in [0, 1]
    - seed: RNG seed

    Returns (n, sorted list of undirected edges (u, v) with u < v).
    """
    k = len(sizes)
    assert k >= 1 and all(s > 0 for s in sizes), "sizes must be positive"
    assert len(P) == k and all(len(row) == k for row in P), "P must be k x k"
    # light symmetry / bounds checks
    for i in range(k):
        for j in range(k):
            pij = P[i][j]
            assert 0.0 <= pij <= 1.0, "probabilities must be in [0, 1]"
            if i <= j:
                # tolerate tiny float differences
                assert abs(P[j][i] - pij) < 1e-12, "P must be symmetric for undirected graphs"

    rnd = random.Random(seed)
    n = sum(sizes)
    edges = set()

    # block offsets
    offsets = []
    cur = 0
    for s in sizes:
        offsets.append(cur)
        cur += s

    # sample edges block-by-block (upper triangle, avoid duplicates)
    for a in range(k):
        oa, sa = offsets[a], sizes[a]
        for b in range(a, k):
            ob, sb = offsets[b], sizes[b]
            p = P[a][b]
            for i in range(sa):
                u = oa + i
                j_start = i + 1 if a == b else 0
                for j in range(j_start, sb):
                    v = ob + j
                    if rnd.random() < p:
                        _add_undirected(edges, u, v)
    return n, sorted(edges)

def make_sbm_graph(
        sizes: List[int],
        p_in: float,
        p_out: float,
        seed: int = 0
        ) -> Tuple[int, List[Tuple[int,int]]]:
    """
    Homogeneous SBM with within-block p_in and between-block p_out.
    Assortative when p_in > p_out, disassortative when p_in < p_out.
    """
    k = len(sizes)
    P = [[p_out]*k for _ in range(k)]
    for i in range(k):
        P[i][i] = p_in
    return stochastic_block_model(sizes, P, seed)


# Test snippet to verify all graph generators work correctly
if __name__ == "__main__":
    print("[TEST] Initializing graph generators...")
    
    # Test parameters
    test_n = 10
    test_seed = 42
    
    def validate_graph(
            name, 
            n, 
            edges
            ) -> None:
        """
        Basic validation for graph structure (bounds, canonical edges, duplicates).
        """
        print(f"[TEST] {name}: n={n}, |E|={len(edges)}")
        
        # Check all edges are valid
        for u, v in edges:
            assert 0 <= u < n and 0 <= v < n, f"Invalid edge ({u}, {v}) for n={n}"
            assert u < v, f"Edge not in canonical form: ({u}, {v})"
        
        # Check no duplicate edges
        assert len(edges) == len(set(edges)), "Duplicate edges found"
        print(f"Valid structure")
    
    try:
        # Test each graph generator
        generators = [
            ("Erdos-Renyi", lambda: erdos_renyi(test_n, 0.3, test_seed)),
            ("Watts-Strogatz", lambda: watts_strogatz(test_n, 4, 0.3, test_seed)),
            ("Barabasi-Albert", lambda: barabasi_albert(test_n, 2, test_seed)),
            ("Random Geometric", lambda: random_geometric(test_n, 0.4, test_seed)),
            ("Cycle", lambda: cycle_graph(test_n)),
            ("Grid 3x3", lambda: grid_graph(3, 3)),
            ("Binary Tree (h=3)", lambda: dary_tree(2, 3)),
            ("Complete", lambda: complete_graph(5)),  # smaller for complete graph
            ("Hyperbolic (deprecated)", lambda: deprecated_make_hyperbolic_random_graph(test_n, 2.0, 1.0, 0.0, test_seed)),
            ("Hyperbolic (parallel)", lambda: make_hyperbolic_random_graph(test_n, 2.0, 1.0, 0.0, test_seed, n_jobs=1)),  # sequential 
            ("Hyperbolic (parallel multi)", lambda: make_hyperbolic_random_graph(test_n, 2.0, 1.0, 0.0, test_seed, n_jobs=-1)),  # all CPUs
            ("SBM (assortative)", lambda: make_sbm_graph([3, 4, 3], 0.8, 0.2, test_seed)),
            ("SBM (disassortative)", lambda: make_sbm_graph([3, 4, 3], 0.2, 0.8, test_seed)),
        ]
        for name, gen_func in generators:
            try:
                n, edges = gen_func()
                validate_graph(name, n, edges)
            except Exception as e:
                print(f"[TEST] {name}: FAILED - {e}")

        print("[TEST] All tests completed!")

    except Exception as e:
        print(f"[TEST] Test suite failed: {e}")
