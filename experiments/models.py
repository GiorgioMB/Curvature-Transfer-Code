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

_global = {"thetas": None, "cah": None, "sah": None}  # cosh(alpha r), sinh(alpha r)

def _add_undirected(edges, u, v):
    if u == v: 
        return
    if u > v:
        u, v = v, u
    edges.add((u, v))

def _init_hrg_shm(meta):
    """Initializer for worker processes: attach to shared memory blocks."""
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

def _tile_worker(i0, i1, j0, j1, is_diag, cR, R, alpha, T, seed):
    """
    Compute edges inside a tile [i0:i1] x [j0:j1] using GEMM-friendly cosine:
        cos(Delta theta) = cos theta_i cos theta_j + sin theta_i sin theta_j.
    For T=0, threshold in 'cosh space'. For T>0, compute d and sample via logistic.
    """
    cah = _global["cah"];   sah = _global["sah"]
    cos_t = _global["cos_t"]; sin_t = _global["sin_t"]

    ci = cah[i0:i1]; si = sah[i0:i1]
    cj = cah[j0:j1]; sj = sah[j0:j1]

    cosi = cos_t[i0:i1]; sini = sin_t[i0:i1]
    cosj = cos_t[j0:j1]; sinj = sin_t[j0:j1]

    # cos(Delta theta) via two outer products
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

def erdos_renyi(n: int, p: float, seed: int = 0) -> Tuple[int, List[Tuple[int,int]]]:
    rnd = random.Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if rnd.random() < p:
                edges.add((u, v))
    return n, sorted(edges)

def watts_strogatz(n: int, k: int, beta: float, seed: int = 0) -> Tuple[int, List[Tuple[int,int]]]:
    """Ring lattice where each node connects to k/2 neighbors on each side; then rewire each edge (u,v) with prob beta."""
    assert k % 2 == 2 or k % 2 == 0
    rnd = random.Random(seed)
    edges = set()
    # initial ring
    half = k // 2
    for u in range(n):
        for d in range(1, half+1):
            v = (u + d) % n
            _add_undirected(edges, u, v)
    # rewire
    # Iterate over original edges (directional sense) to attempt rewiring of one orientation
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

def barabasi_albert(n: int, m: int, seed: int = 0) -> Tuple[int, List[Tuple[int,int]]]:
    """Preferential attachment: start with a clique of size m+1 and attach new nodes with m edges proportional to degree."""
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

def random_geometric(n: int, r: float, seed: int = 0) -> Tuple[int, List[Tuple[int,int]]]:
    """Unit square geometric graph: connect if Euclidean distance < r."""
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

def cycle_graph(n: int) -> Tuple[int, List[Tuple[int,int]]]:
    edges = set()
    for u in range(n):
        _add_undirected(edges, u, (u+1)%n)
    return n, sorted(edges)

def grid_graph(m: int, n: int) -> Tuple[int, List[Tuple[int,int]]]:
    edges = set()
    def id(i,j): return i*n + j
    for i in range(m):
        for j in range(n):
            if i+1 < m: _add_undirected(edges, id(i,j), id(i+1,j))
            if j+1 < n: _add_undirected(edges, id(i,j), id(i,j+1))
    return m*n, sorted(edges)

def dary_tree(d: int, h: int) -> Tuple[int, List[Tuple[int,int]]]:
    """Rooted d-ary tree of height h (root at level 0)."""
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

def complete_graph(n: int) -> Tuple[int, List[Tuple[int,int]]]:
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            _add_undirected(edges, u, v)
    return n, sorted(edges)

def deprecated_make_hyperbolic_random_graph(n: int, R: float, alpha: float = 1.0, T: float = 0.0, seed: int = 0):
    """Generates a random graph in the native hyperbolic model (Krioukov et al. 2010).
    Nodes are distributed in a hyperbolic disk of radius R with curvature -alpha^2.
    Each pair of nodes at hyperbolic distance d is connected with probability 
    p(d) = 1/(1 + exp((d-R)/(2T))). T=0 gives a sharp threshold at d=R.
    Returns (n, edges) where edges is a list of (u,v) with u < v.
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
    n: int, R: float, alpha: float = 1.0, T: float = 0.0, seed: int = 0,
    n_jobs: int | None = None, block_size: int | None = None
):
    """
    Parallel generator for the native HRG (Krioukov et al.):
      - Positions: theta ~ Unif[0,2pi), radial via cosh(alpha r) in [1, cosh(alpha R)]
      - Connection: p(d) = 1 / (1 + exp((d-R)/(2T)))  (T=0 => hard threshold d<=R)
      - n_jobs: Number of worker processes. If None, uses all available CPU cores.
    Returns (n, sorted list of edges (u,v) with u < v).
    """
    if n <= 1:
        return n, []

    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    # Choose block_size to produce plenty of tiles (~O(n_jobs)) but not too tiny
    if block_size is None:
        target_tasks = max(8 * n_jobs, 32)
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

        # --- Enumerate tiles covering the upper triangle
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
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_hrg_shm, initargs=(meta,)
        ) as ex:
            futs = [
                ex.submit(_tile_worker, i0, i1, j0, j1, is_diag, cR, R, alpha, T, seed)
                for (i0, i1, j0, j1, is_diag) in tiles
            ]
            for f in as_completed(futs):
                edges_parts.append(f.result())

        # --- Merge and sort edges
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
        # detach in parent and free shared memory blocks
        for shm in (shm_ca, shm_sa, shm_ct, shm_st):
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

    return n, edges


# Test snippet to verify all graph generators work correctly
if __name__ == "__main__":
    print("Testing graph generators...")
    
    # Test parameters
    test_n = 10
    test_seed = 42
    
    def validate_graph(name, n, edges):
        """Basic validation for graph structure."""
        print(f"{name}: n={n}, |E|={len(edges)}")
        
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
            ("Hyperbolic (parallel)", lambda: make_hyperbolic_random_graph(test_n, 2.0, 1.0, 0.0, test_seed, n_jobs=None)),
        ]
        for name, gen_func in generators:
            try:
                n, edges = gen_func()
                validate_graph(name, n, edges)
            except Exception as e:
                print(f"{name}: FAILED - {e}")
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
