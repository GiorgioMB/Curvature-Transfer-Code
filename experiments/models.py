import math
import random
from typing import List, Tuple

def _add_undirected(edges, u, v):
    if u == v: 
        return
    if u > v:
        u, v = v, u
    edges.add((u, v))

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

def make_hyperbolic_random_graph(n: int, R: float, alpha: float = 1.0, T: float = 0.0, seed: int = 0):
    """Generates a random graph in the native hyperbolic model (Krioukov et al. 2010).
    Nodes are distributed in a hyperbolic disk of radius R with curvature -alpha^2.
    Each pair of nodes at hyperbolic distance d is connected with probability 
    p(d) = 1/(1 + exp((d-R)/(2T))). T=0 gives a sharp threshold at d=R.
    Returns (n, edges) where edges is a list of (u,v) with u < v.
    """
    import math, random
    rnd = random.Random(seed)

    # angles and radii
    thetas = [2.0*math.pi*rnd.random() for _ in range(n)]
    cR = math.cosh(alpha*R)
    rs, cosh_r, sinh_r = [], [], []
    for _ in range(n):
        u = rnd.random()
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
                if rnd.random() < p:
                    edges.add((i, j))
    return n, sorted(edges)
