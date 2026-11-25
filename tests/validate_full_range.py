import sys, random, time, numpy as np, networkx as nx, orcut
from orcut.measures import Walk
sys.path.insert(0, '/home/claude/Curvature-Transfer-Code'); sys.path.insert(0, '/home/claude')
from curvature_transfer.full_range import upper_end, lower_and_relaxed
from curvature_transfer.tq_general import tq_general
from curvature_transfer.tq_lazy import csr_sorted
import optimality as OPT
MODES = {'lazy': (0, 0.0, 0), 0.0: (1, 0.0, 0), 'wj': (2, 0.0, 0), 'degree': (0, 0.0, 1)}
def exact(g, E, deg, measure):
    if measure == 'lazy': return orcut.curvature(g)
    if measure == 'degree':
        return orcut.curvature(g, measure=Walk(weights=1.0 / deg[np.asarray(g.indices)], idleness='lazy'))
    if measure == 'wj':
        k = np.empty(len(E)); b = np.maximum(deg[E[:, 0]], deg[E[:, 1]])
        for bb in np.unique(b):
            ids = np.flatnonzero(b == bb); k[ids] = orcut.curvature(g, measure=1.0 / (bb + 1), edges=ids)
        return k
    return orcut.curvature(g, measure=measure)
random.seed(21); T = {m: [0, 0, 0.0, 0.0, 0, 0, 0] for m in MODES}; cross = [0, 0.0]; t0 = time.time()
for s in range(70):
    G = random.choice([lambda: nx.powerlaw_cluster_graph(random.randint(12, 30), random.randint(2, 3), random.uniform(.2, .9), seed=s),
                       lambda: nx.watts_strogatz_graph(random.randint(12, 30), 4, random.uniform(.1, .4), seed=s),
                       lambda: nx.barabasi_albert_graph(random.randint(12, 30), 2, seed=s),
                       lambda: nx.random_geometric_graph(random.randint(12, 30), random.uniform(.25, .45), seed=s)])()
    G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    g = orcut.Graph.from_edges(list(G.edges())); E = np.ascontiguousarray(np.asarray(g.edges), dtype=np.int64)
    n = int(E.max()) + 1; ip, ix = csr_sorted(n, E); deg = np.diff(ip)
    adj = [set() for _ in range(n)]
    for a, b in E: adj[a].add(int(b)); adj[b].add(int(a))
    EL = [(int(a), int(b)) for a, b in E]
    for m, (im, p, wm) in MODES.items():
        k = exact(g, E, deg, m)
        out = np.empty((len(E), 4)); tq_general(ip, ix, E[:, 0].copy(), E[:, 1].copy(), im, p, wm, out, 1)
        lo_s, hi_f = lower_and_relaxed(adj, deg, EL, m)
        hi_s = np.array([upper_end(adj, deg, m, u, v)[0] for u, v in EL])
        tol = 1e-9; st = T[m]; st[0] += len(E)
        bad = (out[:, 0] > lo_s + tol) | (lo_s > k + tol) | (k > hi_s + tol) | (hi_s > hi_f + tol) | (hi_s > out[:, 1] + tol) | np.isnan(hi_s)
        st[1] += int(bad.sum()); st[2] += float(np.sum(lo_s - out[:, 0])); st[3] += float(np.sum(out[:, 1] - hi_s))
        st[4] += int(np.sum((out[:, 0] > tol) | (out[:, 1] < -tol))); st[5] += int(np.sum((lo_s > tol) | (hi_s < -tol)))
        st[6] += int(np.sum(np.abs(hi_f - hi_s) > 1e-9))
        if m == 'lazy':
            for e in random.sample(range(len(EL)), min(4, len(EL))):
                u, v = EL[e]; ref = 1 - OPT.solve(OPT.edge_info(G, u, v), 'slots_all')
                cross[0] += 1; cross[1] = max(cross[1], abs(ref - hi_s[e]))
for m, st in T.items():
    print(f"{str(m):7s} {st[0]:5d} edges  chain violations {st[1]}  mean gap TQ->range lo {st[2]/st[0]:.4f} hi {st[3]/st[0]:.4f}  "
          f"signs TQ {100*st[4]/st[0]:.1f}% range {100*st[5]/st[0]:.1f}%  (slots bind on {st[6]} edges)")
print(f"lazy upper end vs the earlier integer program: {cross[0]} edges, max |difference| {cross[1]:.2e};  {time.time()-t0:.0f}s")
