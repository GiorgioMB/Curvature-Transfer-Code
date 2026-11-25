import numpy as np, networkx as nx, orcut, random, sys, time
from orcut.measures import Walk
sys.path.insert(0, '/home/claude')
from tq_general import tq_general
def csr(n, E):
    src = np.concatenate([E[:, 0], E[:, 1]]); dst = np.concatenate([E[:, 1], E[:, 0]])
    o = np.lexsort((dst, src)); src, dst = src[o], dst[o]
    ip = np.zeros(n + 1, np.int64); np.add.at(ip, src + 1, 1)
    return np.cumsum(ip), dst.astype(np.int64)
MEAS = {'lazy': (0, 0.0, 0), 'nonlazy': (1, 0.0, 0), 'p=1/2': (1, 0.5, 0), 'p=9/10': (1, 0.9, 0), 'degree-biased': (0, 0.0, 1)}
random.seed(4)
stats = {k: [0, 0, 0.0, 0.0, 0, 0] for k in MEAS}   # edges, violations, gap_lo, gap_hi, signs, ceiling
worst = {k: 0.0 for k in MEAS}
for s in range(160):
    G = random.choice([lambda: nx.powerlaw_cluster_graph(random.randint(15, 60), random.randint(1, 4), random.uniform(.1, .9), seed=s),
                       lambda: nx.watts_strogatz_graph(random.randint(15, 60), random.choice([4, 6]), random.uniform(0, .4), seed=s),
                       lambda: nx.barabasi_albert_graph(random.randint(15, 60), random.randint(1, 4), seed=s),
                       lambda: nx.random_geometric_graph(random.randint(15, 60), random.uniform(.2, .45), seed=s),
                       lambda: nx.gnp_random_graph(random.randint(15, 50), random.uniform(.08, .3), seed=s)])()
    G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    if G.number_of_edges() < 2: continue
    g = orcut.Graph.from_edges(list(G.edges())); E = np.ascontiguousarray(np.asarray(g.edges), dtype=np.int64)
    n = int(E.max()) + 1; ip, ix = csr(n, E)
    for name, (im, p, wm) in MEAS.items():
        if name == 'lazy': k = orcut.curvature(g)
        elif name == 'degree-biased':
            deg = np.diff(np.asarray(g.indptr)); w = 1.0 / deg[np.asarray(g.indices)]
            k = orcut.curvature(g, measure=Walk(weights=w, idleness='lazy'))
        else: k = orcut.curvature(g, measure=p)
        out = np.empty((len(E), 4)); tq_general(ip, ix, E[:, 0].copy(), E[:, 1].copy(), im, p, wm, out, 2)
        lo, up = out[:, 0], out[:, 1]
        st = stats[name]; st[0] += len(E)
        bad = (lo > k + 1e-9) | (up < k - 1e-9); st[1] += int(bad.sum())
        if bad.any(): worst[name] = max(worst[name], float(np.max(np.maximum(lo - k, k - up))))
        st[2] += float(np.sum(k - lo)); st[3] += float(np.sum(up - k))
        st[4] += int(np.sum((lo > 1e-12) | (up < -1e-12))); st[5] += int(np.sum(np.abs(k) > 1e-12))
for name, st in stats.items():
    print(f"{name:14s} {st[0]:6d} edges  violations {st[1]} (worst {worst[name]:.2e})  mean gap lo {st[2]/st[0]:.4f} hi {st[3]/st[0]:.4f}  "
          f"signs {100*st[4]/st[0]:.1f}% of {100*st[5]/st[0]:.1f}% possible")
