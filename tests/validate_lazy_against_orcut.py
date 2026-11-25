import sys, random, numpy as np, networkx as nx, orcut
sys.path.insert(0, '/home/claude/Curvature-Transfer-Code'); sys.path.insert(0, '/home/claude')
from curvature_transfer.tq_lazy import tq_lazy, csr_sorted, bounds_from, bf
from curvature_transfer.tq_general import tq_general
from tq import tq as tq_old, prep

def exact(g):
    r = orcut.curvature(g, exact_fraction=True)
    return r['curvature'], r['w1_num'].astype(object), r['w1_den'].astype(object)

def check(G):
    G = nx.convert_node_labels_to_integers(G)
    g = orcut.Graph.from_edges(list(G.edges())); E = np.ascontiguousarray(np.asarray(g.edges), dtype=np.int64)
    n = int(E.max()) + 1; ip, ix = csr_sorted(n, E)
    k, num, den = exact(g)
    out = tq_lazy(ip, ix, E); L, Wu, Wl = (out[:, c].astype(object) for c in range(3))
    viol = int(np.sum((num * L > Wu * den) | (Wl * den > num * L)))       # exact containment
    lo, up = bounds_from(out)
    return G, E, ip, ix, k, num, den, out, lo, up, viol

def bf_ref(G, u, v):
    a, b = G.degree(u), G.degree(v)
    if a > b: u, v, a, b = v, u, b, a
    Nu, Nv = set(G[u]), set(G[v]); C = Nu & Nv; Ui = Nu - Nv - {v}; Uj = Nv - Nu - {u}
    if a <= 1: return 0.0
    hx = {x: len(set(G[x]) & Uj) for x in Ui}; hy = {y: len(set(G[y]) & Ui) for y in Uj}
    gmax = max(list(hx.values()) + list(hy.values()) + [0]); box = sum(1 for x in hx if hx[x]) + sum(1 for y in hy if hy[y])
    return 2 / a + 2 / b - 2 + (2 / b + 1 / a) * len(C) + (box / (gmax * b) if gmax else 0.0)

random.seed(11); tot = viol = 0; d_lo = d_up = 0.0; s_new = s_old = 0; bfbad = 0; gen_gap = 0.0
for s in range(220):
    G = random.choice([lambda: nx.powerlaw_cluster_graph(random.randint(15, 90), random.randint(1, 5), random.uniform(.05, .95), seed=s),
                       lambda: nx.watts_strogatz_graph(random.randint(15, 90), random.choice([4, 6, 8]), random.uniform(0, .5), seed=s),
                       lambda: nx.barabasi_albert_graph(random.randint(15, 90), random.randint(1, 5), seed=s),
                       lambda: nx.random_geometric_graph(random.randint(15, 90), random.uniform(.15, .4), seed=s),
                       lambda: nx.gnp_random_graph(random.randint(15, 60), random.uniform(.05, .3), seed=s),
                       lambda: nx.random_regular_graph(random.choice([3, 4, 5, 6]), random.choice([20, 30, 40]), seed=s)])()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if G.number_of_edges() < 3: continue
    G, E, ip, ix, k, num, den, out, lo, up, v = check(G)
    tot += len(E); viol += v
    o2, n2 = tq_old(int(E.max()) + 1, E, prep(int(E.max()) + 1, E))
    lo2, up2 = 1 - n2[:, 1] / n2[:, 0], 1 - n2[:, 2] / n2[:, 0]
    d_lo += float(np.sum(lo - lo2)); d_up += float(np.sum(up2 - up))
    s_new += int(np.sum((lo >= 0) | (up < 0))); s_old += int(np.sum((lo2 >= 0) | (up2 < 0)))
    og = np.empty((len(E), 4)); tq_general(ip, ix, E[:, 0].copy(), E[:, 1].copy(), 0, 0.0, 0, og, 2)
    gen_gap += float(np.sum(np.abs(og[:, 0] - lo) + np.abs(og[:, 1] - up)))
    b = bf(ip, ix, E); ref = np.array([bf_ref(G, int(x), int(y)) for x, y in E]); bfbad += int(np.sum(np.abs(b - ref) > 1e-12))
print(f'random graphs: {tot} edges, exact containment violations {viol}; vs previous kernel: lower +{d_lo/tot:.5f}, '
      f'upper -{d_up/tot:.5f} on average; signs {100*s_new/tot:.2f}% (previous {100*s_old/tot:.2f}%); '
      f'mean |integer - float general kernel| {gen_gap/tot:.2e}; BF mismatches {bfbad}')
# exactness claims of the paper
for name, G in [('K_8', nx.complete_graph(8)), ('Q_5', nx.hypercube_graph(5)), ('Q_7', nx.hypercube_graph(7))]:
    _, E, ip, ix, k, num, den, out, lo, up, v = check(G)
    L, Wu, Wl = (out[:, c].astype(object) for c in range(3))
    print(f'{name}: violations {v}, both ends exact on {int(np.sum((num*L == Wu*den) & (Wl*den == num*L)))}/{len(E)} edges')
for name, G in [('tree', nx.random_labeled_tree(60, seed=1)), ('C_9', nx.cycle_graph(9)), ('grid->girth6?', nx.cycle_graph(12))]:
    _, E, ip, ix, k, num, den, out, lo, up, v = check(G)
    L, Wu = out[:, 0].astype(object), out[:, 1].astype(object)
    print(f'{name}: violations {v}, lower end exact on {int(np.sum(num*L == Wu*den))}/{len(E)} edges')
