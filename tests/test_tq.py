"""Fast checks of the triangle-quadrangle code against orcut (the full validations are the validate_*.py
scripts in this directory): exact containment for the lazy measure, containment for every one-step measure,
the chain of Theorem 5.2, and the exactness claims of Section 5."""
import random

import networkx as nx
import numpy as np
import orcut
import pytest
from orcut.measures import Walk

from curvature_transfer.full_range import lower_and_relaxed, upper_end
from curvature_transfer.tq_general import tq_general
from curvature_transfer.tq_lazy import bf, bounds_from, tq_lazy


def _graph(G):
    G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
    g = orcut.Graph.from_edges(np.array(G.edges(), np.int64), num_nodes=G.number_of_nodes())
    ge = np.ascontiguousarray(np.asarray(g.edges), np.int64)
    ip = np.ascontiguousarray(np.asarray(g.indptr), np.int64); ix = np.ascontiguousarray(np.asarray(g.indices), np.int64)
    return G, g, ge, ip, ix


def _graphs(k, seed):
    rnd = random.Random(seed)
    for s in range(k):
        yield rnd.choice([lambda: nx.powerlaw_cluster_graph(rnd.randint(15, 50), rnd.randint(1, 4), rnd.uniform(.1, .9), seed=s),
                          lambda: nx.watts_strogatz_graph(rnd.randint(15, 50), 4, rnd.uniform(0, .4), seed=s),
                          lambda: nx.barabasi_albert_graph(rnd.randint(15, 50), rnd.randint(1, 3), seed=s),
                          lambda: nx.random_geometric_graph(rnd.randint(15, 50), rnd.uniform(.2, .4), seed=s)])()


def _exact_contained(out, r):
    L, Wu, Wl = (out[:, c].astype(object) for c in range(3))
    num, den = r["w1_num"].astype(object), r["w1_den"].astype(object)
    return int(np.sum((num * L > Wu * den) | (Wl * den > num * L)))


def test_lazy_exact_containment():
    for G in _graphs(25, 1):
        G, g, ge, ip, ix = _graph(G)
        assert _exact_contained(tq_lazy(ip, ix, ge), orcut.curvature(g, exact_fraction=True)) == 0


@pytest.mark.parametrize("name,G", [("K8", nx.complete_graph(8)), ("Q5", nx.hypercube_graph(5))])
def test_both_ends_exact(name, G):
    G, g, ge, ip, ix = _graph(G)
    out = tq_lazy(ip, ix, ge); r = orcut.curvature(g, exact_fraction=True)
    L, Wu, Wl = (out[:, c].astype(object) for c in range(3))
    num, den = r["w1_num"].astype(object), r["w1_den"].astype(object)
    assert np.all(num * L == Wu * den) and np.all(Wl * den == num * L)


def test_lower_end_exact_without_short_cycles():
    for G in (nx.random_labeled_tree(40, seed=2), nx.cycle_graph(9)):
        G, g, ge, ip, ix = _graph(G)
        out = tq_lazy(ip, ix, ge); r = orcut.curvature(g, exact_fraction=True)
        assert np.all(r["w1_num"].astype(object) * out[:, 0].astype(object) == out[:, 1].astype(object) * r["w1_den"].astype(object))


@pytest.mark.parametrize("mode,exact", [((0, 0.0, 0), "lazy"), ((1, 0.0, 0), 0.0), ((1, 0.5, 0), 0.5), ((0, 0.0, 1), "degree")])
def test_general_kernel_contains(mode, exact):
    for G in _graphs(12, 2):
        G, g, ge, ip, ix = _graph(G)
        if exact == "degree":
            deg = np.diff(ip); k = orcut.curvature(g, measure=Walk(weights=1.0 / deg[np.asarray(g.indices)], idleness="lazy"))
        else:
            k = orcut.curvature(g, measure=exact)
        o = np.empty((len(ge), 4)); tq_general(ip, ix, ge[:, 0].copy(), ge[:, 1].copy(), *mode, o, 1)
        assert np.all(o[:, 0] <= k + 1e-9) and np.all(k <= o[:, 1] + 1e-9)


@pytest.mark.parametrize("measure", ["lazy", 0.0])
def test_full_range_chain(measure):
    for G in _graphs(4, 3):
        G, g, ge, ip, ix = _graph(G)
        adj = [set(G[v]) for v in range(G.number_of_nodes())]; deg = np.array([len(a) for a in adj])
        edges = [(int(a), int(b)) for a, b in ge]
        k = orcut.curvature(g) if measure == "lazy" else orcut.curvature(g, measure=0.0)
        mode = (0, 0.0, 0) if measure == "lazy" else (1, 0.0, 0)
        o = np.empty((len(ge), 4)); tq_general(ip, ix, ge[:, 0].copy(), ge[:, 1].copy(), *mode, o, 1)
        lo_s, hi_f = lower_and_relaxed(adj, deg, edges, measure)
        hi_s = np.array([upper_end(adj, deg, measure, u, v)[0] for u, v in edges])
        t = 1e-9
        assert np.all(o[:, 0] <= lo_s + t) and np.all(lo_s <= k + t) and np.all(k <= hi_s + t)
        assert np.all(hi_s <= hi_f + t) and np.all(hi_s <= o[:, 1] + t)


def test_bf_matches_definition():
    G, g, ge, ip, ix = _graph(nx.karate_club_graph())
    b = bf(ip, ix, ge)
    for e, (u, v) in enumerate(ge):
        a, c = G.degree(u), G.degree(v)
        if a > c:
            u, v, a, c = v, u, c, a
        Nu, Nv = set(G[u]), set(G[v]); C = Nu & Nv; Ui = Nu - Nv - {v}; Uj = Nv - Nu - {u}
        hx = {x: len(set(G[x]) & Uj) for x in Ui}; hy = {y: len(set(G[y]) & Ui) for y in Uj}
        gm = max(list(hx.values()) + list(hy.values()) + [0])
        box = sum(1 for x in hx if hx[x]) + sum(1 for y in hy if hy[y])
        ref = 0.0 if a <= 1 else 2 / a + 2 / c - 2 + (2 / c + 1 / a) * len(C) + (box / (gm * c) if gm else 0.0)
        assert abs(b[e] - ref) < 1e-12
