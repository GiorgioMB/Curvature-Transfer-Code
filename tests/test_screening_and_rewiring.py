import random

import networkx as nx
import numpy as np
import pytest

from curvature_transfer import CurvatureEngine, Graph, certify_bottom_k, certify_then_solve, certify_threshold
from curvature_transfer import rewiring


def _random_intervals(rng, M):
    c = np.round(rng.normal(size=M), 1)  # many ties on purpose
    lo = c - rng.uniform(0, 1, M) * (rng.random(M) < 0.7)
    up = c + rng.uniform(0, 1, M) * (rng.random(M) < 0.7)
    return c, lo, up


def test_certificates_are_sound():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = int(rng.integers(2, 60))
        c, lo, up = _random_intervals(rng, M)
        cin, cout, _ = certify_threshold(lo, up, 0.0)
        assert np.all(c[cin] < 0) and np.all(c[cout] >= 0)
        k = int(rng.integers(1, M))
        cin, cout, _ = certify_bottom_k(lo, up, k)
        kth = np.sort(c)[k - 1]
        assert np.all(c[cin] <= kth) and cin.sum() <= k
        assert np.all(c[cout] >= kth)


def test_certify_then_solve_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(100):
        M = int(rng.integers(5, 80))
        c, lo, up = _random_intervals(rng, M)
        r = certify_then_solve(lo, up, lambda ids: c[ids], tau=0.0)
        assert np.array_equal(r["answer"], c < 0)
        k = int(rng.integers(1, M))
        r = certify_then_solve(lo, up, lambda ids: c[ids], tau=None, k=k)
        chosen = np.flatnonzero(r["answer"])
        kth = np.sort(c)[k - 1]
        assert len(chosen) == k and np.all(c[chosen] <= kth)
        assert np.all(np.isin(np.flatnonzero(c < kth), chosen))


def test_screening_on_a_real_interval():
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(200, 3, seed=0))
    eng = CurvatureEngine(G)
    c = eng.exact(("c_OR",))["c_OR"]
    b = eng.bounds()
    r = certify_then_solve(b["sharpened_lower"], b["sharpened_upper"],
                           lambda ids: eng.exact(("c_OR",), edges=ids)["c_OR"], tau=0.0)
    assert np.array_equal(r["answer"], c < 0) and r["frac_avoided"] > 0.5


@pytest.mark.parametrize("metric", rewiring.METRICS)
def test_local_scores_equal_global(metric):
    G = nx.convert_node_labels_to_integers(nx.powerlaw_cluster_graph(80, 3, 0.4, seed=2))
    g = Graph.from_edges(np.array(G.edges()))
    add, rem = rewiring.scores(g, metric)
    adj = rewiring._adjacency(g.num_nodes, map(tuple, g.edges))
    for e in range(0, g.num_edges, 9):
        u, v = (int(x) for x in g.edges[e])
        assert rewiring.local_edge_score(adj, u, v, metric, "remove") == pytest.approx(rem[e])
        assert rewiring.local_edge_score(adj, u, v, metric, "add") == pytest.approx(add[e])


@pytest.mark.parametrize("metric", rewiring.METRICS)
def test_rewiring_invariants(metric):
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(60, 2, seed=3))
    E = np.array(G.edges())
    a = rewiring.sdrf(60, E, metric, max_iters=4, rng=random.Random(0))
    b = rewiring.sdrf(60, E, metric, max_iters=4, rng=random.Random(0))
    assert np.array_equal(a, b)  # deterministic given the rng
    assert np.all(a[:, 0] < a[:, 1]) and len(np.unique(a, axis=0)) == len(a)
    c = rewiring.borf(60, E, metric, max_iters=2, batch_add=2, batch_remove=2)
    assert np.all(c[:, 0] < c[:, 1])
    assert abs(len(c) - len(E)) <= 4


def test_fosr_adds_edges():
    E = np.array(nx.path_graph(12).edges())
    out = rewiring.fosr(12, E, max_iters=3)
    assert len(out) == len(E) + 3


def test_pyg_transforms():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from curvature_transfer.pyg import BORFRewiring, SDRFRewiring
    G = nx.barabasi_albert_graph(40, 2, seed=0)
    e = torch.tensor(list(G.edges())).T
    data = Data(num_nodes=40, edge_index=torch.cat([e, e.flip(0)], 1))
    for T in (SDRFRewiring(max_iters=3, seed=0), BORFRewiring(max_iters=2)):
        out = T(data.clone())
        assert out.edge_index.shape[1] == out.edge_attr.shape[0]
        assert set(out.edge_attr.view(-1).tolist()) <= {0.0, 1.0}
