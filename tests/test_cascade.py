"""Certify-then-solve cascade (curvature_transfer.screening.screen) and its kernels."""
import networkx as nx
import numpy as np
import pytest

from curvature_transfer import Graph, certify_threshold, screen
from curvature_transfer import _kernels as K
from curvature_transfer import intervals as I

GRAPHS = {
    "ba": lambda: nx.barabasi_albert_graph(300, 3, seed=1),
    "plc": lambda: nx.powerlaw_cluster_graph(300, 4, 0.5, seed=2),
    "er": lambda: nx.erdos_renyi_graph(300, 0.03, seed=3),
    "rgg": lambda: nx.random_geometric_graph(300, 0.1, seed=4),
    "ws": lambda: nx.watts_strogatz_graph(300, 8, 0.1, seed=5),
    "karate": nx.karate_club_graph,
}


def _graph(G):
    G = nx.convert_node_labels_to_integers(G)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    return Graph.from_edges(np.array(G.edges()), G.number_of_nodes())


@pytest.fixture(scope="module", params=sorted(GRAPHS))
def case(request):
    g = _graph(GRAPHS[request.param]())
    ex = K.exact_ids(g.indptr, g.indices, g.edges, np.arange(g.num_edges, dtype=np.int64), True, False, False, 8)
    return g, ex  # ex[:, 0] = c_OR, ex[:, 3] = TLW dual upper bound (lazy)


def test_lean_exact_kernel_is_identical(case):
    g, ex = case
    v = K.exact_or_ids(g.indptr, g.indices, g.edges, np.arange(g.num_edges, dtype=np.int64), 8)
    assert np.array_equal(v, ex[:, 0])


@pytest.mark.parametrize("tau", [-0.5, 0.0, 0.2])
def test_fused_kernel(case, tau):
    g, ex = case
    r = K.screen_ids(g.indptr, g.indices, g.edges, np.arange(g.num_edges, dtype=np.int64), tau, I.ROUNDING_TOL, 8)
    assert np.array_equal(r[:, 1], ex[:, 3])
    cert = np.isnan(r[:, 0])
    assert np.array_equal(cert, ex[:, 3] + I.ROUNDING_TOL < tau)
    assert np.array_equal(r[~cert, 0], ex[~cert, 0])
    assert np.all(ex[cert, 0] < tau)


def test_stage_one_composes_into_the_sharpened_interval(case):
    g, ex = case
    s = I.stage_one(g)
    bl, bu = I.compute(g, "bf")
    sl, su = I.compute(g, "sharpened")
    assert np.array_equal(s["bf"][0], bl) and np.array_equal(s["bf"][1], bu)
    assert np.array_equal(s["two_hop"][0], sl)
    assert np.array_equal(np.minimum(s["two_hop"][1], ex[:, 3]), su)
    lo, up = I.widen(*s["two_hop"])
    assert np.all(lo <= ex[:, 0]) and np.all(ex[:, 0] <= up)
    # nested in the BF interval (Proposition 5.9; equality holds on many edges, so allow rounding)
    assert np.all(s["two_hop"][0] >= bl - 1e-12) and np.all(s["two_hop"][1] <= bu + 1e-12)


@pytest.mark.parametrize("tau", [0.0, -0.3])
def test_screen_threshold_is_exact_and_decides_like_the_sharpened_interval(case, tau):
    g, ex = case
    c = ex[:, 0]
    r = screen(g, tau=tau)
    assert np.array_equal(r["answer"], c < tau)
    lo, up = I.widen(*I.compute(g, "sharpened"))
    _, _, und = certify_threshold(lo, up, tau)
    assert r["n_solved"] == int(und.sum())
    assert r["n_bf"] <= r["n_two_hop"] <= r["n_two_hop"] + r["n_tlw"] == r["M"] - r["n_solved"]


def test_two_hop_from_stored_arrays_equals_stage_one(case):
    from curvature_transfer import CurvatureEngine
    g, _ = case
    b = CurvatureEngine(g).bounds()
    lo, up = I.two_hop_from_arrays(b)
    s = I.stage_one(g)["two_hop"]
    assert np.array_equal(lo, s[0]) and np.array_equal(up, s[1])
