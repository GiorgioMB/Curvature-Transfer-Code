"""The production exact solver (_kernels) against the reference solver
(_reference): different formulation, different distance computation, different
max-flow code, so bit-identical outputs on varied graphs are a strong check.
The reference itself is checked against linear programming in test_exact.py."""
import networkx as nx
import numpy as np
import pytest

from curvature_transfer import Graph
from curvature_transfer import _kernels as K
from curvature_transfer import _reference as R


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    return Graph.from_edges(np.array(G.edges()), G.number_of_nodes())


def _star_of_cliques(seed):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    hub = 10_000
    for _ in range(4):
        k = int(rng.integers(3, 9))
        base = G.number_of_nodes()
        G.add_edges_from((base + a, base + b) for a in range(k) for b in range(a + 1, k))
        G.add_edge(base, hub)
        G.add_edges_from((base + a, hub + 1 + base + a) for a in range(k))  # pendants
    return G


FAMILIES = {
    "er_sparse": lambda s: nx.gnp_random_graph(150, 0.04, seed=s),
    "er_dense": lambda s: nx.gnp_random_graph(50, 0.5, seed=s),
    "ba_hubs": lambda s: nx.barabasi_albert_graph(200, 3, seed=s),
    "powerlaw_cluster": lambda s: nx.powerlaw_cluster_graph(150, 3, 0.7, seed=s),
    "geometric": lambda s: nx.random_geometric_graph(150, 0.15, seed=s),
    "regular": lambda s: nx.random_regular_graph(4, 60, seed=s),  # degree ties: zero-mass triangle vertices
    "sbm": lambda s: nx.stochastic_block_model([25, 25], [[0.4, 0.05], [0.05, 0.4]], seed=s),
    "star_of_cliques": _star_of_cliques,
}
FIXED = {
    "complete": nx.complete_graph(8), "star": nx.star_graph(7), "single_edge": nx.path_graph(2),
    "path": nx.path_graph(5), "cycle": nx.cycle_graph(7), "petersen": nx.petersen_graph(),
    "complete_bipartite": nx.complete_bipartite_graph(4, 6), "wheel": nx.wheel_graph(8),
    "barbell": nx.barbell_graph(5, 2), "grid": nx.grid_2d_graph(4, 5), "karate": nx.karate_club_graph(),
}


def _all(g):
    return np.arange(g.num_edges, dtype=np.int64)


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_reference_on_random_families(family, seed):
    g = _g(FAMILIES[family](seed))
    new = K.exact_ids(g.indptr, g.indices, g.edges, _all(g), True, True, True, 3)
    ref = R.exact_ids_reference(g.indptr, g.indices, g.edges, _all(g), True, True, True, 3)
    assert np.array_equal(new, ref, equal_nan=True)


@pytest.mark.parametrize("name", sorted(FIXED))
def test_matches_reference_on_fixed_graphs(name):
    g = _g(FIXED[name])
    new = K.exact_ids(g.indptr, g.indices, g.edges, _all(g), True, True, True, 2)
    ref = R.exact_ids_reference(g.indptr, g.indices, g.edges, _all(g), True, True, True, 2)
    assert np.array_equal(new, ref, equal_nan=True)


def test_closed_forms():
    g = _g(nx.complete_graph(6))  # identical lazy laws; non-lazy: move 1/d from j to i
    out = K.exact_ids(g.indptr, g.indices, g.edges, _all(g), True, True, False, 1)
    assert np.allclose(out[:, 0], 1.0) and np.allclose(out[:, 1], 1 - 1 / 5)
    g = _g(nx.path_graph(2))
    out = K.exact_ids(g.indptr, g.indices, g.edges, _all(g), True, True, False, 1)
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0
    g = _g(nx.star_graph(3))  # center-leaf, lazy: W1 = 3/4 -> 1/4
    out = K.exact_ids(g.indptr, g.indices, g.edges, _all(g), True, False, False, 1)
    assert np.allclose(out[:, 0], 0.25)


def test_driver_modes_agree():
    g = _g(nx.powerlaw_cluster_graph(200, 3, 0.5, seed=3))
    ids = _all(g)
    ex = K.exact_ids(g.indptr, g.indices, g.edges, ids, True, True, True, 4)
    assert np.array_equal(K.exact_or_ids(g.indptr, g.indices, g.edges, ids, 4), ex[:, 0])
    assert np.array_equal(K.tlw_lazy_ids(g.indptr, g.indices, g.edges, ids, 4), ex[:, 3])
    assert np.all(ex[:, 3] >= ex[:, 0] - 1e-12)  # TLW is an upper bound on c_OR
    tau, tol = float(np.median(ex[:, 0])), 1e-9
    sc = K.screen_ids(g.indptr, g.indices, g.edges, ids, tau, tol, 4)
    decided = sc[:, 1] + tol < tau
    assert np.array_equal(sc[:, 1], ex[:, 3])
    assert np.all(np.isnan(sc[decided, 0])) and np.array_equal(sc[~decided, 0], ex[~decided, 0])


def test_requested_columns_only():
    g = _g(nx.karate_club_graph())
    out = K.exact_ids(g.indptr, g.indices, g.edges, _all(g), False, True, False, 2)
    assert np.all(np.isnan(out[:, [0, 2, 3]])) and not np.any(np.isnan(out[:, [1, 4]]))


def test_column_blocking_is_exact():
    """Tiny bitset budget: columns are processed in many blocks."""
    g = _g(nx.barabasi_albert_graph(300, 4, seed=5))
    ids = _all(g)
    full = K._run(g.indptr, g.indices, g.edges, ids, 0, True, True, True, 0.0, 0.0, K.N_EXACT, 2)
    blocked = K._run(g.indptr, g.indices, g.edges, ids, 0, True, True, True, 0.0, 0.0, K.N_EXACT, 2, 32)
    assert np.array_equal(full, blocked, equal_nan=True)


def test_chunking_and_subsets_do_not_change_values():
    g = _g(nx.random_geometric_graph(250, 0.12, seed=2))
    ids = _all(g)
    base = K.exact_ids(g.indptr, g.indices, g.edges, ids, True, True, True, 1)
    for nch in (2, 7, 64):
        assert np.array_equal(K.exact_ids(g.indptr, g.indices, g.edges, ids, True, True, True, nch), base,
                              equal_nan=True)
    sub = np.random.default_rng(0).permutation(ids)[: len(ids) // 3]
    assert np.array_equal(K.exact_ids(g.indptr, g.indices, g.edges, sub, True, True, True, 3), base[sub],
                          equal_nan=True)
    assert K.exact_ids(g.indptr, g.indices, g.edges, ids[:0], True, True, True, 3).shape == (0, K.N_EXACT)
