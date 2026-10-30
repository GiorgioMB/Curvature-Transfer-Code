"""The production 2-hop kernel (``_kernels._stats_edge``) against its oracle
(``edge_stats_one``), and its TLW certificate against the certificate that the
exact solver computes on its local adjacency and coverage structure."""
import networkx as nx
import numpy as np
import pytest

from curvature_transfer import Graph
from curvature_transfer import _kernels as K


def _g(G):
    G = nx.convert_node_labels_to_integers(nx.Graph(G))
    G.remove_edges_from(nx.selfloop_edges(G))
    return Graph.from_edges(np.array(G.edges()).reshape(-1, 2), G.number_of_nodes())


def _dense_cross(k=40, extra=6, seed=0):
    """Edge (0, 1) with U_0 and U_1 of size k joined completely (k^2 cross edges,
    far beyond the initial cross-edge buffer), plus pendant paths and noise."""
    rng = np.random.default_rng(seed)
    A = list(range(2, 2 + k))
    Bv = list(range(2 + k, 2 + 2 * k))
    E = [(0, 1)] + [(0, a) for a in A] + [(1, b) for b in Bv] + [(a, b) for a in A for b in Bv]
    T = list(range(2 + 2 * k, 2 + 2 * k + extra))  # triangle vertices
    E += [(0, t) for t in T] + [(1, t) for t in T]
    E += [(int(rng.choice(A)), t) for t in T[: extra // 2]]
    nxt = 2 + 2 * k + extra
    for a in A[:5]:
        E += [(a, nxt), (nxt, nxt + 1)]
        nxt += 2
    return nx.Graph(E)


GRAPHS = {
    "er_sparse": lambda: nx.gnp_random_graph(200, 0.03, seed=1),
    "er_dense": lambda: nx.gnp_random_graph(60, 0.45, seed=2),
    "ba": lambda: nx.barabasi_albert_graph(300, 3, seed=3),
    "ba_dense": lambda: nx.barabasi_albert_graph(150, 12, seed=4),
    "plc": lambda: nx.powerlaw_cluster_graph(200, 4, 0.6, seed=5),
    "rgg": lambda: nx.random_geometric_graph(200, 0.13, seed=6),
    "regular": lambda: nx.random_regular_graph(5, 60, seed=7),  # degree ties everywhere
    "ws": lambda: nx.watts_strogatz_graph(120, 6, 0.2, seed=8),
    "sbm": lambda: nx.stochastic_block_model([30, 30], [[0.3, 0.05], [0.05, 0.3]], seed=9),
    "dense_cross": _dense_cross,
    "star": lambda: nx.star_graph(9),
    "complete": lambda: nx.complete_graph(9),
    "complete_bipartite": lambda: nx.complete_bipartite_graph(4, 7),
    "petersen": nx.petersen_graph,
    "grid": lambda: nx.grid_2d_graph(5, 6),
    "barbell": lambda: nx.barbell_graph(5, 2),
    "single_edge": lambda: nx.path_graph(2),
    "karate": nx.karate_club_graph,
}


@pytest.fixture(scope="module", params=sorted(GRAPHS))
def case(request):
    g = _g(GRAPHS[request.param]())
    ids = np.arange(g.num_edges, dtype=np.int64)
    oracle = {lg: np.array([K.edge_stats_one(g.indptr, g.indices, a, b, lg, True) for a, b in g.edges])
              for lg in (False, True)}
    ex = K.exact_ids(g.indptr, g.indices, g.edges, ids, True, False, False, 3)
    return g, oracle, ex


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("tlw", [False, True])
def test_statistics_equal_the_oracle(case, legacy, tlw):
    g, oracle, _ = case
    new = K.edge_stats_all(g.indptr, g.indices, g.edges, legacy, True, tlw)
    assert new.shape == (g.num_edges, K.N_STATS)
    assert np.array_equal(new[:, :8], oracle[legacy][:, :8])
    if not tlw:
        assert np.all(new[:, 8] == 0)


def test_matching_can_be_skipped(case):
    g, oracle, _ = case
    new = K.edge_stats_all(g.indptr, g.indices, g.edges, False, False, False)
    keep = [0, 1, 2, 3, 4, 5, 7]
    assert np.array_equal(new[:, keep], oracle[False][:, keep]) and np.all(new[:, 6] == 0)


def test_tlw_equals_the_exact_solver_certificate(case):
    g, _, ex = case
    ids = np.arange(g.num_edges, dtype=np.int64)
    fast = K.tlw_lazy_ids(g.indptr, g.indices, g.edges, ids, 4)
    assert np.array_equal(fast, ex[:, 3])
    assert np.array_equal(K.tlw_lazy_ids_local(g.indptr, g.indices, g.edges, ids, 4), ex[:, 3])
    assert np.all(fast >= ex[:, 0] - 1e-12)  # an upper bound on c_OR
    sub = np.random.default_rng(0).permutation(ids)[: max(1, len(ids) // 3)]
    assert np.array_equal(K.tlw_lazy_ids(g.indptr, g.indices, g.edges, sub, 4), ex[sub, 3])
    raw = K.edge_stats_all(g.indptr, g.indices, g.edges, False, True, True)
    assert np.array_equal(K.tlw_lazy_from_stats(raw, g.deg[g.edges[:, 0]], g.deg[g.edges[:, 1]]), ex[:, 3])


@pytest.mark.parametrize("nchunks,block", [(2, 1), (5, 3), (7, 64)])
def test_block_cyclic_chunks_do_not_change_values(case, nchunks, block):
    """More chunks than threads (the public wrapper clamps to one per thread),
    small blocks: every edge is computed exactly once, with the same values."""
    g, _, _ = case
    base = K._stats_run(g.indptr, g.indices, g.edges, True, True, True, 1, 64)
    assert np.array_equal(K._stats_run(g.indptr, g.indices, g.edges, True, True, True, nchunks, block), base)


def test_empty_and_reordered_inputs(case):
    g, oracle, _ = case
    assert K.edge_stats_all(g.indptr, g.indices, g.edges[:0], False).shape == (0, K.N_STATS)
    p = np.random.default_rng(1).permutation(g.num_edges)
    rev = g.edges[p][:, ::-1]  # reversed endpoints: oriented columns swap accordingly
    new = K.edge_stats_all(g.indptr, g.indices, rev, False, True, True)
    ref = oracle[False][p]
    assert np.array_equal(new[:, [0, 3, 4, 6]], ref[:, [0, 3, 4, 6]])
    assert np.array_equal(new[:, 1], ref[:, 2]) and np.array_equal(new[:, 2], ref[:, 1])
    assert np.array_equal(new[:, 7], ref[:, 3] - ref[:, 7])
    fwd = K.edge_stats_all(g.indptr, g.indices, g.edges[p], False, True, True)
    assert np.array_equal(new[:, 8], fwd[:, 8])  # the certificate is symmetric


def test_dense_cross_graph_grows_the_buffer():
    g = _g(_dense_cross(k=40))
    e = g.edge_id(0, 1)
    row = K.edge_stats_all(g.indptr, g.indices, g.edges[e:e + 1], False, True, True)[0]
    assert row[3] == 80 and row[6] == 40  # every vertex of U_0, U_1 has a cross edge; perfect matching
    assert np.array_equal(row[:8], K.edge_stats_one(g.indptr, g.indices, 0, 1, False, True)[:8])


def test_exact_driver_block_cyclic_chunks_do_not_change_values():
    """The exact driver with more chunks than threads (the wrappers clamp to one
    per thread): blocks dealt cyclically, every position computed once."""
    g = _g(nx.barabasi_albert_graph(250, 4, seed=11))
    ids = np.random.default_rng(2).permutation(g.num_edges).astype(np.int64)
    base = K._run(g.indptr, g.indices, g.edges, ids, 0, True, True, True, 0.0, 0.0, K.N_EXACT, 1)
    for nch in (2, 5, 37):
        out = K._run(g.indptr, g.indices, g.edges, ids, 0, True, True, True, 0.0, 0.0, K.N_EXACT, nch)
        assert np.array_equal(out, base, equal_nan=True)
