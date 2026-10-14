"""Exact curvatures against an LP reference, and the min-cut solver itself."""
import networkx as nx
import numpy as np
import pytest

from conftest import small_graphs
from curvature_transfer import CurvatureEngine
from curvature_transfer import _kernels as K


@pytest.mark.parametrize("G", small_graphs(), ids=lambda g: f"n{g.number_of_nodes()}m{g.number_of_edges()}")
def test_exact_matches_lp(G, ref):
    eng = CurvatureEngine(G)
    ex = eng.exact()
    for e, (i, j) in enumerate(eng.edges):
        for key in ("c_OR", "c_OR0", "c_LLY"):
            assert ex[key][e] == pytest.approx(ref(G, int(i), int(j), key), abs=1e-9)


def test_subset_matches_full():
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(80, 3, seed=0))
    eng = CurvatureEngine(G)
    full = eng.exact(("c_OR",))["c_OR"]
    ids = np.array([5, 0, 17, 3])
    assert np.allclose(eng.exact(("c_OR",), edges=ids)["c_OR"], full[ids])


def test_known_values():
    # complete graph K_n with alpha = 1/n: both measures are uniform on V
    assert np.allclose(CurvatureEngine(nx.complete_graph(7)).exact()["c_OR"], 1.0)
    # hypercube Q_n: LLY curvature 2/n, lazy OR (1 - 1/(n+1)) 2/n, non-lazy 0
    ex = CurvatureEngine(nx.convert_node_labels_to_integers(nx.hypercube_graph(4))).exact()
    assert np.allclose(ex["c_LLY"], 0.5) and np.allclose(ex["c_OR"], 0.4) and np.allclose(ex["c_OR0"], 0.0)
    # long cycles are flat
    assert np.allclose(CurvatureEngine(nx.cycle_graph(9)).exact()["c_LLY"], 0.0)


def test_closure_on_random_metric_problems():
    """w1_closure against linprog on random graph metrics with random masses."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(0)
    for trial in range(40):
        G = nx.gnp_random_graph(9, 0.4, seed=trial)
        if not nx.is_connected(G):
            continue
        d = dict(nx.all_pairs_shortest_path_length(G))
        if max(max(r.values()) for r in d.values()) > 3:
            continue
        D = np.array([[d[a][b] for b in range(9)] for a in range(9)], dtype=np.int8)
        w = rng.integers(-5, 6, 9)
        w[-1] -= w.sum()
        P, N = np.flatnonzero(w > 0), np.flatnonzero(w < 0)
        c = np.array([D[a, b] for a in P for b in N], float)
        A = np.zeros((len(P) + len(N), len(P) * len(N)))
        for x in range(len(P)):
            A[x, x * len(N):(x + 1) * len(N)] = 1
        for y in range(len(N)):
            A[len(P) + y, y::len(N)] = 1
        bvec = np.r_[w[P], -w[N]].astype(float)
        want = linprog(c, A_eq=A, b_eq=bvec, bounds=(0, None), method="highs").fun
        assert K.w1_closure(D.copy(), w.astype(np.int64).copy()) == pytest.approx(want, abs=1e-6)


def test_threads_do_not_change_results():
    G = nx.convert_node_labels_to_integers(nx.random_geometric_graph(120, 0.18, seed=1))
    a = CurvatureEngine(G, n_jobs=1).exact()["c_OR"]
    b = CurvatureEngine(G, n_jobs=-1).exact()["c_OR"]
    assert np.array_equal(a, b)
