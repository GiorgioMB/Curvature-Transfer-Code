"""Exact curvatures (orcut, through CurvatureEngine) against an LP reference. The solver itself is tested
in orcut (against its independent oracle and linear programming)."""
import networkx as nx
import numpy as np
import pytest

from conftest import small_graphs
from curvature_transfer import CurvatureEngine


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


def test_threads_do_not_change_results():
    G = nx.convert_node_labels_to_integers(nx.random_geometric_graph(120, 0.18, seed=1))
    a = CurvatureEngine(G, n_jobs=1).exact()["c_OR"]
    b = CurvatureEngine(G, n_jobs=-1).exact()["c_OR"]
    assert np.array_equal(a, b)
