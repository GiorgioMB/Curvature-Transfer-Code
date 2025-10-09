import numpy as np
import pytest

from conftest import undirected_edge_index, Data, EPS
import pyg_curvature as pc

@pytest.mark.parametrize("edges, n, expected_bf, expected_or, expected_or0", [
    ([(0,1),(1,2),(0,2)], 3, 1.5, 1.0, 1/2),        # K3
    ([(0,1),(1,2),(2,3),(3,0)], 4, 1.0, 2/3, None), # C4
    ([(0,1),(1,2),(2,3)], 4, 0.0, 1/3, None),       # P4
])

def test_motif_values(edges, n, expected_bf, expected_or, expected_or0):
    import torch
    ei = undirected_edge_index(n, edges)
    eng = pc.CurvatureEngine(Data(n, ei))
    base = eng.compute_all(n_jobs=1)

    # Helper to average per-edge values on the undirected set
    bf = float(np.mean(base["c_BF"]))
    orv = float(np.mean(base["c_OR"]))
    if expected_bf is not None:
        assert abs(bf - expected_bf) < 5e-12
    if expected_or is not None:
        assert abs(orv - expected_or) < 5e-12

    if expected_or0 is not None:
        assert abs(float(np.mean(base["c_OR0"])) - expected_or0) < 5e-12

def test_leaf_edges_bf_is_zero():
    # Star on 5 nodes: all leaf edges should have c_BF = 0 by convention
    import torch
    edges = [(0,1),(0,2),(0,3),(0,4)]
    ei = undirected_edge_index(5, edges)
    eng = pc.CurvatureEngine(Data(5, ei))
    bf = eng.compute_all(n_jobs=1)["c_BF"]
    assert np.allclose(bf, 0.0, atol=EPS)
