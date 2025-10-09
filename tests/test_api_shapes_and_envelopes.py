import numpy as np
import torch
from conftest import Data, undirected_edge_index
import pyg_curvature as pc

def test_scalar_vs_vector_args_yield_correct_shapes():
    ei = undirected_edge_index(4, [(0,1),(1,2),(2,3),(3,0)])
    eng = pc.CurvatureEngine(Data(4, ei))
    base = eng.compute_all()

    # Passing a scalar should broadcast to one value per undirected edge
    lb = eng.varphi_BF_to_OR(zeta=0.0)
    ub = eng.psi_BF_to_OR(zeta=0.0)
    assert lb.shape == ub.shape == base["c_OR"].shape

    # Passing a vector should preserve shape
    lb2 = eng.varphi_OR_to_BF(theta=base["c_OR"])
    ub2 = eng.psi_OR_to_BF(theta=base["c_OR"])
    assert lb2.shape == ub2.shape == base["c_BF"].shape

def test_c4_term_is_safe():
    # Static method, edge cases
    assert pc.CurvatureEngine._C4_edge(0, 5) == 0.0
    assert pc.CurvatureEngine._C4_edge(3, 0) == 0.0
    assert abs(pc.CurvatureEngine._C4_edge(3, 6) - 0.5) < 1e-12
