import numpy as np
import pyg_curvature as pc

def test_wasserstein_uniform_2x2_identity_is_zero():
    C = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    supply = np.empty(C.shape[0], dtype=np.float64)
    demand = np.empty(C.shape[1], dtype=np.float64)
    val = pc.wasserstein1_uniform(C, supply, demand)
    assert abs(val - 0.0) < 1e-12

def test_wasserstein_uniform_shift_invariance():
    # Adding a constant c to all entries should increase the optimal value by c.
    C = np.array([[0.0, 2.0, 3.0],
                  [2.0, 0.0, 1.0]])
    supply = np.empty(C.shape[0], dtype=np.float64)
    demand = np.empty(C.shape[1], dtype=np.float64)
    base = pc.wasserstein1_uniform(C, supply, demand)
    shifted = pc.wasserstein1_uniform(C + 7.0, supply, demand)
    assert abs((shifted - base) - 7.0) < 1e-12

def test_northwest_corner_basis_size():
    supply = np.array([0.5, 0.5])
    demand = np.array([1/3, 1/3, 1/3])
    X, basis = pc._northwest_corner(supply, demand)
    m, n = X.shape
    assert len(basis) == m + n - 1, "Initial basis must be of size m+n-1"
