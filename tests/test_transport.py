import numpy as np
import pytest

def test_northwest_corner_basis_size(pc):
    cost = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=float)
    supply = np.array([0.5, 0.5], dtype=float)
    demand = np.array([1/3, 1/3, 1/3], dtype=float)
    X, basis = pc._northwest_corner(supply, demand)
    assert X.shape == cost.shape
    # Basis should have m + n - 1 entries
    assert len(basis) == (cost.shape[0] + cost.shape[1] - 1)

def test_transportation_simplex_small_cases(pc):
    # 1x1
    cost = np.array([[2.5]], dtype=float)
    supply = np.array([1.0], dtype=float)
    demand = np.array([1.0], dtype=float)
    X, val = pc._transportation_simplex(cost, supply, demand)
    assert np.isclose(val, 2.5)
    assert np.isclose(X.sum(), 1.0)

    # 2x2 with zero on diagonal -> optimal value 0
    cost = np.array([[0.0, 2.0],
                     [2.0, 0.0]], dtype=float)
    supply = np.array([0.5, 0.5], dtype=float)
    demand = np.array([0.5, 0.5], dtype=float)
    X, val = pc._transportation_simplex(cost, supply, demand)
    assert np.isclose(val, 0.0, atol=1e-12)
    # Row/col sums
    assert np.allclose(X.sum(axis=1), supply)
    assert np.allclose(X.sum(axis=0), demand)

    # Rectangular case 2x3: easy structure -> check non-negativity & marginal feasibility
    cost = np.array([[1.0, 0.0, 3.0],
                     [0.0, 2.0, 1.0]], dtype=float)
    supply = np.array([0.5, 0.5], dtype=float)
    demand = np.array([1/3, 1/3, 1/3], dtype=float)
    X, val = pc._transportation_simplex(cost, supply, demand)
    assert (X >= -1e-12).all()
    assert np.allclose(X.sum(axis=1), supply, atol=1e-12)
    assert np.allclose(X.sum(axis=0), demand, atol=1e-12)
    # Lower bound: must be >= 0.0
    assert val >= -1e-12

def test_wasserstein1_uniform(pc):
    # Using the same 2x2 case: uniform marginals -> value 0
    cost = np.array([[0.0, 2.0],
                     [2.0, 0.0]], dtype=float)
    supply = np.empty(cost.shape[0], dtype=np.float64)
    demand = np.empty(cost.shape[1], dtype=np.float64)
    w = pc.wasserstein1_uniform(cost, supply, demand)
    assert np.isclose(w, 0.0, atol=1e-12)
