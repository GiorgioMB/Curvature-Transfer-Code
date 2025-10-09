
import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, settings, assume, HealthCheck

pc = pytest.importorskip("pyg_curvature")

def _uniform(m, n):
    return np.full(m, 1.0/m, dtype=float), np.full(n, 1.0/n, dtype=float)

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(n=st.integers(min_value=1, max_value=4))
def test_zero_diagonal_cost_yields_zero(n):
    C = np.full((n, n), 1.0, dtype=float)
    np.fill_diagonal(C, 0.0)
    s, d = _uniform(n, n)
    X, val = pc._transportation_simplex(C, s, d)
    assert np.isclose(val, 0.0, atol=1e-12)
    assert np.allclose(X.sum(axis=1), s)
    assert np.allclose(X.sum(axis=0), d)

@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(m=st.integers(min_value=1, max_value=3),
       n=st.integers(min_value=1, max_value=5))
def test_rectangular_transport_feasibility(m, n):
    C = np.arange(m*n, dtype=float).reshape(m, n) % 7
    s, d = _uniform(m, n)
    X, val = pc._transportation_simplex(C, s, d)
    assert (X >= -1e-12).all()
    assert np.allclose(X.sum(axis=1), s, atol=1e-12)
    assert np.allclose(X.sum(axis=0), d, atol=1e-12)
    assert val >= -1e-9
    assert val <= C.max() + 1e-9

@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(m=st.integers(min_value=1, max_value=4),
       n=st.integers(min_value=1, max_value=4))
def test_wasserstein_uniform_bounds(m, n):
    C = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            C[i, j] = abs(i - j)
    w = pc.wasserstein1_uniform(C, n_left=m, n_right=n)
    assert w >= -1e-12
    assert w <= C.max() + 1e-12
