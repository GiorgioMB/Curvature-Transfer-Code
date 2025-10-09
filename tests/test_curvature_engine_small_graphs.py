import numpy as np
import pytest

def _edges_as_list(arr):
    return [list(x) for x in arr.astype(int).tolist()]

@pytest.mark.parametrize("graph_fixture", ["make_triangle","make_square","make_path3","make_star4"])
def test_compute_all_shapes(graph_fixture, request, new_engine):
    data = request.getfixturevalue(graph_fixture)()
    eng = new_engine(data, n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    M = base["edges"].shape[0]
    # All vectors must have length M
    for key in ["deg_i","deg_j","triangle","Xi","sho_max","C4","c_BF","c_OR","c_OR0","Theta_Const","Theta_Slope"]:
        assert base[key].shape == (M,)

def test_triangle_exact_values(make_triangle, new_engine):
    eng = new_engine(make_triangle(), n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    # Degrees 2 for every edge
    assert np.all(base["deg_i"] == 2) and np.all(base["deg_j"] == 2)
    # One triangle per edge
    assert np.all(base["triangle"] == 1)
    # No 4-cycles -> C4 = 0
    assert np.allclose(base["C4"], 0.0)
    # Balanced Forman: 2/2 + 2/2 - 2 + (2*1/2 + 1/2) = 1.5
    assert np.allclose(base["c_BF"], 1.5, atol=1e-12)
    # Lazy OR: identical lazy measures -> curvature 1
    assert np.allclose(base["c_OR"], 1.0, atol=1e-12)
    # Non-lazy OR0 on triangle equals 0.5 (see analysis)
    assert np.allclose(base["c_OR0"], 0.5, atol=1e-12)

def test_square_characteristics(make_square, new_engine):
    eng = new_engine(make_square(), n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    # C4 cycle: Xi=2, sho_max=2 -> C4=1; triangles 0
    assert np.all(base["triangle"] == 0)
    assert np.allclose(base["C4"], 1.0, atol=1e-12)
    # BF = 0 (degree term) + C4 = 1
    assert np.allclose(base["c_BF"], 1.0, atol=1e-12)
    # Non-lazy OR0 should be 0 (all neighbor distances = 1)
    assert np.allclose(base["c_OR0"], 0.0, atol=1e-12)
    # Lazy OR is bounded: -2 <= c_OR <= 1
    assert np.all(base["c_OR"] <= 1.0 + 1e-12)
    assert np.all(base["c_OR"] >= -2.0 - 1e-12)

def test_degree_one_edges_have_zero_BF(make_path3, make_star4, new_engine):
    for data in (make_path3(), make_star4()):
        eng = new_engine(data, n_jobs=1)
        base = eng.compute_all(n_jobs=1)
        # On both graphs every edge has a degree-1 endpoint
        assert np.allclose(base["c_BF"], 0.0, atol=1e-12)
        # c_OR, c_OR0 are always <= 1
        assert np.all(base["c_OR"] <= 1.0 + 1e-12)
        assert np.all(base["c_OR0"] <= 1.0 + 1e-12)
