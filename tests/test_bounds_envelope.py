import numpy as np
import pytest

@pytest.mark.parametrize("graph_fixture", ["make_triangle","make_square","make_path3","make_star4"])
def test_bounds_BF_to_OR_and_back(graph_fixture, request, new_engine):
    data = request.getfixturevalue(graph_fixture)()
    eng = new_engine(data, n_jobs=1)
    base = eng.compute_all(n_jobs=1)

    # Given BF -> bounds on OR must contain actual OR
    bf_bounds = eng.bounds_from_BF(base["c_BF"])
    lower, upper = bf_bounds["c_OR_lower_from_c_BF"], bf_bounds["c_OR_upper_from_c_BF"]
    assert lower.shape == upper.shape == base["c_OR"].shape
    assert np.all(lower <= base["c_OR"] + 1e-9), "Lower bound must be <= actual OR"
    assert np.all(upper + 1e-9 >= base["c_OR"]), "Upper bound must be >= actual OR"

    # Given OR -> bounds on BF must contain actual BF
    or_bounds = eng.bounds_from_OR(base["c_OR"])
    bf_lo, bf_hi = or_bounds["c_BF_lower_from_c_OR"], or_bounds["c_BF_upper_from_c_OR"]
    assert np.all(bf_lo <= base["c_BF"] + 1e-9)
    assert np.all(bf_hi + 1e-9 >= base["c_BF"])

def test_theta_envelope_consistency(make_square, new_engine):
    eng = new_engine(make_square(), n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    # Check theta(idx, tri) == Const + Slope * tri
    for eidx in range(base["edges"].shape[0]):
        val, cst, slp = eng.Theta_alpha(eidx, t=base["triangle"][eidx])
        assert np.isclose(val, base["Theta_Const"][eidx] + base["Theta_Slope"][eidx] * base["triangle"][eidx], atol=1e-12)
        assert np.isclose(cst, base["Theta_Const"][eidx], atol=1e-12)
        assert np.isclose(slp, base["Theta_Slope"][eidx], atol=1e-12)
