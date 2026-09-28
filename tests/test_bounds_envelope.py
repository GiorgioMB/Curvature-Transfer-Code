import numpy as np
import pytest

def assert_bounded(val, lower, upper, name="Value", eidx=None):
    tol = 1e-9
    ctx = f" on edge {eidx}" if eidx is not None else ""
    assert val + tol >= lower, (
        f"LOWER BOUND VIOLATION{ctx}:\n"
        f"  Exact {name}: {val}\n"
        f"  Lower Bound:  {lower}\n"
        f"  Diff:         {lower - val} (Bound exceeded exact value!)"
    )
    assert val - tol <= upper, (
        f"UPPER BOUND VIOLATION{ctx}:\n"
        f"  Exact {name}: {val}\n"
        f"  Upper Bound:  {upper}\n"
        f"  Diff:         {val - upper} (Exact value exceeded bound!)"
    )

@pytest.mark.parametrize("graph_fixture", [
    "make_triangle", 
    "make_square", 
    "make_path3", 
    "make_star4"
])
def test_bounds_BF_to_OR_and_back(graph_fixture, request, new_engine):
    data = request.getfixturevalue(graph_fixture)()
    eng = new_engine(data, n_jobs=1)
    base = eng.compute_all(n_jobs=1)

    c_BF = base["c_BF"]
    c_OR = base["c_OR"]

    bf_bounds = eng.bounds_from_BF(c_BF)
    or_lower = bf_bounds["c_OR_lower_from_c_BF"]
    or_upper = bf_bounds["c_OR_upper_from_c_BF"]
    
    assert or_lower.shape == or_upper.shape == c_OR.shape

    for i in range(len(c_OR)):
        assert_bounded(c_OR[i], or_lower[i], or_upper[i], name="c_OR (Lazy)", eidx=i)

    or_bounds = eng.bounds_from_OR(c_OR)
    bf_lower = or_bounds["c_BF_lower_from_c_OR"]
    bf_upper = or_bounds["c_BF_upper_from_c_OR"]
    
    for i in range(len(c_BF)):
        assert_bounded(c_BF[i], bf_lower[i], bf_upper[i], name="c_BF (Balanced Forman)", eidx=i)

def test_theta_envelope_consistency(make_square, new_engine):
    eng = new_engine(make_square(), n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    
    for eidx in range(base["edges"].shape[0]):
        val, cst, slp = eng.Theta_alpha(eidx, t=base["triangle"][eidx])
        assert np.isclose(val, base["Theta_Const"][eidx] + base["Theta_Slope"][eidx] * base["triangle"][eidx], atol=1e-12)
        assert np.isclose(cst, base["Theta_Const"][eidx], atol=1e-12)
        assert np.isclose(slp, base["Theta_Slope"][eidx], atol=1e-12)

@pytest.mark.parametrize("graph_fixture", ["make_triangle", "make_square", "make_star4"])
def test_smax_replaces_c4_correctly(graph_fixture, request, new_engine):
    data = request.getfixturevalue(graph_fixture)()
    eng = new_engine(data, n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    
    smax_arr = base.get("smax", None)
    c4_arr = base.get("C4", None)
    
    assert smax_arr is not None, "Engine must output exact 'smax' from the new bipartite matcher"
    assert c4_arr is not None, "Engine must output 'C4'"
    
    for i in range(len(smax_arr)):
        # smax must be a valid proportion [0, 1]
        assert 0.0 <= smax_arr[i] <= 1.0 + 1e-12
        
        assert smax_arr[i] <= c4_arr[i] + 1e-12, (
            f"SMAX VIOLATION on edge {i}:\n"
            f"  smax (exact):   {smax_arr[i]}\n"
            f"  C4 (heuristic): {c4_arr[i]}\n"
            f"  smax cannot be larger than the heuristic capacity!"
        )

def test_lazy_transport_envelope_is_upper_bound(make_square, new_engine):
    """Explicitly tests the standalone lazy transport envelope."""
    eng = new_engine(make_square(), n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    
    for eidx in range(base["edges"].shape[0]):
        exact_lazy_or = base["c_OR"][eidx]
        env = eng.lazy_transport_envelope(eidx)
        upper_bound = env["cOR_upper"]
        
        assert exact_lazy_or <= upper_bound + 1e-9, (
            f"LAZY ENVELOPE VIOLATION on edge {eidx}:\n"
            f"  Exact c_OR:  {exact_lazy_or}\n"
            f"  Upper Bound: {upper_bound}\n"
            f"  Diff:        {exact_lazy_or - upper_bound}"
        )
