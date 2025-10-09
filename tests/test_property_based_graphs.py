import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st, settings, HealthCheck

torch = pytest.importorskip("torch")
tg_data = pytest.importorskip("torch_geometric.data")
Data = tg_data.Data

pc = pytest.importorskip("pyg_curvature")

@st.composite
def simple_connected_graphs(draw, min_n=3, max_n=10, p_min=0.15, p_max=0.5):
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    # Spanning tree for connectivity
    parents = [draw(st.integers(min_value=0, max_value=i-1)) for i in range(1, n)]
    edges = set()
    for i in range(1, n):
        u = parents[i-1]; v = i
        a, b = (u, v) if u < v else (v, u)
        if a != b:
            edges.add((a, b))

    p = draw(st.floats(min_value=p_min, max_value=p_max))
    pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    rands = draw(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=len(pairs), max_size=len(pairs)))
    for (u, v), r in zip(pairs, rands):
        if r < p:
            edges.add((u, v))

    if not edges:
        edges.add((0, 1))

    u_list, v_list = [], []
    for (u, v) in sorted(edges):
        u_list += [u, v]
        v_list += [v, u]
    ei = torch.tensor([u_list, v_list], dtype=torch.long)
    return Data(num_nodes=n, edge_index=ei)

@settings(deadline=None, max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(g=simple_connected_graphs())
def test_invariants_and_bounds_hold(g):
    eng = pc.CurvatureEngine(g, n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    M = base["edges"].shape[0]
    for k in ["deg_i","deg_j","triangle","Xi","sho_max","C4","c_BF","c_OR","c_OR0","Theta_Const","Theta_Slope"]:
        assert base[k].shape == (M,), f"{k} has wrong length"
    assert np.all(base["c_OR"]  <= 1.0 + 1e-9)
    assert np.all(base["c_OR0"] <= 1.0 + 1e-9)

    mask_deg1 = np.minimum(base["deg_i"], base["deg_j"]) <= 1 + 1e-12
    if mask_deg1.any():
        assert np.allclose(base["c_BF"][mask_deg1], 0.0, atol=1e-12)

    bf_bounds = eng.bounds_from_BF(base["c_BF"])
    lower, upper = bf_bounds["c_OR_lower_from_c_BF"], bf_bounds["c_OR_upper_from_c_BF"]
    assert np.all(lower <= base["c_OR"]  + 1e-8)
    assert np.all(upper + 1e-8 >= base["c_OR"])

    or_bounds = eng.bounds_from_OR(base["c_OR"])
    bf_lo, bf_hi = or_bounds["c_BF_lower_from_c_OR"], or_bounds["c_BF_upper_from_c_OR"]
    assert np.all(bf_lo <= base["c_BF"] + 1e-8)
    assert np.all(bf_hi + 1e-8 >= base["c_BF"])

    for eidx in range(M):
        tri = float(base["triangle"][eidx])
        val, cst, slp = eng.Theta_alpha(eidx, t=tri)
        assert np.isclose(val, base["Theta_Const"][eidx] + base["Theta_Slope"][eidx]*tri, atol=1e-10)
