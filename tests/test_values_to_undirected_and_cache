import numpy as np
import pytest

def test_c_or0_cache(make_triangle, new_engine):
    eng = new_engine(make_triangle(), n_jobs=1)
    base = eng.compute_all(n_jobs=1)
    v1 = eng._get_c_OR0_all(force_recompute=False).copy()
    # Invalidate and recompute
    v2 = eng._get_c_OR0_all(force_recompute=True).copy()
    assert np.allclose(v1, base["c_OR0"])
    assert np.allclose(v1, v2)

def test_values_to_undirected_passthrough(pc, torch_mod):
    # If already undirected length, returns unchanged
    eng_like = type("X",(object,),{})()
    eng_like.edges = [(0,1),(0,2)]
    arr = np.array([0.1, 0.2], dtype=float)
    out = pc.CurvatureEngine._as_edgewise(eng_like, arr, "dummy")
    assert np.allclose(out, arr)
