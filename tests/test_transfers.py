import numpy as np
import pytest
from conftest import EPS

def test_bf_to_or_bounds_hold(engine):
    base = engine.compute_all()
    res = engine.bounds_from_BF()  # compute BF first, then transform
    c_or = base["c_OR"]
    lower = res["c_OR_lower_from_c_BF"]
    upper = res["c_OR_upper_from_c_BF"]

    assert lower.shape == upper.shape == c_or.shape
    assert np.all(lower <= upper + 1e-12), "Lower bound must not exceed upper bound"
    assert np.all(c_or >= lower - EPS), "c_OR must be above varphi_BF->OR lower bound"
    assert np.all(c_or <= upper + EPS), "c_OR must be below psi_BF->OR upper bound"


def test_or_to_bf_bounds_hold(engine):
    base = engine.compute_all()
    res = engine.bounds_from_OR()  # compute OR first, then transform
    c_bf = base["c_BF"]
    lower = res["c_BF_lower_from_c_OR"]
    upper = res["c_BF_upper_from_c_OR"]

    assert lower.shape == upper.shape == c_bf.shape
    assert np.all(lower <= upper + 1e-12), "Lower bound must not exceed upper bound"
    assert np.all(c_bf >= lower - EPS), "c_BF must be above varphi_OR->BF lower bound"
    assert np.all(c_bf <= upper + EPS), "c_BF must be below psi_OR->BF upper bound"
