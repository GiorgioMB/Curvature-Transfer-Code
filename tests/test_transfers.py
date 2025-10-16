import numpy as np
import pytest
from conftest import EPS
from typing import Dict

def test_bf_to_or_bounds_hold(engine):
    base = engine.compute_all()
    res = engine.bounds_from_BF(base["c_BF"])  # compute BF first, then transform
    c_or = base["c_OR"]
    lower = res["c_OR_lower_from_c_BF"]
    upper = res["c_OR_upper_from_c_BF"]

    assert lower.shape == upper.shape == c_or.shape
    assert np.all(lower <= upper + 1e-12), "Lower bound must not exceed upper bound"
    assert np.all(c_or >= lower - EPS), "c_OR must be above varphi_BF->OR lower bound"
    assert np.all(c_or <= upper + EPS), "c_OR must be below psi_BF->OR upper bound"


def test_or_to_bf_bounds_hold(engine):
    base = engine.compute_all()
    res = engine.bounds_from_OR(base["c_OR"])  # compute OR first, then transform
    c_bf = base["c_BF"]
    lower = res["c_BF_lower_from_c_OR"]
    upper = res["c_BF_upper_from_c_OR"]

    assert lower.shape == upper.shape == c_bf.shape
    assert np.all(lower <= upper + 1e-12), "Lower bound must not exceed upper bound"
    assert np.all(c_bf >= lower - EPS), "c_BF must be above varphi_OR->BF lower bound"
    assert np.all(c_bf <= upper + EPS), "c_BF must be below psi_OR->BF upper bound"

def deep_copy_dict_ndarrays(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in d.items()}

def compare_dict_ndarrays(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    equal_nan: bool = False,
    max_diffs: int = 20,
) -> bool:
    ok = True

    # compare keys
    ka, kb = set(a.keys()), set(b.keys())
    only_in_a = sorted(ka - kb)
    only_in_b = sorted(kb - ka)
    if only_in_a:
        print("Keys only in first dict:", only_in_a)
        ok = False
    if only_in_b:
        print("Keys only in second dict:", only_in_b)
        ok = False

    # compare values for shared keys (sorted order)
    for k in sorted(ka & kb):
        va, vb = a[k], b[k]

        if va.shape != vb.shape:
            print(f"[{k}] shape differs: {va.shape} != {vb.shape}")
            ok = False
            continue

        # exact vs tolerant comparison
        if rtol == 0.0 and atol == 0.0 and not equal_nan:
            equal_mask = (va == vb)
        else:
            equal_mask = np.isclose(va, vb, rtol=rtol, atol=atol, equal_nan=equal_nan)

        if not np.all(equal_mask):
            ok = False
            # find and print first `max_diffs` different positions
            diff_idx = np.argwhere(~equal_mask)
            n_show = min(len(diff_idx), max_diffs)
            print(f"[{k}] {len(diff_idx)} differing values (showing first {n_show}):")
            for i in range(n_show):
                idx = tuple(diff_idx[i])
                print(f"  {k}{idx}: {va[idx]} != {vb[idx]}")

    return ok


def test_base_immutability_by_OR_BF_bounds(engine):
    """
    Ensures that passing as parameter base[..] to either bounds_from_BF or bounds_from_OR
    does not mutate it
    """
    base = engine.compute_all()
    base_original = deep_copy_dict_ndarrays(base)
    assert compare_dict_ndarrays(base, base_original)

    engine.bounds_from_BF(base["c_BF"])
    assert compare_dict_ndarrays(engine.compute_all(), base_original)

    engine.bounds_from_OR(base["c_OR"])
    assert compare_dict_ndarrays(engine.compute_all(), base_original)