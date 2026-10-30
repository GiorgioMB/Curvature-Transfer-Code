import networkx as nx
import numpy as np

from curvature_transfer import CurvatureEngine, certify_then_solve, certify_threshold


def _random_intervals(rng, M):
    c = np.round(rng.normal(size=M), 1)  # many ties on purpose
    lo = c - rng.uniform(0, 1, M) * (rng.random(M) < 0.7)
    up = c + rng.uniform(0, 1, M) * (rng.random(M) < 0.7)
    return c, lo, up


def test_certificates_are_sound():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = int(rng.integers(2, 60))
        c, lo, up = _random_intervals(rng, M)
        cin, cout, _ = certify_threshold(lo, up, 0.0)
        assert np.all(c[cin] < 0) and np.all(c[cout] >= 0)


def test_certify_then_solve_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(100):
        M = int(rng.integers(5, 80))
        c, lo, up = _random_intervals(rng, M)
        r = certify_then_solve(lo, up, lambda ids: c[ids], tau=0.0)
        assert np.array_equal(r["answer"], c < 0)


def test_screening_on_a_real_interval():
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(200, 3, seed=0))
    eng = CurvatureEngine(G)
    c = eng.exact(("c_OR",))["c_OR"]
    b = eng.bounds()
    r = certify_then_solve(b["sharpened_lower"], b["sharpened_upper"],
                           lambda ids: eng.exact(("c_OR",), edges=ids)["c_OR"], tau=0.0)
    assert np.array_equal(r["answer"], c < 0) and r["frac_avoided"] > 0.5
