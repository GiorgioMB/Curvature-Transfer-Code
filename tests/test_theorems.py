"""Every bound and structural claim of the paper, checked against exact values."""
import networkx as nx
import numpy as np
import pytest

from conftest import random_graphs
from curvature_transfer import CurvatureEngine
from curvature_transfer import bounds as B

TOL = 1e-9
GRAPHS = random_graphs(4) + [nx.convert_node_labels_to_integers(g) for g in (
    nx.complete_graph(6), nx.hypercube_graph(3), nx.hypercube_graph(5), nx.grid_2d_graph(6, 6),
    nx.karate_club_graph(), nx.petersen_graph(), nx.dodecahedral_graph(), nx.star_graph(7), nx.balanced_tree(3, 3))]


@pytest.fixture(scope="module", params=["paper", "legacy"])
def results(request):
    out = []
    for G in GRAPHS:
        r = CurvatureEngine(G, varpi=request.param).compute_all()
        out.append(r)
    return out


def _all(results, f):
    for r in results:
        assert np.all(f(r))


def test_nonlazy_lower_bounds(results):
    for k in ("jl_lower", "tlw_lower", "matching_lower"):
        _all(results, lambda r: r[k] <= r["c_OR0"] + TOL)


def test_nonlazy_upper_bounds(results):
    for k in ("jl_upper", "envelope_upper_nonlazy", "tlw_dual_nonlazy"):
        _all(results, lambda r: r[k] >= r["c_OR0"] - TOL)


def test_lazy_lower_bounds(results):
    for k in ("jl_lower_lazy", "matching_lower_lazy", "bf_lower", "sharpened_lower"):
        _all(results, lambda r: r[k] <= r["c_OR"] + TOL)
    for r in results:
        ok = ~np.isnan(r["bf_lower_preliminary"])
        assert np.all(r["bf_lower_preliminary"][ok] <= r["c_OR"][ok] + TOL)


def test_lazy_upper_bounds(results):
    for k in ("envelope_upper", "zero_cost_upper", "tlw_dual_lazy", "bf_upper", "sharpened_upper", "theta_at_t"):
        _all(results, lambda r: r[k] >= r["c_OR"] - TOL)


def test_lly_relations(results):
    """Lemma 3.3 (Bourne et al.) and Lemma 3.4 (Topping et al.)."""
    _all(results, lambda r: r["c_BF"] <= r["c_LLY"] + TOL)
    _all(results, lambda r: r["c_OR0"] <= r["c_LLY"] + TOL)
    _all(results, lambda r: r["c_LLY"] <= r["c_OR0"] + 2 / np.maximum(r["deg_i"], r["deg_j"]) + TOL)
    for r in results:  # Remark 4.5: the lift is lossless on equal-degree edges
        eq = r["deg_i"] == r["deg_j"]
        assert np.allclose(r["c_OR"][eq], (1 - 1 / (r["deg_i"][eq] + 1)) * r["c_LLY"][eq])


def test_bf_identity_and_deficits(results):
    """Proposition 5.1."""
    for r in results:
        m = np.minimum(r["deg_i"], r["deg_j"]) >= 2
        A, Bd = r["bf_deficit_A"][m], r["bf_deficit_B"][m]
        zmax = r["triangle"][m] / np.maximum(r["deg_i"], r["deg_j"])[m]
        assert np.allclose(r["c_BF"][m], zmax - A - Bd)
        assert np.all(A >= Bd - TOL)
        assert np.all(r["bf_clipped"][m] <= r["matching_lower"][m] + TOL)
        assert np.all(r["c_BF"][m] - r["c_OR0"][m] <= np.maximum(-A, 0) + np.maximum(-Bd, 0) + TOL)
        sparse = Bd >= 0
        assert np.all(r["c_BF"][m][sparse] <= r["c_OR0"][m][sparse] + TOL)


def test_dominance(results):
    """Remark 5.5 and Proposition 5.7."""
    for r in results:
        ok = ~np.isnan(r["bf_lower_preliminary"])
        assert np.all(r["bf_lower"][ok] >= r["bf_lower_preliminary"][ok] - TOL)
        assert np.all(r["sharpened_lower"] >= np.maximum(r["bf_lower"], r["matching_lower_lazy"]) - TOL)
        assert np.all(r["envelope_upper"] <= r["bf_upper"] + TOL)


def test_reverse_corollaries(results):
    _all(results, lambda r: r["c_BF_lower_from_c_OR"] <= r["c_BF"] + TOL)
    _all(results, lambda r: r["c_BF_upper_from_c_OR"] >= r["c_BF"] - TOL)


def test_exactness_on_complete_graphs_and_hypercubes():
    for G in (nx.complete_graph(8), nx.hypercube_graph(4), nx.hypercube_graph(6)):
        r = CurvatureEngine(nx.convert_node_labels_to_integers(G)).compute_all()
        assert np.allclose(r["bf_lower"], r["c_OR"])


def test_matching_bound_exact_on_trees_and_hypercubes():
    for G in (nx.balanced_tree(3, 3), nx.hypercube_graph(4)):
        r = CurvatureEngine(nx.convert_node_labels_to_integers(G)).compute_all()
        assert np.allclose(r["matching_lower"], r["c_OR0"])


def test_legacy_c4_never_exceeds_paper_c4():
    for G in GRAPHS[:10]:
        a = CurvatureEngine(G).stats()["C4"]
        b = CurvatureEngine(G, varpi="legacy").stats()["C4"]
        assert np.all(b <= a + TOL)


def test_psi_knot_set_attains_the_maximum():
    """Theorem 5.4: the max over the knot set equals a dense-grid max."""
    rng = np.random.default_rng(0)
    di = rng.integers(2, 30, 300).astype(float); dj = rng.integers(2, 30, 300).astype(float)
    zeta = rng.uniform(-2, 1.5, 300)
    psi = B.psi_bf_to_or(zeta, di, dj)
    S, T, _ = B.degree_terms(di, dj)
    b = np.maximum(0, zeta - S)
    tmax = np.minimum(np.minimum(di, dj) - 1, b / T)
    for k in range(300):
        ts = np.linspace(0, max(tmax[k], 0), 2001)
        _, arg = B.psi_bf_to_or(np.full_like(ts, zeta[k]), np.full_like(ts, di[k]), np.full_like(ts, dj[k]), True)
        # evaluate Psi_hat on the grid through the public function's internals
        best = -np.inf
        for t in ts:
            Ai, Aj = (di[k] - 1 - t) / (di[k] + 1), (dj[k] - 1 - t) / (dj[k] + 1)
            Sig = di[k] + dj[k] + 2; dmax = max(di[k], dj[k])
            Bv = dmax * (dmax - 1) / Sig * (b[k] - T[k] * t)
            Dv = (di[k] + dj[k] - 2 - 2 * t) / Sig
            wi, wj = 1 / (di[k] + 1), 1 / (dj[k] + 1)
            C = min(t * abs(wi - wj), Ai + Aj)
            v = -1 + 4 * min(wi, wj) + 2 * abs(wi - wj) + 2 * t * min(wi, wj) + max(0, min(Ai, Aj, Bv, Dv)) + C
            best = max(best, v)
        assert psi[k] >= best - 1e-9
        assert psi[k] <= best + 1e-3  # the grid resolution


def test_moduli_monotone():
    di = np.full(200, 7.0); dj = np.full(200, 12.0); z = np.linspace(-2, 2, 200)
    for f in (B.phi_bf_to_or(z, di, dj), B.psi_bf_to_or(z, di, dj), B.phi_or_to_bf(z, di, dj), B.s0(z, di, dj)):
        assert np.all(np.diff(f) >= -1e-12)
