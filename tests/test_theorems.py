"""Every bound and structural claim of the paper, checked against exact values."""
import networkx as nx
import numpy as np
import pytest

from conftest import random_graphs
from curvature_transfer import CurvatureEngine
from curvature_transfer import bounds as B

TOL = 1e-9
_NAMED = (nx.complete_graph(6), nx.hypercube_graph(3), nx.hypercube_graph(5), nx.grid_2d_graph(6, 6),
          nx.karate_club_graph(), nx.petersen_graph(), nx.dodecahedral_graph(), nx.heawood_graph(),
          nx.star_graph(7), nx.balanced_tree(3, 3), nx.cycle_graph(7), nx.wheel_graph(8),
          nx.ladder_graph(6), nx.complete_bipartite_graph(3, 5), nx.random_regular_graph(3, 40, seed=1))
GRAPHS = random_graphs(4) + [nx.convert_node_labels_to_integers(g) for g in _NAMED]


@pytest.fixture(scope="module", params=["paper", "legacy"])
def results(request):
    return [CurvatureEngine(G, varpi=request.param).compute_all() for G in GRAPHS]


@pytest.fixture(scope="module")
def paper_results():
    return [(G, CurvatureEngine(G).compute_all()) for G in GRAPHS]


def _all(results, f):
    for r in results:
        assert np.all(f(r))


def _ab(r):
    return np.minimum(r["deg_i"], r["deg_j"]), np.maximum(r["deg_i"], r["deg_j"])


# ------------------------------------------------------------ validity


def test_nonlazy_lower_bounds(results):
    for k in ("jl_lower", "tlw_lower", "matching_lower", "bf_clipped"):
        _all(results, lambda r: ~(r[k] > r["c_OR0"] + TOL))


def test_nonlazy_upper_bounds(results):
    for k in ("jl_upper", "envelope_upper_nonlazy", "tlw_dual_nonlazy"):
        _all(results, lambda r: r[k] >= r["c_OR0"] - TOL)


def test_lazy_lower_bounds(results):
    for k in ("jl_lower_lazy", "matching_lower_lazy", "lazy_lower", "bf_lower", "bf_lower_modulus", "sharpened_lower"):
        _all(results, lambda r: r[k] <= r["c_OR"] + TOL)
    for r in results:
        ok = ~np.isnan(r["bf_lower_preliminary"])
        assert np.all(r["bf_lower_preliminary"][ok] <= r["c_OR"][ok] + TOL)


def test_lazy_upper_bounds(results):
    for k in ("envelope_upper", "zero_cost_upper", "tlw_dual_lazy", "bf_upper", "lazy_upper", "sharpened_upper",
              "coverage_upper"):
        _all(results, lambda r: r[k] >= r["c_OR"] - TOL)


def test_lly_relations(results):
    """Lemmas 3.3-3.4, c_OR-0 <= c_LLY (Section 3), and c_LLY <= q (Proposition 4.1)."""
    _all(results, lambda r: r["c_BF"] <= r["c_LLY"] + TOL)
    _all(results, lambda r: r["c_OR0"] <= r["c_LLY"] + TOL)
    _all(results, lambda r: r["c_LLY"] <= (2 + r["triangle"]) / _ab(r)[1] + TOL)
    for r in results:  # Lemma 3.3 (i) at equal degrees
        eq = r["deg_i"] == r["deg_j"]
        assert np.allclose(r["c_OR"][eq], (1 - 1 / (r["deg_i"][eq] + 1)) * r["c_LLY"][eq])


def test_bf_identity_deficits_and_c4(results):
    """Lemma 3.6 and Proposition 5.1."""
    for r in results:
        a, b = _ab(r)
        m = a >= 2
        assert np.all(r["C4"] <= 2 * r["matching"] / b + TOL)  # Lemma 3.6
        A, Bd = r["bf_deficit_A"][m], r["bf_deficit_B"][m]
        assert np.allclose(r["c_BF"][m], r["triangle"][m] / b[m] - A - Bd)
        assert np.all(A >= Bd - TOL)
        assert np.all(r["bf_clipped"][m] <= r["matching_lower"][m] + TOL)
        assert np.all(r["c_BF"][m] - r["c_OR0"][m] <= np.maximum(-A, 0) + np.maximum(-Bd, 0) + TOL)
        assert np.all(r["c_BF"][m][Bd >= 0] <= r["c_OR0"][m][Bd >= 0] + TOL)


def test_dominance(results):
    """Remark 5.3, Propositions 4.10, 5.5, 5.9, and Corollary 5.8."""
    for r in results:
        a, b = _ab(r)
        di, dj, bf = r["deg_i"], r["deg_j"], r["c_BF"]
        S, T, _ = B.degree_terms(di, dj)
        ok = ~np.isnan(r["bf_lower_preliminary"])
        assert np.all(r["bf_lower_modulus"][ok] >= r["bf_lower_preliminary"][ok] - TOL)
        assert np.all(r["bf_lower"] >= r["bf_lower_modulus"] - TOL)              # phi^Z >= phi
        assert np.all(r["lazy_lower"] >= r["bf_lower"] - TOL)                    # Prop 5.9, lower end
        assert np.all(r["lazy_upper"] <= r["bf_upper"] + TOL)                    # Prop 5.9, upper end
        assert np.all(r["lazy_upper"] <= r["envelope_upper"] + TOL)              # Prop 4.10
        assert np.all(r["lazy_upper"] <= r["coverage_upper"] + TOL)              # Prop 4.10
        assert np.all(r["coverage_upper"] <= r["zero_cost_upper"] + TOL)
        lifted = B.lift_mix(np.maximum(r["matching_lower"], bf), di, dj, r["triangle"])
        assert np.all(r["lazy_lower"] >= lifted - TOL)                          # Remark 5.3
        assert np.all(bf <= S + T * (a - 1) + TOL)                               # Prop 5.5, range
        assert np.all(r["bf_lower_modulus"] >= B.lift_tf(bf, di, dj) - TOL)       # Prop 5.5, comparison
        assert np.all(B.lift_tf(bf, di, dj) >= B.lift(bf, di, dj) - TOL)


def test_deficit_identities():
    """The 'equivalently' forms of Theorems 4.7, 5.7 and Proposition 4.10."""
    rng = np.random.default_rng(1)
    n = 20000
    a = rng.integers(1, 40, n).astype(float); b = a + rng.integers(0, 40, n)
    t = np.floor(rng.uniform(0, 1, n) * a); Ui = a - 1 - t
    m = np.floor(rng.uniform(0, 1, n) * (Ui + 1)); xi_hi = m + np.floor(rng.uniform(0, 1, n) * (b - 1 - t - m + 1))
    wlo, whi = 1 / (a + 1), 1 / (b + 1); D = wlo - whi
    Theta = (3 + t) * whi - wlo
    assert np.allclose(B.lazy_lower(a, b, t, m), Theta - t * D - 2 * (Ui * wlo - m * whi))
    mUU = np.minimum(m * wlo, xi_hi * whi)
    assert np.allclose(B.lazy_upper(a, b, t, m, 0 * a, xi_hi), Theta - (Ui * wlo - mUU))
    zeta = rng.uniform(-2, 2, n)
    S, T, _ = B.degree_terms(a, b)
    bud = np.maximum(zeta - S, 0)
    deficit = np.maximum((a - 1 - t) * wlo - b * (b - 1) * (bud - T * t) / (a + b + 2), 0)
    assert np.allclose(B.psi_hat(t, zeta, a, b), (3 + t) * whi - wlo - deficit)
    # phi is the lazy bound of a virtual edge with (zeta - S)_+/T triangles and no cross edges
    assert np.allclose(B.phi_bf_to_or(zeta, a, b), B.lazy_lower(a, b, bud / T, 0 * a))
    ts = np.clip(np.floor(bud / T + B.INT_TOL), 0, a - 1)
    assert np.allclose(B.phi_int_bf_to_or(zeta, a, b), B.lazy_lower(a, b, ts, b * (bud - T * ts) / 2))


def test_lift_tf_closed_form_and_inverse():
    """Corollary 4.2: the closed form equals max{Lambda, (1-alpha_max)L - Delta(1 - 2/b)}."""
    a, b = np.meshgrid(np.arange(1, 16.0), np.arange(1, 16.0))
    a, b = a[a <= b], b[a <= b]
    for L in np.linspace(-2, 2, 81):
        Lv = np.full_like(a, L)
        _, _, _, amax, D = B.idleness(a, b)
        old = np.maximum(B.lift(Lv, a, b), (1 - amax) * Lv - D * (1 - 2 / b))
        assert np.allclose(B.lift_tf(Lv, a, b), old, atol=1e-14)
        assert np.allclose(B.lift_tf_inverse(B.lift_tf(Lv, a, b), a, b), Lv)


def test_reverse_corollaries(results):
    for r in results:
        a, _ = _ab(r)
        lo = r["c_BF_lower_from_c_OR"]
        assert not np.isnan(lo).any()
        assert np.all(lo <= r["c_BF"] + TOL)
        assert np.all(r["c_BF_upper_from_c_OR"] >= r["c_BF"] - TOL)
        m2 = a >= 2  # Appendix C: never weaker than inverting Lambda^tf through Lemma 3.4
        first = B.psi_or_to_bf(r["c_OR"], r["deg_i"], r["deg_j"])
        assert np.all(first[m2] <= B.psi_or_to_bf_via_lift(r["c_OR"], r["deg_i"], r["deg_j"])[m2] + TOL)


# ------------------------------------------------------------ exactness


def _classes(G, edges):
    """(a) edges on no cycle of length <= 5; (b) equal degrees and a perfect matching
    between U_i and U_j; (c) equal degrees and B_ij itself a perfect matching."""
    ca, cb, cc = [], [], []
    for u, v in edges:
        Nu, Nv = set(G[u]), set(G[v])
        Uu, Uv = Nu - Nv - {v}, Nv - Nu - {u}
        cross = [(x, y) for x in Uu for y in G[x] if y in Uv]
        five = any(set(G[z]) & Uv for x in Uu for z in G[x] if z not in (u, v) and z not in Uu | Uv)
        ca.append(not (Nu & Nv) and not cross and not five)
        eq = G.degree(u) == G.degree(v)
        B_ = nx.Graph(); B_.add_nodes_from((0, x) for x in Uu); B_.add_nodes_from((1, y) for y in Uv)
        B_.add_edges_from(((0, x), (1, y)) for x, y in cross)
        mm = len(nx.bipartite.maximum_matching(B_, top_nodes=[(0, x) for x in Uu])) // 2 if Uu else 0
        cb.append(eq and mm == len(Uu) == len(Uv))
        cc.append(eq and len(Uu) == len(Uv) and all(d == 1 for _, d in B_.degree()))
    return np.array(ca), np.array(cb), np.array(cc)


def test_exactness_classes(paper_results):
    counts = np.zeros(3, int)
    for G, r in paper_results:
        ca, cb, cc = _classes(G, r["edges"])
        counts += [ca.sum(), cb.sum(), cc.sum()]
        ex, ex0 = r["c_OR"], r["c_OR0"]
        for k in ("lazy_lower", "bf_lower_modulus", "bf_lower"):          # Theorems 4.7 and 5.4
            assert np.allclose(r[k][ca], ex[ca])
        assert np.allclose(r["matching_lower"][ca], ex0[ca])           # Theorem 4.3
        for k in ("lazy_lower", "lazy_upper"):                          # Theorem 4.7, Proposition 4.10
            assert np.allclose(r[k][cb], ex[cb])
        assert np.allclose(r["matching_lower"][cb], ex0[cb])
        assert np.allclose(r["bf_lower_modulus"][cc], ex[cc])          # Theorem 5.4
        leaf = np.minimum(r["deg_i"], r["deg_j"]) == 1                 # Corollary 5.8
        for k in ("bf_lower", "bf_upper"):
            assert np.allclose(r[k][leaf], ex[leaf])
    assert counts.min() > 0, counts


def test_bf_interval_is_a_point_on_complete_graphs():
    for n in (3, 5, 8):
        r = CurvatureEngine(nx.complete_graph(n)).compute_all()
        assert np.allclose(r["bf_lower"], 1) and np.allclose(r["bf_upper"], 1) and np.allclose(r["c_OR"], 1)


def test_upper_modulus_not_exact_on_hypercubes():
    r = CurvatureEngine(nx.convert_node_labels_to_integers(nx.hypercube_graph(3))).compute_all()
    assert np.allclose(r["bf_upper"], 0.75) and np.allclose(r["c_OR"], 0.5)


# --------------------------------------------------------- the moduli


def test_psi_candidates_attain_the_integer_maximum():
    """Theorem 5.7: the candidate rule equals the maximum over all admissible integers."""
    rng = np.random.default_rng(0)
    n = 3000
    di = rng.integers(1, 40, n).astype(float); dj = rng.integers(1, 40, n).astype(float)
    zeta = rng.uniform(-2, 2, n)
    psi = B.psi_bf_to_or(zeta, di, dj)
    S, T, _ = B.degree_terms(di, dj)
    tmax = np.minimum(np.minimum(di, dj) - 1, np.maximum(zeta - S, 0) / T)
    for k in range(n):
        ts = np.arange(0, np.floor(tmax[k] + B.INT_TOL) + 1)
        brute = np.max(B.psi_hat(ts, np.full_like(ts, zeta[k]), np.full_like(ts, di[k]), np.full_like(ts, dj[k])))
        assert abs(psi[k] - brute) < 1e-12


def test_crossing_example():
    """At a threshold no edge attains, the two moduli need not be ordered (Section 1.1)."""
    assert np.isclose(B.phi_bf_to_or(49 / 48, 2.0, 2.0), 49 / 72)
    assert np.isclose(B.psi_bf_to_or(49 / 48, 2.0, 2.0), 2 / 3)


def test_moduli_monotone():
    di = np.full(400, 7.0); dj = np.full(400, 12.0); z = np.linspace(-2, 3, 400)
    for f in (B.phi_bf_to_or(z, di, dj), B.psi_bf_to_or(z, di, dj), B.psi_or_to_bf(z, di, dj),
              B.lift_tf(z, di, dj), B.lift_mix(z, di, dj, 2.0)):
        assert np.all(np.diff(f) >= -1e-12)
    lo = B.phi_or_to_bf(z, di, dj)
    ok = ~np.isnan(lo)
    assert np.all(np.diff(lo[ok]) >= -1e-12)


def test_legacy_c4_never_exceeds_paper_c4():
    for G in GRAPHS[:10]:
        a = CurvatureEngine(G).stats()["C4"]
        b = CurvatureEngine(G, varpi="legacy").stats()["C4"]
        assert np.all(b <= a + TOL)
