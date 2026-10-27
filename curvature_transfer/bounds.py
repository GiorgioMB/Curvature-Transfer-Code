"""Edgewise bounds, vectorized over edges.

Every function takes numpy arrays of per-edge statistics (degrees, triangle
counts, ...) and returns an array with one value per edge. Names and result
numbers follow the journal version of the paper ("What Balanced Forman
Curvature Certifies About Ollivier--Ricci Curvature").

Notation (edge (i, j), rho = degree, a = rho_min, b = rho_max):
  S  = 2/rho_i + 2/rho_j - 2,   T = 2/b + 1/a,   K = 1 - 1/a - 1/b,
  Zmax = tri/b, Zmin = tri/a,
  C4 = Xi / (varpi_max * b)  (0 if varpi_max = 0),
  sigma = m(i, j) / b        (matching statistic),
  w_u = alpha_u = 1/(rho_u + 1),  w_lo = 1/(a + 1) >= w_hi = 1/(b + 1),
  Delta = w_lo - w_hi,
  L0 = 3/(a+1) + 3/(b+1) - 2,  kappa1 = (1/(a+1) + 2/(b+1))/T,  kappa2 = b/(b+1).

Lemma 4.6 (one-sided residuals): once the mass shared by the two lazy
measures is kept in place, all residual demand sits on the unique neighbors
of the higher-degree endpoint. The lower ends below come from explicit
couplings and the upper ends from explicit 1-Lipschitz potentials.

Triangle counts are integers. Where a bound takes a floor or ceiling of a
quantity that is an integer in exact arithmetic, it is taken in the direction
that can only weaken the bound (``INT_TOL``), so that rounding never produces
an invalid certificate.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "INT_TOL", "degree_terms", "idleness", "modulus_constants",
    "lift", "lift_mix", "lift_tf", "lift_tf_inverse",
    "bf_curvature", "c4_term",
    "jl_lower", "jl_upper", "matching_lower", "tlw_lower",
    "lazy_lower", "lazy_upper", "coverage_upper", "envelope_upper", "zero_cost_upper",
    "phi_bf_to_or", "phi_int_bf_to_or", "psi_hat", "psi_bf_to_or", "phi_bf_to_or_preliminary",
    "bf_deficits", "bf_clipped", "sharpened_lower",
    "t_min", "phi_or_to_bf", "psi_or_to_bf", "psi_or_to_bf_via_lift",
]

INT_TOL = 1e-9


def _f(x):
    return np.asarray(x, dtype=float)


def _ab(di, dj):
    di, dj = _f(di), _f(dj)
    return np.minimum(di, dj), np.maximum(di, dj)


def degree_terms(di, dj):
    """S, T, K for each edge."""
    di, dj = _f(di), _f(dj)
    a, b = _ab(di, dj)
    S = 2.0 / di + 2.0 / dj - 2.0
    T = 2.0 / b + 1.0 / a
    K = 1.0 - 1.0 / a - 1.0 / b
    return S, T, K


def idleness(di, dj):
    """alpha_i, alpha_j, alpha_min, alpha_max, Delta for alpha_u = 1/(rho_u+1)."""
    ai, aj = 1.0 / (_f(di) + 1.0), 1.0 / (_f(dj) + 1.0)
    amin, amax = np.minimum(ai, aj), np.maximum(ai, aj)
    return ai, aj, amin, amax, amax - amin


def modulus_constants(di, dj):
    """a, b, S, T, L0, kappa1, kappa2, w_lo, w_hi, Delta (Theorem 5.4)."""
    a, b = _ab(di, dj)
    S, T, _ = degree_terms(di, dj)
    wlo, whi = 1.0 / (a + 1.0), 1.0 / (b + 1.0)
    L0 = 3.0 * wlo + 3.0 * whi - 2.0
    k1 = (wlo + 2.0 * whi) / T
    k2 = b / (b + 1.0)
    return a, b, S, T, L0, k1, k2, wlo, whi, wlo - whi


# ------------------------------------------------------------- lifts


def lift(L, di, dj):
    """Lambda_ij(L) (Proposition 4.1): the lift through the triangle inequality,
    (1 - alpha_hat(L)) L - Delta with alpha_hat = alpha_min if L >= 0 and
    alpha_max otherwise. Dominated by :func:`lift_tf` and :func:`lift_mix`."""
    L = _f(L)
    _, _, amin, amax, D = idleness(di, dj)
    return np.where(L >= 0, (1.0 - amin) * L, (1.0 - amax) * L) - D


def lift_mix(L, di, dj, tri):
    """Lambda^mix_ij(L) of Proposition 4.1: c_OR >= (1 - alpha_max) L - Delta (1 - q)
    for every L <= c_LLY, with q = (2 + tri)/rho_max."""
    _, _, _, amax, D = idleness(di, dj)
    q = (2.0 + _f(tri)) / np.maximum(_f(di), _f(dj))
    return (1.0 - amax) * _f(L) - D * (1.0 - q)


def lift_tf(L, di, dj):
    """Lambda^tf_ij(L) of Corollary 4.2: (1 - alpha_max) L - Delta (1 - max{L, 2/rho_max})."""
    L = _f(L)
    _, _, _, amax, D = idleness(di, dj)
    b = np.maximum(_f(di), _f(dj))
    return (1.0 - amax) * L - D * (1.0 - np.maximum(L, 2.0 / b))


def lift_tf_inverse(y, di, dj):
    """The inverse of the strictly increasing map Lambda^tf_ij."""
    y = _f(y)
    _, _, amin, amax, D = idleness(di, dj)
    b = np.maximum(_f(di), _f(dj))
    knee = (1.0 - amin) * 2.0 / b - D  # Lambda^tf(2/b)
    return np.where(y >= knee, (y + D) / (1.0 - amin), (y + D * (1.0 - 2.0 / b)) / (1.0 - amax))


# --------------------------------------------------------------- BF


def c4_term(Xi, varpi, di, dj):
    Xi, varpi = _f(Xi), _f(varpi)
    dmax = np.maximum(_f(di), _f(dj))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(varpi > 0, Xi / (varpi * dmax), 0.0)


def bf_curvature(di, dj, tri, C4):
    """Balanced Forman curvature (Topping et al. 2022, Definition 1)."""
    S, T, _ = degree_terms(di, dj)
    val = S + T * _f(tri) + _f(C4)
    return np.where(np.minimum(_f(di), _f(dj)) > 1, val, 0.0)


# ------------------------------------------------------ non-lazy bounds


def jl_lower(di, dj, tri, Sm=0.0):
    """Jost--Liu lower bound on c_OR-0 (Sm=0), or with the matching term
    (Theorem 4.3, eq. JL-aug) when Sm = sigma = m(i,j)/rho_max."""
    _, _, K = degree_terms(di, dj)
    tri = _f(tri)
    dmin, dmax = _ab(di, dj)
    zmax, zmin = tri / dmax, tri / dmin
    Sm = _f(Sm)
    return zmax - np.maximum(K - zmax - Sm, 0.0) - np.maximum(K - zmin - Sm, 0.0)


def matching_lower(di, dj, tri, match):
    """L^square_ij of Theorem 4.3."""
    return jl_lower(di, dj, tri, _f(match) / np.maximum(_f(di), _f(dj)))


def jl_upper(di, dj, tri):
    """Jost--Liu upper bound tri/rho_max on c_OR-0."""
    return _f(tri) / np.maximum(_f(di), _f(dj))


def tlw_lower(di, dj, tri, nUi, nUj):
    """Tian--Lubberts--Weber (2025), Thm 9.2, unit weights, zero idleness.
    (Ported unchanged from the experiments of the preliminary version.)"""
    mx, my = 1.0 / _f(di), 1.0 / _f(dj)
    tri, nUi, nUj = _f(tri), _f(nUi), _f(nUj)
    cc = tri * np.abs(mx - my)
    cross = nUi * mx - my - tri * np.maximum(my - mx, 0.0)
    return 1.0 - nUi * mx - nUj * my - cc - np.abs(cross)


# ------------------------------------------------------- lazy bounds


def lazy_lower(di, dj, tri, match):
    """L^lazy_ij of Theorem 4.7: (3 + tri)/(a + 1) + (3 + 2 tri + 2 m)/(b + 1) - 2.
    Equivalently Theta - tri Delta - 2 R with R = |U_lo| w_lo - m w_hi."""
    a, b = _ab(di, dj)
    t, m = _f(tri), _f(match)
    return (3.0 + t) / (a + 1.0) + (3.0 + 2.0 * t + 2.0 * m) / (b + 1.0) - 2.0


def lazy_upper(di, dj, tri, match, xi_i, xi_j):
    """U^lazy_ij of Proposition 4.10: -1 + 2 m0 + (1 + tri) Delta + m^UU, with
    m0 = (2 + tri) w_hi and m^UU = min{m w_lo, |xi_hi| w_hi}, where xi_hi is the set
    of cross-edge vertices of the higher-degree endpoint (``xi_i`` and ``xi_j`` are
    given in the orientation of ``di`` and ``dj``). For equal degrees m^UU = m w."""
    di, dj = _f(di), _f(dj)
    a, b, _, _, _, _, _, wlo, whi, D = modulus_constants(di, dj)
    t, m = _f(tri), _f(match)
    xi_hi = np.where(di <= dj, _f(xi_j), _f(xi_i))
    mUU = np.minimum(m * wlo, xi_hi * whi)
    return -1.0 + 2.0 * (2.0 + t) * whi + (1.0 + t) * D + mUU


def coverage_upper(di, dj, tri):
    """Theta_ij(tri) = (3 + tri)/(b + 1) - 1/(a + 1) = m0 - Delta, the last
    inequality of Proposition 4.10 (degrees and triangle count only)."""
    a, b = _ab(di, dj)
    return (3.0 + _f(tri)) / (b + 1.0) - 1.0 / (a + 1.0)


def envelope_upper(di, dj, tri, nUi, nUj, Xi, ai=None, aj=None):
    """U^env_ij: Proposition 4.8 with the slacks at their upper ends.
    Holds for any idleness alpha_i, alpha_j in [0, 1); defaults to 1/(rho+1).
    For the lazy measure it is dominated by :func:`lazy_upper`."""
    di, dj, tri, nUi, nUj, Xi = map(_f, (di, dj, tri, nUi, nUj, Xi))
    if ai is None:
        ai, aj = 1.0 / (di + 1.0), 1.0 / (dj + 1.0)
    ai, aj = _f(ai) * np.ones_like(di), _f(aj) * np.ones_like(dj)
    wi, wj = (1.0 - ai) / di, (1.0 - aj) / dj
    Sig = di / (1.0 - ai) + dj / (1.0 - aj)
    zi, zj = np.minimum(ai, wj), np.minimum(aj, wi)
    ends = np.abs(ai - wj) + np.abs(aj - wi)
    mUU = np.minimum(np.minimum(nUi * wi, nUj * wj), Xi / Sig)
    mT = np.minimum(tri * np.abs(wi - wj), nUi * wi + nUj * wj)
    return -1.0 + 2.0 * (zi + zj) + ends + 2.0 * tri * np.minimum(wi, wj) + mUU + mT


def zero_cost_upper(di, dj, tri, ai=None, aj=None):
    """c_OR <= total zero-cost mass (lazy form of the JL upper bound)."""
    di, dj, tri = _f(di), _f(dj), _f(tri)
    if ai is None:
        ai, aj = 1.0 / (di + 1.0), 1.0 / (dj + 1.0)
    wi, wj = (1.0 - ai) / di, (1.0 - aj) / dj
    return np.minimum(ai, wj) + np.minimum(aj, wi) + tri * np.minimum(wi, wj)


# ------------------------------------------------------------- BF -> OR


def phi_bf_to_or(zeta, di, dj):
    """Lower modulus phi_{a,b} of Theorem 5.4: c_BF >= zeta => c_OR >= L0 + kappa1 [zeta - S]_+."""
    _, _, S, _, L0, k1, _, _, _, _ = modulus_constants(di, dj)
    return L0 + k1 * np.maximum(_f(zeta) - S, 0.0)


def phi_int_bf_to_or(cbf, di, dj):
    """Lower end phi^Z_{a,b}(c_BF) of the BF interval (Corollary 5.8), valid at
    zeta = c_BF only: the worst edge with this BF value puts the integer part
    t* = min{a - 1, floor(z/T)}, z = [c_BF - S]_+, on triangles and the rest on
    matched 4-cycles: L0 + kappa1 T t* + kappa2 (z - T t*) >= phi_{a,b}(c_BF).
    t* is rounded up within INT_TOL, which can only lower the bound."""
    a, _, S, T, L0, k1, k2, _, _, _ = modulus_constants(di, dj)
    z = np.maximum(_f(cbf) - S, 0.0)
    ts = np.clip(np.floor(z / T + INT_TOL), 0.0, np.maximum(a - 1.0, 0.0))
    return L0 + k1 * T * ts + k2 * (z - T * ts)


def psi_hat(t, zeta, di, dj):
    """The two-piece envelope Psi_hat(t) of Theorem 5.7 at real t."""
    a, b, S, T, _, _, _, wlo, whi, D = modulus_constants(di, dj)
    t = _f(t)
    bud = np.maximum(_f(zeta) - S, 0.0)
    eta = b * (b - 1.0) / (a + b + 2.0)
    p1 = (2.0 + t) * whi - D
    p2 = -1.0 + 2.0 * (2.0 + t) * whi + (1.0 + t) * D + eta * (bud - T * t)
    return np.minimum(p1, p2)


def _t_max_int(zeta, di, dj):
    a, _, S, T, _, _, _, _, _, _ = modulus_constants(di, dj)
    bud = np.maximum(_f(zeta) - S, 0.0)
    tmax = np.minimum(a - 1.0, bud / T)
    # rounded up within INT_TOL: including one more integer can only raise the maximum
    return np.maximum(np.floor(tmax + INT_TOL), 0.0)


def psi_bf_to_or(zeta, di, dj, return_argmax=False):
    """Upper modulus psi_{a,b} of Theorem 5.7: the maximum of Psi_hat over the integers
    0, 1, ..., floor(tri_max), attained at 0, at floor(tri_max), or at an integer
    adjacent to the point where the two pieces of Psi_hat cross."""
    zeta = _f(zeta)
    di, dj = np.broadcast_arrays(_f(di), _f(dj))
    zeta = np.broadcast_to(zeta, di.shape)
    a, b, S, T, _, _, _, wlo, whi, D = modulus_constants(di, dj)
    bud = np.maximum(zeta - S, 0.0)
    eta = b * (b - 1.0) / (a + b + 2.0)
    tmi = _t_max_int(zeta, di, dj)
    # p2 - p1 = (-1 + 2 w_lo + eta bud) + t (w_lo - eta T)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (1.0 - 2.0 * wlo - eta * bud) / (wlo - eta * T)
    tc = np.where(np.isfinite(tc), tc, 0.0)
    best = np.full(di.shape, -np.inf)
    arg = np.zeros(di.shape)
    for t in (np.zeros_like(tmi), tmi, np.floor(tc), np.floor(tc) + 1.0):
        t = np.clip(t, 0.0, tmi)
        v = psi_hat(t, zeta, di, dj)
        better = v > best
        best = np.where(better, v, best)
        arg = np.where(better, t, arg)
    return (best, arg) if return_argmax else best


def phi_bf_to_or_preliminary(zeta, di, dj, C4):
    """Lower modulus of the preliminary version (arXiv:2603.13535, Thm 4.1):
    the quadrangle-augmented JL bound fed with the triangle floor and the matching
    floor implied by c_BF >= zeta, then lifted. Dominated by phi_bf_to_or.
    NaN on edges with rho_min = 1."""
    z = _f(zeta)
    S, T, K = degree_terms(di, dj)
    C4 = _f(C4)
    dmin, dmax = _ab(di, dj)
    Z = np.maximum(0.0, (z - S - C4) / T)
    zbmax, zbmin = Z / dmax, Z / dmin
    floor = np.maximum(0.5 * C4, 0.5 * np.maximum(0.0, z - S - T * (dmin - 1.0)))
    phi0 = -np.maximum(K - zbmax - floor, 0.0) - np.maximum(K - zbmin - floor, 0.0) + zbmax
    return np.where(dmin >= 2, lift(phi0, di, dj), np.nan)


def bf_deficits(di, dj, tri, C4):
    """BF deficits A^BF, B^BF of Proposition 5.1 (NaN where rho_min = 1)."""
    _, _, K = degree_terms(di, dj)
    dmin, dmax = _ab(di, dj)
    A = K - _f(tri) / dmax - 0.5 * _f(C4)
    B = K - _f(tri) / dmin - 0.5 * _f(C4)
    ok = dmin >= 2
    return np.where(ok, A, np.nan), np.where(ok, B, np.nan)


def bf_clipped(cbf, A, B):
    """L^BF = c_BF - [-A]_+ - [-B]_+  (<= c_OR-0 by Proposition 5.1(ii))."""
    return _f(cbf) - np.maximum(-_f(A), 0.0) - np.maximum(-_f(B), 0.0)


def sharpened_lower(di, dj, tri, match):
    """Lower end of the 2-hop and sharpened intervals: L^lazy (Proposition 5.9)."""
    return lazy_lower(di, dj, tri, match)


# ------------------------------------------------------------- OR -> BF


def t_min(theta, di, dj):
    """t_min(theta) = max{0, ceil((theta + Delta)(b + 1)) - 2} of Corollary C.1
    (the ceiling is rounded down within INT_TOL, which can only weaken the bound)."""
    _, b, _, _, _, _, _, _, _, D = modulus_constants(di, dj)
    return np.maximum(0.0, np.ceil((_f(theta) + D) * (b + 1.0) - INT_TOL) - 2.0)


def phi_or_to_bf(theta, di, dj):
    """Corollary C.1: c_OR >= theta  =>  c_BF >= S + T t_min(theta) if rho_min >= 2;
    NaN where t_min(theta) > rho_min - 1 (no edge with these degrees has c_OR >= theta);
    0 on edges with rho_min = 1, where BF is 0 by definition."""
    a, _, S, T, _, _, _, _, _, _ = modulus_constants(di, dj)
    tm = t_min(theta, di, dj)
    val = np.where(tm > a - 1.0, np.nan, S + T * tm)
    return np.where(a >= 2, val, 0.0)


def psi_or_to_bf(theta, di, dj):
    """Corollary C.2: c_OR <= theta  =>  c_BF <= S + min{T (a - 1), (theta - L0)/kappa1}
    if rho_min >= 2 (the inverse of the lower modulus); 0 on edges with rho_min = 1."""
    a, _, S, T, L0, k1, _, _, _, _ = modulus_constants(di, dj)
    val = S + np.minimum(T * (a - 1.0), (_f(theta) - L0) / k1)
    return np.where(a >= 2, val, 0.0)


def psi_or_to_bf_via_lift(theta, di, dj):
    """The bound obtained by inverting Lambda^tf through Lemma 3.4, c_BF <= (Lambda^tf)^{-1}(theta),
    which Corollary C.2 is never weaker than (Appendix C)."""
    return lift_tf_inverse(theta, di, dj)
