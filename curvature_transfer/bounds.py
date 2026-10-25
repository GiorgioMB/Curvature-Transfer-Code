"""Edgewise bounds, vectorized over edges.

Every function takes numpy arrays of per-edge statistics (degrees, triangle
counts, ...) and returns an array with one value per edge. Names follow the
journal version of the paper ("What Balanced Forman Curvature Certifies About
Ollivier--Ricci Curvature"); equation and result numbers refer to it.

Notation (edge (i, j), rho = degree):
  S  = 2/rho_i + 2/rho_j - 2,   T = 2/rho_max + 1/rho_min,
  K  = 1 - 1/rho_min - 1/rho_max,
  Zmax = tri/rho_max, Zmin = tri/rho_min,
  C4 = Xi / (varpi_max * rho_max)  (0 if varpi_max = 0),
  Sm = m(i, j) / rho_max            (matching statistic, script S),
  alpha_u = 1/(rho_u + 1), Delta = alpha_max - alpha_min.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "degree_terms", "idleness", "lift", "s0", "bf_curvature", "c4_term",
    "jl_lower", "jl_upper", "matching_lower", "tlw_lower",
    "envelope_upper", "zero_cost_upper", "theta_const_slope",
    "phi_bf_to_or", "psi_bf_to_or", "phi_bf_to_or_preliminary",
    "bf_deficits", "bf_clipped", "sharpened_lower",
    "phi_or_to_bf", "psi_or_to_bf_matching", "psi_or_to_bf",
]

_EPS = 1e-15


def _f(x):
    return np.asarray(x, dtype=float)


def degree_terms(di, dj):
    """S, T, K for each edge."""
    di, dj = _f(di), _f(dj)
    dmin, dmax = np.minimum(di, dj), np.maximum(di, dj)
    S = 2.0 / di + 2.0 / dj - 2.0
    T = 2.0 / dmax + 1.0 / dmin
    K = 1.0 - 1.0 / dmin - 1.0 / dmax
    return S, T, K


def idleness(di, dj):
    """alpha_i, alpha_j, alpha_min, alpha_max, Delta for alpha_u = 1/(rho_u+1)."""
    ai, aj = 1.0 / (_f(di) + 1.0), 1.0 / (_f(dj) + 1.0)
    amin, amax = np.minimum(ai, aj), np.maximum(ai, aj)
    return ai, aj, amin, amax, amax - amin


# ------------------------------------------------------------- lifting


def lift(L, di, dj):
    """Lambda_ij(L) of Corollary 4.3(ii): the lazy lower bound obtained from a
    lower bound L on the non-lazy curvature, or (Proposition 4.4) on the LLY
    curvature: (1 - alpha_hat(L)) L - Delta, alpha_hat = alpha_min if L >= 0."""
    L = _f(L)
    _, _, amin, amax, D = idleness(di, dj)
    return np.where(L >= 0, (1.0 - amin) * L, (1.0 - amax) * L) - D


def s0(theta, di, dj):
    """s_0(theta) of Corollary 4.3(iii): c_OR <= theta implies
    c_LLY <= s_0(theta) (Proposition 4.4(ii)) and hence c_OR-0 <= s_0(theta)."""
    th = _f(theta)
    _, _, amin, amax, D = idleness(di, dj)
    num = th + D
    return num / (1.0 - np.where(num >= 0, amin, amax))


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
    (Theorem 4.8, eq. JL-aug) when Sm = m(i,j)/rho_max."""
    _, _, K = degree_terms(di, dj)
    tri = _f(tri)
    dmin, dmax = np.minimum(_f(di), _f(dj)), np.maximum(_f(di), _f(dj))
    zmax, zmin = tri / dmax, tri / dmin
    Sm = _f(Sm)
    return zmax - np.maximum(K - zmax - Sm, 0.0) - np.maximum(K - zmin - Sm, 0.0)


def matching_lower(di, dj, tri, match):
    """L^square_ij of Theorem 4.8."""
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


# ------------------------------------------------------- lazy envelopes


def envelope_upper(di, dj, tri, nUi, nUj, Xi, ai=None, aj=None):
    """U^env_ij: Proposition 4.6 with the slacks at their upper ends.
    Holds for any idleness alpha_i, alpha_j in [0, 1); defaults to 1/(rho+1)."""
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


def theta_const_slope(di, dj):
    """Const_alpha and Slope_alpha of Proposition C.1 (coverage envelope)."""
    di, dj = _f(di), _f(dj)
    ai, aj, _, _, D = idleness(di, dj)
    Sig = di + dj + 2.0
    z = np.minimum(ai, aj)
    const = -1.0 + 4.0 * z + 2.0 * D + (di + dj - 2.0) / Sig
    slope = ai + aj - 2.0 / Sig
    return const, slope


# ------------------------------------------------------------- BF -> OR


def phi_bf_to_or(zeta, di, dj, S_plus_C4=None):
    """Lower transfer modulus of Theorem 5.3: c_BF >= zeta  =>  c_OR >= Lambda(zeta).
    If ``S_plus_C4`` (= S + C4, valid for rho_min >= 2) is given, the free
    strengthening Lambda(max(zeta, S + C4)) is applied on edges with rho_min >= 2."""
    z = _f(zeta)
    if S_plus_C4 is not None:
        ok = np.minimum(_f(di), _f(dj)) >= 2
        z = np.where(ok, np.maximum(z, _f(S_plus_C4)), z)
    return lift(z, di, dj)


def psi_bf_to_or(zeta, di, dj, return_argmax=False):
    """Upper transfer modulus of Theorem 5.4 (depends on zeta and degrees only):
    the maximum of the piecewise-affine envelope Psi_hat over the knot set K."""
    zeta = _f(zeta)
    di, dj = np.broadcast_arrays(_f(di), _f(dj))
    zeta = np.broadcast_to(zeta, di.shape)
    S, T, _ = degree_terms(di, dj)
    b = np.maximum(0.0, zeta - S)
    Sig = di + dj + 2.0
    wi, wj = 1.0 / (di + 1.0), 1.0 / (dj + 1.0)
    wmin = np.minimum(wi, wj)
    dmax = np.maximum(di, dj)
    sho = dmax * (dmax - 1.0)
    eta = sho / Sig
    z = np.minimum(wi, wj)  # z_i = z_j = min(alpha_i, alpha_j)
    rsum = 2.0 * np.abs(wi - wj)
    base = -1.0 + 4.0 * z + rsum
    t_max = np.minimum(np.minimum(di, dj) - 1.0, np.where(T > 0, b / T, np.inf))

    def Psi(t):
        Ai = (di - 1.0 - t) * wi
        Aj = (dj - 1.0 - t) * wj
        B = eta * (b - T * t)
        Dd = ((di + dj - 2.0) - 2.0 * t) / Sig
        C = np.minimum(t * np.abs(wi - wj), Ai + Aj)
        return base + 2.0 * t * wmin + np.maximum(0.0, np.minimum(np.minimum(Ai, Aj), np.minimum(B, Dd))) + C

    with np.errstate(divide="ignore", invalid="ignore"):
        cands = [
            np.zeros_like(di), t_max,
            (eta * b - wi * (di - 1.0)) / (eta * T - wi),
            (eta * b - wj * (dj - 1.0)) / (eta * T - wj),
            (wj * (dj - 1.0) - wi * (di - 1.0)) / (wj - wi),
            (wi * (di - 1.0) + wj * (dj - 1.0)) / (wi + wj + np.abs(wi - wj)),
            ((di + dj - 2.0) / Sig - wi * (di - 1.0)) / (2.0 / Sig - wi),
            ((di + dj - 2.0) / Sig - wj * (dj - 1.0)) / (2.0 / Sig - wj),
            ((di + dj - 2.0) - sho * b) / (2.0 - sho * T),
        ]
    best = np.full(di.shape, -np.inf)
    arg = np.zeros(di.shape)
    for t in cands:
        t = np.where(np.isfinite(t), t, 0.0)
        t = np.clip(t, 0.0, np.maximum(t_max, 0.0))
        v = Psi(t)
        better = v > best
        best = np.where(better, v, best)
        arg = np.where(better, t, arg)
    return (best, arg) if return_argmax else best


def phi_bf_to_or_preliminary(zeta, di, dj, C4):
    """Lower modulus of the preliminary version (arXiv:2603.13535, Thm 4.1):
    Theorem 4.8 fed with the triangle floor and the matching floor implied by
    c_BF >= zeta, then lifted. Dominated by phi_bf_to_or (Remark 5.5).
    NaN on edges with rho_min = 1."""
    z = _f(zeta)
    S, T, K = degree_terms(di, dj)
    C4 = _f(C4)
    dmin, dmax = np.minimum(_f(di), _f(dj)), np.maximum(_f(di), _f(dj))
    Z = np.maximum(0.0, (z - S - C4) / T)
    zbmax, zbmin = Z / dmax, Z / dmin
    floor = np.maximum(0.5 * C4, 0.5 * np.maximum(0.0, z - S - T * (dmin - 1.0)))
    phi0 = -np.maximum(K - zbmax - floor, 0.0) - np.maximum(K - zbmin - floor, 0.0) + zbmax
    return np.where(dmin >= 2, lift(phi0, di, dj), np.nan)


def bf_deficits(di, dj, tri, C4):
    """BF deficits A^BF, B^BF of Proposition 5.1 (NaN where rho_min = 1)."""
    _, _, K = degree_terms(di, dj)
    dmin, dmax = np.minimum(_f(di), _f(dj)), np.maximum(_f(di), _f(dj))
    A = K - _f(tri) / dmax - 0.5 * _f(C4)
    B = K - _f(tri) / dmin - 0.5 * _f(C4)
    ok = dmin >= 2
    return np.where(ok, A, np.nan), np.where(ok, B, np.nan)


def bf_clipped(cbf, A, B):
    """L^BF = c_BF - [-A]_+ - [-B]_+  (<= c_OR-0 by Proposition 5.1(ii))."""
    return _f(cbf) - np.maximum(-_f(A), 0.0) - np.maximum(-_f(B), 0.0)


def sharpened_lower(Lsq, cbf, di, dj):
    """Lower end of the sharpened interval (Proposition 5.7)."""
    return lift(np.maximum(_f(Lsq), _f(cbf)), di, dj)


# ------------------------------------------------------------- OR -> BF


def phi_or_to_bf(theta, di, dj):
    """Corollary C.2: c_OR >= theta  =>  c_BF >= S + T t_min(theta)
    (0 on edges with rho_min = 1, where BF is 0 by definition)."""
    th = _f(theta)
    S, T, _ = degree_terms(di, dj)
    const, slope = theta_const_slope(di, dj)
    tmin = np.minimum(np.maximum((th - const) / slope, 0.0), np.minimum(_f(di), _f(dj)) - 1.0)
    return np.where(np.minimum(_f(di), _f(dj)) >= 2, S + T * tmin, 0.0)


def psi_or_to_bf_matching(theta, di, dj, C4, match, clip=True):
    """The second term of Corollary C.3 (Theorem 4.4 of the preliminary
    version): S + T u_max(theta) + C4, with u_max the closed-form inversion of
    the quadrangle-augmented JL bound. With ``clip`` the triangle envelope is
    also capped at rho_min - 1, which is valid and never looser."""
    S, T, K = degree_terms(di, dj)
    di, dj = _f(di), _f(dj)
    dmin, dmax = np.minimum(di, dj), np.maximum(di, dj)
    sz = s0(theta, di, dj)
    Kb = np.maximum(K - _f(match) / dmax, 0.0)
    su = Kb * (2.0 * dmin / dmax - 1.0)
    u_zero = np.where(sz >= 0, dmax * sz, 0.0)
    u = np.select(
        [sz < -2.0 * Kb, sz <= su, sz <= Kb],
        [0.0, (sz + 2.0 * Kb) / T, 0.5 * dmax * (sz + Kb)],
        default=dmax * sz,
    )
    u = np.where(Kb > 0, u, u_zero)
    if clip:
        u = np.minimum(u, dmin - 1.0)
    return S + T * u + _f(C4)


def psi_or_to_bf(theta, di, dj, C4, match, return_parts=False):
    """Corollary C.3: c_OR <= theta  =>  c_BF <= min(s_0(theta), psi_matching)."""
    a = s0(theta, di, dj)
    b = psi_or_to_bf_matching(theta, di, dj, C4, match)
    out = np.minimum(a, b)
    return (out, a, b) if return_parts else out
