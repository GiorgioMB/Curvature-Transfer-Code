"""Published combinatorial bounds on the non-lazy Ollivier-Ricci curvature (mu_u uniform on N(u)), the
baselines of Table 2. With a = min(d_i, d_j), b = max(d_i, d_j), t triangles on (i, j), K = 1 - 1/a - 1/b:

  BM   Bhattacharya and Mukherjee: the lower bound of Jost and Liu with the matching term M/b, M the
       maximum matching of the cross-edge graph between U_i and U_j,
           kappa >= t/b - (K - t/b - M/b)_+ - (K - t/a - M/b)_+,
       and the upper bound t/b of Jost and Liu.
  TLW  Tian, Lubberts and Weber: Theorem 9.2 (unit weights, zero idleness), on both labellings of the edge,
           kappa >= 1 - |U_i|/d_i - |U_j|/d_j - t |1/d_i - 1/d_j| - | |U_i|/d_i - 1/d_j - t (1/d_j - 1/d_i)_+ |,
       and their dual bound kappa <= 1 - max(sum_P r_+ dist(., N), sum_N r_- dist(., P)), which for the
       non-lazy measure is 1 - tau = t/b (every supply vertex is adjacent to i or j, which carry demand, and
       every demand vertex to j or i, which carry supply).

Every value is a rational with denominator a b (``den``), so decisions are exact after ``snap``.
"""
import numpy as np
from numba import njit, prange

NSTAT = 6   # a, b, t, |U_i|, |U_j|, M (with the labelling d_i <= d_j)


@njit(cache=True)
def _augment(x0, Hp, Hy, matchL, matchR, seenY, fromY, queue, stamp):
    head = 0; tail = 0
    queue[tail] = x0; tail += 1
    while head < tail:
        u = queue[head]; head += 1
        for f in range(Hp[u], Hp[u + 1]):
            y = Hy[f]
            if seenY[y] == stamp:
                continue
            seenY[y] = stamp
            fromY[y] = u
            if matchR[y] < 0:
                cy = y
                while True:
                    u2 = fromY[cy]
                    nxt = matchL[u2]
                    matchL[u2] = cy; matchR[cy] = u2
                    if u2 == x0:
                        return True
                    cy = nxt
            queue[tail] = matchR[y]; tail += 1
    return False


@njit(cache=True, parallel=True)
def _stats_run(indptr, indices, eu, ev, nchunks, out):
    n = indptr.shape[0] - 1
    m = eu.shape[0]
    deg = np.empty(n, np.int64)
    for z in range(n):
        deg[z] = indptr[z + 1] - indptr[z]
    dmax = 1; S2 = 1
    for z in range(n):
        if deg[z] > dmax:
            dmax = deg[z]
        s2 = 0
        for f in range(indptr[z], indptr[z + 1]):
            s2 += deg[indices[f]]
        if s2 > S2:
            S2 = s2
    D = dmax + 2; L = S2 + 2
    for c in prange(nchunks):
        lo = c * m // nchunks; hi = (c + 1) * m // nchunks
        stamp = np.zeros(n, np.int64); tag = np.zeros(n, np.int8); lidx = np.zeros(n, np.int64)
        Ui = np.empty(D, np.int64); Hp = np.empty(D + 1, np.int64); Hy = np.empty(L, np.int64)
        matchL = np.empty(D, np.int64); matchR = np.empty(D, np.int64)
        seenY = np.zeros(D, np.int64); fromY = np.empty(D, np.int64); queue = np.empty(D, np.int64)
        st = 0
        for e in range(lo, hi):
            u = eu[e]; v = ev[e]
            if deg[u] <= deg[v]:
                i = u; j = v
            else:
                i = v; j = u
            ep = e + 1
            p = indptr[i]; pe = indptr[i + 1]; q = indptr[j]; qe = indptr[j + 1]
            nx = 0; ny = 0; t = 0
            while p < pe or q < qe:
                if q >= qe or (p < pe and indices[p] < indices[q]):
                    w = indices[p]
                    if w != j:
                        Ui[nx] = w; stamp[w] = ep; tag[w] = 1; lidx[w] = nx; nx += 1
                    p += 1
                elif p >= pe or indices[q] < indices[p]:
                    w = indices[q]
                    if w != i:
                        stamp[w] = ep; tag[w] = 2; lidx[w] = ny; ny += 1
                    q += 1
                else:
                    t += 1; p += 1; q += 1
            hp = 0
            for xx in range(nx):
                Hp[xx] = hp
                x = Ui[xx]
                for f in range(indptr[x], indptr[x + 1]):
                    w = indices[f]
                    if stamp[w] == ep and tag[w] == 2:
                        Hy[hp] = lidx[w]; hp += 1
            Hp[nx] = hp
            for xx in range(nx):
                matchL[xx] = -1
            for yy in range(ny):
                matchR[yy] = -1
            M = 0
            for xx in range(nx):
                if Hp[xx + 1] > Hp[xx]:
                    st += 1
                    if _augment(xx, Hp, Hy, matchL, matchR, seenY, fromY, queue, st):
                        M += 1
            out[e, 0] = deg[i]; out[e, 1] = deg[j]; out[e, 2] = t
            out[e, 3] = nx; out[e, 4] = ny; out[e, 5] = M


def edge_statistics(indptr, indices, E, nchunks=None):
    """(m, 6) int64: a = d_i <= b = d_j, t, |U_i|, |U_j|, the maximum matching M of the cross-edge graph."""
    import numba
    E = np.asarray(E, np.int64)
    out = np.empty((len(E), NSTAT), np.int64)
    _stats_run(indptr, indices, np.ascontiguousarray(E[:, 0]), np.ascontiguousarray(E[:, 1]),
               nchunks or max(1, min(len(E), 8 * numba.get_num_threads())), out)
    return out


def bm_tlw(indptr, indices, E, nchunks=None):
    """The baselines for every edge of E: dict of float arrays bm_lower, tlw_lower, upper (t/b, the upper end of
    both), and den = a b, the denominator of all three."""
    s = edge_statistics(indptr, indices, E, nchunks).astype(np.float64)
    a, b, t, nUi, nUj, M = (s[:, c] for c in range(NSTAT))
    K = 1.0 - 1.0 / a - 1.0 / b
    sig = M / b
    bm = t / b - np.maximum(K - t / b - sig, 0.0) - np.maximum(K - t / a - sig, 0.0)

    def tlw(dx, dy, nx, ny):
        mx, my = 1.0 / dx, 1.0 / dy
        cross = nx * mx - my - t * np.maximum(my - mx, 0.0)
        return 1.0 - nx * mx - ny * my - t * np.abs(mx - my) - np.abs(cross)

    tl = np.maximum(tlw(a, b, nUi, nUj), tlw(b, a, nUj, nUi))
    return {"bm_lower": bm, "tlw_lower": tl, "upper": t / b, "den": (s[:, 0] * s[:, 1]).astype(np.int64),
            "stats": s.astype(np.int64)}


def snap(x, den, tol=1e-6):
    """Integer numerators of values x known to be multiples of 1/den (exact decisions); raises if not."""
    y = np.asarray(x, np.float64) * np.asarray(den, np.float64)
    r = np.rint(y)
    bad = np.abs(y - r) > tol
    if np.any(bad):
        raise ValueError(f"{int(bad.sum())} values are not on the grid 1/den (max deviation {np.max(np.abs(y - r)):.2e})")
    return r.astype(np.int64)
