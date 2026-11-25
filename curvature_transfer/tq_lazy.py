"""Triangle-quadrangle certificate (Theorem 5.3) for the lazy measure, in exact integers.

Per edge (i, j) with a = d_i <= b = d_j, the masses are scaled by L = (a+1)(b+1): w_i -> b+1, w_j -> a+1,
Delta = w_i - w_j -> b-a. Residual (Lemma 4.1): supply Delta on S = {i, j} u C and w_i on U_i, demand w_j on
U_j, tau = |U_j| w_j. Only the building blocks are read: the lists of i, j, U_i, C and the degrees of U_j.

  lower end  kappa >= 1 - W_up / L, W_up the cost of the greedy plan of Theorem 5.3 (iv): cross edges
             (cost 1); common neighbours not adjacent to all of U_j, along their edges into U_j (cost 1);
             detours of length two through C and U_i (cost 2); then j and the common neighbours adjacent to
             all of U_j (cost 1), the rest of S (cost 2), the rest of U_i (cost 3).
  upper end  kappa <= 1 - W_lo / L, W_lo = 2 tau - eta' + D3 with eta' = Delta (the term at i and j, since
             F_ij = 0 for the lazy measure) plus, over the connected components (breadth-first search) of the
             cross edges and the edges from C into U_j, the smaller of the capped supply and capped demand.
Every decision compares integers.
"""
import numpy as np
from numba import njit, prange

NOUT = 3    # L, W_up, W_lo


@njit(cache=True)
def _lazy_edge(indptr, indices, deg, u, v, ep, stamp, tag, lidx, Ui, Uj, Cc, Hp, Hy, UUp, UUx, XCp, XCk,
               KYp, KYy, KXp, KXx, cur, dHi, dHj, dCU, dCj, nout, resx, resy, resk, order, cnt, badY,
               gptr, gadj, gdeg, comp, queue, capv, res):
    if deg[u] <= deg[v]:
        i = u; j = v
    else:
        i = v; j = u
    a = deg[i]; b = deg[j]
    L = (a + 1) * (b + 1); wi = b + 1; wj = a + 1; D = b - a
    stamp[i] = ep; tag[i] = 4; stamp[j] = ep; tag[j] = 5
    # ---------- merge: U_i, U_j, C
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
                Uj[ny] = w; stamp[w] = ep; tag[w] = 2; lidx[w] = ny; ny += 1
            q += 1
        else:
            w = indices[p]
            Cc[t] = w; stamp[w] = ep; tag[w] = 3; lidx[w] = t; t += 1
            p += 1; q += 1
    # ---------- scan A: lists of U_i
    hp = 0; up = 0; cp = 0
    for xx in range(nx):
        x = Ui[xx]
        Hp[xx] = hp; UUp[xx] = up; XCp[xx] = cp
        dHi[xx] = 0; nout[xx] = 0
        for f in range(indptr[x], indptr[x + 1]):
            w = indices[f]
            if stamp[w] == ep:
                tg = tag[w]
                if tg == 2:
                    Hy[hp] = lidx[w]; hp += 1; dHi[xx] += 1
                elif tg == 1:
                    UUx[up] = lidx[w]; up += 1
                elif tg == 3:
                    XCk[cp] = lidx[w]; cp += 1
            else:
                nout[xx] += 1
    Hp[nx] = hp; UUp[nx] = up; XCp[nx] = cp
    for yy in range(ny):
        dHj[yy] = 0; dCj[yy] = 0; badY[yy] = 0
    for f in range(hp):
        dHj[Hy[f]] += 1
    # ---------- scan B: lists of C
    kp = 0
    for kk in range(t):
        k = Cc[kk]
        KYp[kk] = kp; dCU[kk] = 0
        for f in range(indptr[k], indptr[k + 1]):
            w = indices[f]
            if stamp[w] == ep and tag[w] == 2:
                KYy[kp] = lidx[w]; kp += 1; dCU[kk] += 1; dCj[lidx[w]] += 1
    KYp[t] = kp
    for kk in range(t + 1):
        KXp[kk] = 0
    for f in range(cp):
        KXp[XCk[f] + 1] += 1
    for kk in range(t):
        KXp[kk + 1] += KXp[kk]
    for kk in range(t):
        cur[kk] = KXp[kk]
    for xx in range(nx):
        for f in range(XCp[xx], XCp[xx + 1]):
            kk = XCk[f]
            KXx[cur[kk]] = xx; cur[kk] += 1
    tau = ny * wj
    # ---------- lower end: the greedy plan
    for xx in range(nx):
        resx[xx] = wi
    for yy in range(ny):
        resy[yy] = wj
    for kk in range(t):
        resk[kk] = D
    cost = 0
    # cross edges, vertices of small H-degree first (cost 1)
    for z in range(ny + 2):
        cnt[z] = 0
    for xx in range(nx):
        cnt[dHi[xx]] += 1
    acc = 0
    for z in range(ny + 1):
        c0 = cnt[z]; cnt[z] = acc; acc += c0
    for xx in range(nx):
        order[cnt[dHi[xx]]] = xx; cnt[dHi[xx]] += 1
    for idx in range(nx):
        xx = order[idx]
        for f in range(Hp[xx], Hp[xx + 1]):
            yy = Hy[f]
            s = resx[xx] if resx[xx] < resy[yy] else resy[yy]
            if s > 0:
                resx[xx] -= s; resy[yy] -= s; cost += s
                if resx[xx] == 0:
                    break
    # common neighbours with fallback 2 (not adjacent to all of U_j), along their edges (cost 1)
    if D > 0:
        for kk in range(t):
            if dCU[kk] == ny:
                continue
            for f in range(KYp[kk], KYp[kk + 1]):
                yy = KYy[f]
                s = resk[kk] if resk[kk] < resy[yy] else resy[yy]
                if s > 0:
                    resk[kk] -= s; resy[yy] -= s; cost += s
                    if resk[kk] == 0:
                        break
    # detours of length two through a common neighbour or a vertex of U_i (cost 2)
    for kk in range(t):
        pa = KXp[kk]; pb = KXp[kk + 1]; qa = KYp[kk]; qb = KYp[kk + 1]
        while pa < pb and qa < qb:
            xx = KXx[pa]; yy = KYy[qa]
            s = resx[xx] if resx[xx] < resy[yy] else resy[yy]
            if s > 0:
                resx[xx] -= s; resy[yy] -= s; cost += 2 * s
            if resx[xx] == 0:
                pa += 1
            if resy[yy] == 0:
                qa += 1
    for xp in range(nx):
        if dHi[xp] == 0:
            continue
        pa = UUp[xp]; pb = UUp[xp + 1]; qa = Hp[xp]; qb = Hp[xp + 1]
        while pa < pb and qa < qb:
            xx = UUx[pa]; yy = Hy[qa]
            s = resx[xx] if resx[xx] < resy[yy] else resy[yy]
            if s > 0:
                resx[xx] -= s; resy[yy] -= s; cost += 2 * s
            if resx[xx] == 0:
                pa += 1
            if resy[yy] == 0:
                qa += 1
    # what remains: fallback-1 suppliers at cost 1, the rest of S at cost 2, the rest of U_i at cost 3
    rem_y = 0
    for yy in range(ny):
        rem_y += resy[yy]
    s1 = D
    if D > 0:
        for kk in range(t):
            if dCU[kk] == ny:
                s1 += resk[kk]
    if s1 > rem_y:
        s1 = rem_y
    cost += s1; rem_y -= s1
    rest_S = D + D - s1
    for kk in range(t):
        rest_S += resk[kk]
    if rest_S > rem_y:
        rest_S = rem_y
    cost += 2 * rest_S; rem_y -= rest_S
    cost += 3 * rem_y
    W_up = cost
    # ---------- upper end: eta' and D3
    eta = D               # the term at i and j: F_ij = 0, O_i = 0, O_j = tau >= Delta
    NN = nx + t + ny
    for z in range(NN + 1):
        gdeg[z] = 0
    for xx in range(nx):
        for f in range(Hp[xx], Hp[xx + 1]):
            gdeg[xx] += 1; gdeg[nx + t + Hy[f]] += 1
    if D > 0:
        for kk in range(t):
            for f in range(KYp[kk], KYp[kk + 1]):
                gdeg[nx + kk] += 1; gdeg[nx + t + KYy[f]] += 1
    gptr[0] = 0
    for z in range(NN):
        gptr[z + 1] = gptr[z] + gdeg[z]
    for z in range(NN):
        cur[z] = gptr[z]
    for xx in range(nx):
        for f in range(Hp[xx], Hp[xx + 1]):
            o = nx + t + Hy[f]
            gadj[cur[xx]] = o; cur[xx] += 1; gadj[cur[o]] = xx; cur[o] += 1
    if D > 0:
        for kk in range(t):
            me = nx + kk
            for f in range(KYp[kk], KYp[kk + 1]):
                o = nx + t + KYy[f]
                gadj[cur[me]] = o; cur[me] += 1; gadj[cur[o]] = me; cur[o] += 1
    # capped supply of x and k, capped demand of y (all neighbours, j included)
    for xx in range(nx):
        c = dHi[xx] * wj
        capv[xx] = wi if wi < c else c
    for kk in range(t):
        c = dCU[kk] * wj
        capv[nx + kk] = D if D < c else c
    for yy in range(ny):
        c = dHj[yy] * wi + dCj[yy] * D + D
        capv[nx + t + yy] = wj if wj < c else c
    for z in range(NN):
        comp[z] = -1
    for z0 in range(NN):
        if comp[z0] >= 0 or gptr[z0 + 1] == gptr[z0]:
            continue
        head = 0; tail = 0; queue[tail] = z0; tail += 1; comp[z0] = z0
        sS = 0; sD = 0
        while head < tail:
            z = queue[head]; head += 1
            if z < nx + t:
                sS += capv[z]
            else:
                sD += capv[z]
            for f in range(gptr[z], gptr[z + 1]):
                o = gadj[f]
                if comp[o] < 0:
                    comp[o] = z0; queue[tail] = o; tail += 1
        eta += sS if sS < sD else sD
    # certified distance-3 mass
    nX3 = 0
    for xx in range(nx):
        if dHi[xx] == 0 and nout[xx] == 0:
            ok = True
            for f in range(XCp[xx], XCp[xx + 1]):
                if dCU[XCk[f]] > 0:
                    ok = False
                    break
            if ok:
                for f in range(UUp[xx], UUp[xx + 1]):
                    if dHi[UUx[f]] > 0:
                        ok = False
                        break
            if ok:
                nX3 += 1
    for kk in range(t):
        if KXp[kk + 1] > KXp[kk]:
            for f in range(KYp[kk], KYp[kk + 1]):
                badY[KYy[f]] = 1
    nY3 = 0
    for yy in range(ny):
        if dHj[yy] == 0 and badY[yy] == 0 and deg[Uj[yy]] == 1 + dCj[yy]:
            nY3 += 1
    D3 = nX3 * wi
    y3 = nY3 * wj - (2 + t) * D
    if y3 > D3:
        D3 = y3
    W_lo = 2 * tau - eta + D3
    res[0] = L; res[1] = W_up; res[2] = W_lo


@njit(cache=True, parallel=True)
def tq_lazy_run(indptr, indices, eu, ev, nchunks, out):
    n = indptr.shape[0] - 1
    m = eu.shape[0]
    deg = np.empty(n, np.int64)
    for z in range(n):
        deg[z] = indptr[z + 1] - indptr[z]
    dmax = 1
    S2 = 1
    for z in range(n):
        if deg[z] > dmax:
            dmax = deg[z]
        s2 = 0
        for f in range(indptr[z], indptr[z + 1]):
            s2 += deg[indices[f]]
        if s2 > S2:
            S2 = s2
    Dm = dmax + 3; Lm = S2 + 3
    for c in prange(nchunks):
        lo = c * m // nchunks; hi = (c + 1) * m // nchunks
        stamp = np.zeros(n, np.int64); tag = np.zeros(n, np.int8); lidx = np.zeros(n, np.int64)
        Ui = np.empty(Dm, np.int64); Uj = np.empty(Dm, np.int64); Cc = np.empty(Dm, np.int64)
        Hp = np.empty(Dm, np.int64); Hy = np.empty(Lm, np.int64); UUp = np.empty(Dm, np.int64)
        UUx = np.empty(Lm, np.int64); XCp = np.empty(Dm, np.int64); XCk = np.empty(Lm, np.int64)
        KYp = np.empty(Dm, np.int64); KYy = np.empty(Lm, np.int64); KXp = np.empty(Dm, np.int64)
        KXx = np.empty(Lm, np.int64); cur = np.empty(3 * Dm, np.int64)
        dHi = np.empty(Dm, np.int64); dHj = np.empty(Dm, np.int64); dCU = np.empty(Dm, np.int64)
        dCj = np.empty(Dm, np.int64); nout = np.empty(Dm, np.int64)
        resx = np.empty(Dm, np.int64); resy = np.empty(Dm, np.int64); resk = np.empty(Dm, np.int64)
        order = np.empty(Dm, np.int64); cnt = np.empty(Dm + 2, np.int64); badY = np.empty(Dm, np.int8)
        gptr = np.empty(3 * Dm + 1, np.int64); gadj = np.empty(2 * Lm + 2 * Dm, np.int64)
        gdeg = np.empty(3 * Dm + 1, np.int64); comp = np.empty(3 * Dm, np.int64)
        queue = np.empty(3 * Dm, np.int64); capv = np.empty(3 * Dm, np.int64)
        res = np.empty(NOUT, np.int64)
        for e in range(lo, hi):
            _lazy_edge(indptr, indices, deg, eu[e], ev[e], e + 1, stamp, tag, lidx, Ui, Uj, Cc, Hp, Hy,
                       UUp, UUx, XCp, XCk, KYp, KYy, KXp, KXx, cur, dHi, dHj, dCU, dCj, nout, resx, resy,
                       resk, order, cnt, badY, gptr, gadj, gdeg, comp, queue, capv, res)
            out[e, 0] = res[0]; out[e, 1] = res[1]; out[e, 2] = res[2]


@njit(cache=True, parallel=True)
def bf_run(indptr, indices, eu, ev, nchunks, out):
    """Balanced Forman curvature (Topping et al., Definition 1): exactly the work BF needs (the merge
    of the lists of i and j and one pass over the lists of U_i), for the runtime comparison."""
    n = indptr.shape[0] - 1
    m = eu.shape[0]
    for c in prange(nchunks):
        lo = c * m // nchunks; hi = (c + 1) * m // nchunks
        stamp = np.zeros(n, np.int64); tag = np.zeros(n, np.int8); hits = np.zeros(n, np.int64)
        for e in range(lo, hi):
            u = eu[e]; v = ev[e]
            if indptr[u + 1] - indptr[u] <= indptr[v + 1] - indptr[v]:
                i = u; j = v
            else:
                i = v; j = u
            a = indptr[i + 1] - indptr[i]; b = indptr[j + 1] - indptr[j]
            ep = e + 1
            p = indptr[i]; pe = indptr[i + 1]; q = indptr[j]; qe = indptr[j + 1]
            t = 0
            while p < pe or q < qe:
                if q >= qe or (p < pe and indices[p] < indices[q]):
                    w = indices[p]
                    if w != j:
                        stamp[w] = ep; tag[w] = 1; hits[w] = 0
                    p += 1
                elif p >= pe or indices[q] < indices[p]:
                    w = indices[q]
                    if w != i:
                        stamp[w] = ep; tag[w] = 2; hits[w] = 0
                    q += 1
                else:
                    t += 1; p += 1; q += 1
            nxi = 0; gmax = 0
            for f in range(indptr[i], indptr[i + 1]):
                x = indices[f]
                if stamp[x] != ep or tag[x] != 1:
                    continue
                h = 0
                for g in range(indptr[x], indptr[x + 1]):
                    w = indices[g]
                    if stamp[w] == ep and tag[w] == 2:
                        h += 1; hits[w] += 1
                if h > 0:
                    nxi += 1
                if h > gmax:
                    gmax = h
            nxj = 0
            for f in range(indptr[j], indptr[j + 1]):
                y = indices[f]
                if stamp[y] == ep and tag[y] == 2 and hits[y] > 0:
                    nxj += 1
                    if hits[y] > gmax:
                        gmax = hits[y]
            if a <= 1:
                out[e] = 0.0
            else:
                val = 2.0 / a + 2.0 / b - 2.0 + (2.0 / b + 1.0 / a) * t
                if gmax > 0:
                    val += (nxi + nxj) / (gmax * b)
                out[e] = val


def csr_sorted(n, E):
    """Symmetric CSR with sorted neighbour lists from an (m, 2) array of undirected edges."""
    E = np.asarray(E, dtype=np.int64)
    src = np.concatenate([E[:, 0], E[:, 1]]); dst = np.concatenate([E[:, 1], E[:, 0]])
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    return np.cumsum(indptr), np.ascontiguousarray(dst)


def default_chunks(m, threads=None):
    import numba
    th = threads or numba.get_num_threads()
    return max(1, min(m, 8 * th))


def tq_lazy(indptr, indices, E, nchunks=None):
    """Exact integer certificate for every edge of E: an (m, 3) int64 array (L, W_up, W_lo) with
    kappa >= 1 - W_up/L and kappa <= 1 - W_lo/L."""
    E = np.asarray(E, dtype=np.int64)
    out = np.empty((len(E), NOUT), np.int64)
    tq_lazy_run(indptr, indices, np.ascontiguousarray(E[:, 0]), np.ascontiguousarray(E[:, 1]),
                nchunks or default_chunks(len(E)), out)
    return out


def bounds_from(out):
    """Float view of the exact certificate (for widths and plots; decisions use the integers)."""
    L = out[:, 0].astype(np.float64)
    return 1.0 - out[:, 1] / L, 1.0 - out[:, 2] / L


def bf(indptr, indices, E, nchunks=None):
    E = np.asarray(E, dtype=np.int64)
    out = np.empty(len(E), np.float64)
    bf_run(indptr, indices, np.ascontiguousarray(E[:, 0]), np.ascontiguousarray(E[:, 1]),
           nchunks or default_chunks(len(E)), out)
    return out
