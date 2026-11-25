"""Triangle-quadrangle (TQ) certificate for the lazy Ollivier-Ricci curvature of every edge.

Information used per edge (i, j), a = deg i <= b = deg j: the neighbour lists of U_i (the smaller unique
side, exactly what computing BF requires) and of the common neighbours C. Everything it reveals is a
triangle or a quadrangle through i or j: the cross-edge graph H (BF's quadrangles), C-U_j edges
(quadrangles with a chord), C-U_i and U_i-U_i edges (triangles at i). The larger side U_j is never scanned,
no distance matrix, no flow solver.

Scaled integer masses (times L = (a+1)(b+1)): w_i -> b+1, w_j -> a+1, Delta -> b-a.
Residual transport (Lemma 4.6 of the paper): supply Delta at i, j and each k in C, w_i at each x in U_i;
demand w_j at each y in U_j. Baseline plan cost W0 = (3+2t) Delta + 3 |U_i| w_i (j->1, i->2, k->2, x->3).

Lower bound: W1 <= W0 - savings of a feasible greedy plan:
   2 per unit x->y along H, 1 per unit k->y along C-U_j edges,
   1 per unit x->y through a known common neighbour (a hub in C, or a hub x' in U_i with x ~ x').
Upper bound: in any completion, W1 >= 2 tau - eta1 + (mass certified at distance 3), where eta1 is the mass
   moved along edges: eta1 <= Delta (from j) + max flow of (U_i via H, C via C-U_j) into U_j; the max flow is
   bounded component by component (min of capped supply and capped demand). A vertex x in U_i is
   certified at distance 3 from U_j when all of its neighbours are i, common neighbours without U_j
   neighbours, or vertices of U_i without cross edges.
"""
import numpy as np
from numba import njit

NOUT = 10  # columns of the output


@njit(cache=True)
def _find(par, z):
    r = z
    while par[r] != r:
        r = par[r]
    while par[z] != r:
        nz = par[z]
        par[z] = r
        z = nz
    return r


@njit(cache=True)
def tq_kernel(indptr, indices, eu, ev, out, num, S2max, dmax):
    """out[e] = (kBF, L_comp, U_comp, L_tq, U_tq, a, b, t, Xi, gmax);
    num[e] = (L, W_upper_scaled, W_lower_scaled): kOR >= 1 - W_upper/L and kOR <= 1 - W_lower/L exactly."""
    n = indptr.shape[0] - 1
    m = eu.shape[0]
    stamp = np.zeros(n, np.int64)
    tag = np.zeros(n, np.int8)
    lidx = np.zeros(n, np.int64)
    Ui = np.empty(dmax + 1, np.int64); Uj = np.empty(dmax + 1, np.int64); Cc = np.empty(dmax + 1, np.int64)
    Hp = np.empty(dmax + 2, np.int64); Hy = np.empty(S2max + 1, np.int64)
    UUp = np.empty(dmax + 2, np.int64); UUx = np.empty(S2max + 1, np.int64)
    XCp = np.empty(dmax + 2, np.int64); XCk = np.empty(S2max + 1, np.int64)
    KYp = np.empty(dmax + 2, np.int64); KYy = np.empty(S2max + 1, np.int64)
    KXp = np.empty(dmax + 2, np.int64); KXx = np.empty(S2max + 1, np.int64); cur = np.empty(dmax + 2, np.int64)
    dHi = np.empty(dmax + 1, np.int64); dHj = np.empty(dmax + 1, np.int64)
    dCU = np.empty(dmax + 1, np.int64); dCj = np.empty(dmax + 1, np.int64); nout = np.empty(dmax + 1, np.int64)
    resx = np.empty(dmax + 1, np.int64); resy = np.empty(dmax + 1, np.int64); resk = np.empty(dmax + 1, np.int64)
    order = np.empty(dmax + 1, np.int64); cnt = np.empty(dmax + 2, np.int64)
    par = np.empty(3 * dmax + 3, np.int64); ssum = np.empty(3 * dmax + 3, np.int64); dsum = np.empty(3 * dmax + 3, np.int64)
    for e in range(m):
        u = eu[e]
        v = ev[e]
        if indptr[u + 1] - indptr[u] <= indptr[v + 1] - indptr[v]:
            i = u; j = v
        else:
            i = v; j = u
        a = indptr[i + 1] - indptr[i]
        b = indptr[j + 1] - indptr[j]
        ep = e + 1
        # ---- merge the two sorted neighbour lists: U_i, U_j, C
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
        # ---- scan A: neighbour lists of U_i (what BF needs)
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
                    else:
                        XCk[cp] = lidx[w]; cp += 1
                elif w != i:
                    nout[xx] += 1
        Hp[nx] = hp; UUp[nx] = up; XCp[nx] = cp
        for yy in range(ny):
            dHj[yy] = 0; dCj[yy] = 0
        for f in range(hp):
            dHj[Hy[f]] += 1
        # ---- scan B: neighbour lists of C (C-U_j edges)
        kp = 0
        for kk in range(t):
            k = Cc[kk]
            KYp[kk] = kp; dCU[kk] = 0
            for f in range(indptr[k], indptr[k + 1]):
                w = indices[f]
                if stamp[w] == ep and tag[w] == 2:
                    KYy[kp] = lidx[w]; kp += 1; dCU[kk] += 1; dCj[lidx[w]] += 1
        KYp[t] = kp
        # invert x -> C lists into k -> U_i lists
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
        # ---- BF and its statistics
        xi = 0; gmax = 0
        for xx in range(nx):
            if dHi[xx] > 0:
                xi += 1
                if dHi[xx] > gmax: gmax = dHi[xx]
        for yy in range(ny):
            if dHj[yy] > 0:
                xi += 1
                if dHj[yy] > gmax: gmax = dHj[yy]
        if a <= 1:
            bf = 0.0
        else:
            bf = 2.0 / a + 2.0 / b - 2.0 + (2.0 / b + 1.0 / a) * t + (xi / (gmax * b) if gmax > 0 else 0.0)
        L = (a + 1) * (b + 1); wis = b + 1; wjs = a + 1; Ds = b - a
        # components interval (BF ingredients only)
        Lc = (3.0 + t) / (a + 1) + (3.0 + 2 * t + (xi / gmax if gmax > 0 else 0.0)) / (b + 1) - 2.0
        th = (3.0 + t) / (b + 1) - 1.0 / (a + 1)
        Uc = th - max(0.0, (a - 1.0 - t) / (a + 1) - xi / (a + b + 2.0))
        # ---- lower bound: greedy feasible plan
        for xx in range(nx): resx[xx] = wis
        for yy in range(ny): resy[yy] = wjs
        for kk in range(t): resk[kk] = Ds
        for z in range(ny + 2): cnt[z] = 0
        for xx in range(nx): cnt[dHi[xx]] += 1
        acc = 0
        for z in range(ny + 1):
            c0 = cnt[z]; cnt[z] = acc; acc += c0
        for xx in range(nx):
            order[cnt[dHi[xx]]] = xx; cnt[dHi[xx]] += 1
        F1 = 0
        for idx in range(nx):
            xx = order[idx]
            if dHi[xx] == 0:
                continue
            r = resx[xx]
            for f in range(Hp[xx], Hp[xx + 1]):
                yy = Hy[f]
                s = r if r < resy[yy] else resy[yy]
                if s > 0:
                    r -= s; resy[yy] -= s; F1 += s
                    if r == 0: break
            resx[xx] = r
        G = 0
        if Ds > 0:
            for kk in range(t):
                r = resk[kk]
                for f in range(KYp[kk], KYp[kk + 1]):
                    yy = KYy[f]
                    s = r if r < resy[yy] else resy[yy]
                    if s > 0:
                        r -= s; resy[yy] -= s; G += s
                        if r == 0: break
                resk[kk] = r
        F2 = 0
        for kk in range(t):                      # hubs in C
            pa = KXp[kk]; pb = KXp[kk + 1]; qa = KYp[kk]; qb = KYp[kk + 1]
            while pa < pb and qa < qb:
                xx = KXx[pa]; yy = KYy[qa]
                s = resx[xx] if resx[xx] < resy[yy] else resy[yy]
                if s > 0:
                    resx[xx] -= s; resy[yy] -= s; F2 += s
                if resx[xx] == 0: pa += 1
                if resy[yy] == 0: qa += 1
        for xp in range(nx):                     # hubs in U_i (x ~ x', x' ~ y)
            if dHi[xp] == 0:
                continue
            pa = UUp[xp]; pb = UUp[xp + 1]; qa = Hp[xp]; qb = Hp[xp + 1]
            while pa < pb and qa < qb:
                xx = UUx[pa]; yy = Hy[qa]
                s = resx[xx] if resx[xx] < resy[yy] else resy[yy]
                if s > 0:
                    resx[xx] -= s; resy[yy] -= s; F2 += s
                if resx[xx] == 0: pa += 1
                if resy[yy] == 0: qa += 1
        Wub = (3 + 2 * t) * Ds + 3 * nx * wis - (2 * F1 + G + F2)
        # ---- upper bound: component-wise cut of the distance-1 flow, certified distance-3 mass
        NN = nx + t + ny
        for z in range(NN):
            par[z] = z; ssum[z] = 0; dsum[z] = 0
        for xx in range(nx):
            for f in range(Hp[xx], Hp[xx + 1]):
                ra = _find(par, xx); rb = _find(par, nx + t + Hy[f])
                if ra != rb: par[ra] = rb
        if Ds > 0:
            for kk in range(t):
                for f in range(KYp[kk], KYp[kk + 1]):
                    ra = _find(par, nx + kk); rb = _find(par, nx + t + KYy[f])
                    if ra != rb: par[ra] = rb
        for xx in range(nx):
            if dHi[xx] > 0:
                cap = dHi[xx] * wjs
                ssum[_find(par, xx)] += cap if cap < wis else wis
        if Ds > 0:
            for kk in range(t):
                if dCU[kk] > 0:
                    cap = dCU[kk] * wjs
                    ssum[_find(par, nx + kk)] += cap if cap < Ds else Ds
        for yy in range(ny):
            cap = dHj[yy] * wis + (dCj[yy] * Ds if Ds > 0 else 0)
            if cap > 0:
                dsum[_find(par, nx + t + yy)] += cap if cap < wjs else wjs
        eta = 0
        for z in range(NN):
            if par[z] == z:
                eta += ssum[z] if ssum[z] < dsum[z] else dsum[z]
        far = 0
        for xx in range(nx):
            if dHi[xx] == 0 and nout[xx] == 0:
                ok = True
                for f in range(XCp[xx], XCp[xx + 1]):
                    if dCU[XCk[f]] > 0:
                        ok = False; break
                if ok:
                    for f in range(UUp[xx], UUp[xx + 1]):
                        if dHi[UUx[f]] > 0:
                            ok = False; break
                if ok: far += 1
        # closed vertices of U_j: deg(y) = 1 + (known edges to C); if no edge to U_i and no C-neighbour of y
        # touches U_i, y is at distance 3 from all of U_i; their demand beyond the near supply (i, j, C) travels 3
        for kk in range(t):
            cur[kk] = 1 if KXp[kk + 1] > KXp[kk] else 0
        for yy in range(ny):
            resy[yy] = 1                       # reuse as a flag: 1 = far candidate
        for kk in range(t):
            if cur[kk] == 1:
                for f in range(KYp[kk], KYp[kk + 1]):
                    resy[KYy[f]] = 0
        nyfar = 0
        for yy in range(ny):
            if resy[yy] == 1 and dHj[yy] == 0 and indptr[Uj[yy] + 1] - indptr[Uj[yy]] == 1 + dCj[yy]:
                nyfar += 1
        yexc = nyfar * wjs - (2 + t) * Ds
        if yexc < 0: yexc = 0
        d3 = far * wis
        if yexc > d3: d3 = yexc
        Wlb = 2 * ny * wjs - Ds - eta + d3
        if a <= 1:            # leaf endpoint: both bounds are exact (no cycle through the edge)
            Wub = 3 * Ds
            Wlb = 3 * Ds
        out[e, 0] = bf; out[e, 1] = Lc; out[e, 2] = Uc
        out[e, 3] = 1.0 - Wub / L; out[e, 4] = 1.0 - Wlb / L
        out[e, 5] = a; out[e, 6] = b; out[e, 7] = t; out[e, 8] = xi; out[e, 9] = gmax
        num[e, 0] = L; num[e, 1] = Wub; num[e, 2] = Wlb


def csr_from_edges(n, E):
    E = np.asarray(E, dtype=np.int64)
    src = np.concatenate([E[:, 0], E[:, 1]]); dst = np.concatenate([E[:, 1], E[:, 0]])
    o = np.lexsort((dst, src)); src = src[o]; dst = dst[o]
    indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, src + 1, 1); indptr = np.cumsum(indptr)
    return indptr, dst.astype(np.int64)


def prep(n, E):
    indptr, indices = csr_from_edges(n, E)
    deg = np.diff(indptr)
    S2 = np.zeros(n, np.int64)
    np.add.at(S2, np.repeat(np.arange(n), deg), deg[indices])
    return indptr, indices, int(S2.max()) + 1, int(deg.max()) + 1


def tq(n, E, pre=None):
    indptr, indices, S2max, dmax = prep(n, E) if pre is None else pre
    E = np.asarray(E, dtype=np.int64)
    out = np.empty((len(E), NOUT)); num = np.empty((len(E), 3), np.int64)
    tq_kernel(indptr, indices, E[:, 0].copy(), E[:, 1].copy(), out, num, S2max, dmax)
    return out, num
