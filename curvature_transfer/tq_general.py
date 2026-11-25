"""Triangle-quadrangle certificate (Theorem 5.3) for one-step measures, in float64.

Measures: mu_u(u) = alpha_u and mu_u(v) = (1 - alpha_u) * omega_v / sum_{w in N(u)} omega_w on N(u), with
  idleness mode   0: lazy (alpha_u = 1/(d_u+1)),  1: uniform p,  2: uniform p_e = 1/(max(d_i,d_j)+1) per edge
                  (the idleness w_j at which Corollary 5.4 certifies every p in [w_j, 1]);
  weight mode     0: uniform (omega_v = 1),  1: degree-biased (omega_v = 1/d_v).
Per edge: lower end 1 - cost(greedy plan), upper end 1 - 2 tau + eta' - D3 (Theorem 5.3 (i), (ii), (iv)).
The building blocks are read exactly as in the paper: the lists of i, j, U_i, C, and the degrees of U_j.
"""
import numpy as np
from numba import njit, prange

EPS = 1e-12
NOUT = 4   # lower, upper, tau, cost


@njit(cache=True)
def _route(a, b):
    return a if a < b else b


@njit(cache=True)
def _edge(indptr, indices, deg, u, v, ep, idle_mode, p, wmode,
          stamp, tag, lidx, Ui, Uj, Cc, Hp, Hy, UUp, UUx, XCp, XCk, KYp, KYy, CCp, CCk, KXp, KXx, cur,
          dHi, dHj, dCU, dCj, nout, rS, rx, ry, resS, resx, resy, order, cnt, supNbY, badY,
          gptr, gadj, gdeg, comp, queue, capv, out):
    if deg[u] <= deg[v]:
        i = u; j = v
    else:
        i = v; j = u
    a = deg[i]; b = deg[j]
    # ---------- idleness and weights
    if idle_mode == 0:
        al_i = 1.0 / (a + 1.0); al_j = 1.0 / (b + 1.0)
    elif idle_mode == 1:
        al_i = p; al_j = p
    else:
        al_i = 1.0 / (b + 1.0); al_j = al_i
    # ---------- merge: S = {i, j} u C (local 0 = i, 1 = j, 2.. = C), U_i, U_j
    stamp[i] = ep; tag[i] = 4; stamp[j] = ep; tag[j] = 5
    Om_i = 0.0; Om_j = 0.0
    pp = indptr[i]; pe = indptr[i + 1]; q = indptr[j]; qe = indptr[j + 1]
    nx = 0; ny = 0; t = 0
    while pp < pe or q < qe:
        if q >= qe or (pp < pe and indices[pp] < indices[q]):
            w = indices[pp]
            om = 1.0 if wmode == 0 else 1.0 / deg[w]
            Om_i += om
            if w != j:
                Ui[nx] = w; stamp[w] = ep; tag[w] = 1; lidx[w] = nx; nx += 1
            pp += 1
        elif pp >= pe or indices[q] < indices[pp]:
            w = indices[q]
            om = 1.0 if wmode == 0 else 1.0 / deg[w]
            Om_j += om
            if w != i:
                Uj[ny] = w; stamp[w] = ep; tag[w] = 2; lidx[w] = ny; ny += 1
            q += 1
        else:
            w = indices[pp]
            om = 1.0 if wmode == 0 else 1.0 / deg[w]
            Om_i += om; Om_j += om
            Cc[t] = w; stamp[w] = ep; tag[w] = 3; lidx[w] = t; t += 1
            pp += 1; q += 1
    ns = t + 2
    # ---------- residuals
    oi = 1.0 if wmode == 0 else 1.0 / deg[i]
    oj = 1.0 if wmode == 0 else 1.0 / deg[j]
    rS[0] = al_i - (1.0 - al_j) * oi / Om_j
    rS[1] = (1.0 - al_i) * oj / Om_i - al_j
    for kk in range(t):
        k = Cc[kk]
        om = 1.0 if wmode == 0 else 1.0 / deg[k]
        rS[2 + kk] = (1.0 - al_i) * om / Om_i - (1.0 - al_j) * om / Om_j
    for xx in range(nx):
        x = Ui[xx]
        rx[xx] = (1.0 - al_i) * ((1.0 if wmode == 0 else 1.0 / deg[x]) / Om_i)
    for yy in range(ny):
        y = Uj[yy]
        ry[yy] = (1.0 - al_j) * ((1.0 if wmode == 0 else 1.0 / deg[y]) / Om_j)
    for s in range(ns):
        if -EPS < rS[s] < EPS:
            rS[s] = 0.0
    tau = 0.0; rSp = 0.0; rSm = 0.0
    for s in range(ns):
        if rS[s] > 0.0:
            rSp += rS[s]
        else:
            rSm -= rS[s]
    tau = rSp
    for xx in range(nx):
        tau += rx[xx]
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
        dHj[yy] = 0; dCj[yy] = 0; supNbY[yy] = 0.0; badY[yy] = 0
    for f in range(hp):
        dHj[Hy[f]] += 1
    # ---------- scan B: lists of C
    kp = 0; ccp = 0
    for kk in range(t):
        k = Cc[kk]
        KYp[kk] = kp; CCp[kk] = ccp; dCU[kk] = 0
        for f in range(indptr[k], indptr[k + 1]):
            w = indices[f]
            if stamp[w] == ep:
                tg = tag[w]
                if tg == 2:
                    KYy[kp] = lidx[w]; kp += 1; dCU[kk] += 1; dCj[lidx[w]] += 1
                elif tg == 3:
                    CCk[ccp] = lidx[w]; ccp += 1
    KYp[t] = kp; CCp[t] = ccp
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
    # ---------- lower end: greedy plan
    for s in range(ns):
        resS[s] = rS[s] if rS[s] > 0.0 else -rS[s]
    for xx in range(nx):
        resx[xx] = rx[xx]
    for yy in range(ny):
        resy[yy] = ry[yy]
    cost = 0.0
    # (1) U_i -> U_j along cross edges, vertices of low H-degree first (cost 1)
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
            s_ = _route(resx[xx], resy[yy])
            if s_ > 0.0:
                resx[xx] -= s_; resy[yy] -= s_; cost += s_
                if resx[xx] <= 0.0:
                    break
    # (1b) U_i -> adjacent demand on S (i, and common neighbours), cost 1
    if rS[0] < 0.0:
        for xx in range(nx):
            s_ = _route(resx[xx], resS[0])
            if s_ > 0.0:
                resx[xx] -= s_; resS[0] -= s_; cost += s_
    for xx in range(nx):
        for f in range(XCp[xx], XCp[xx + 1]):
            s = 2 + XCk[f]
            if rS[s] < 0.0:
                s_ = _route(resx[xx], resS[s])
                if s_ > 0.0:
                    resx[xx] -= s_; resS[s] -= s_; cost += s_
    # fallback 1 for a supplier of S adjacent to every demand vertex
    nQ = 0
    for s in range(ns):
        if rS[s] < 0.0:
            nQ += 1
    for yy in range(ny):
        if ry[yy] > 0.0:
            nQ += 1
    nCdem = 0
    for kk in range(t):
        if rS[2 + kk] < 0.0:
            nCdem += 1
    for phase in range(2):    # phase 0: suppliers with fallback 2 (step 2); phase 1: all (step 4a)
        for s in range(ns):
            if rS[s] <= 0.0 or resS[s] <= 0.0:
                continue
            if s == 0:
                adj = (1 if rS[1] < 0.0 else 0) + nCdem
            elif s == 1:
                adj = (1 if rS[0] < 0.0 else 0) + nCdem + ny
            else:
                kk = s - 2
                adj = (1 if rS[0] < 0.0 else 0) + (1 if rS[1] < 0.0 else 0) + dCU[kk]
                for f in range(CCp[kk], CCp[kk + 1]):
                    if rS[2 + CCk[f]] < 0.0:
                        adj += 1
            fb1 = adj == nQ
            if phase == 0 and fb1:
                continue
            # route to adjacent demand at cost 1
            for s2 in range(2):
                if s2 != s and rS[s2] < 0.0 and (s >= 2 or s2 != s):
                    s_ = _route(resS[s], resS[s2])
                    if s_ > 0.0:
                        resS[s] -= s_; resS[s2] -= s_; cost += s_
            if s <= 1:
                for kk in range(t):
                    if rS[2 + kk] < 0.0:
                        s_ = _route(resS[s], resS[2 + kk])
                        if s_ > 0.0:
                            resS[s] -= s_; resS[2 + kk] -= s_; cost += s_
                if s == 1:
                    for yy in range(ny):
                        s_ = _route(resS[s], resy[yy])
                        if s_ > 0.0:
                            resS[s] -= s_; resy[yy] -= s_; cost += s_
            else:
                kk = s - 2
                for f in range(CCp[kk], CCp[kk + 1]):
                    s2 = 2 + CCk[f]
                    if rS[s2] < 0.0:
                        s_ = _route(resS[s], resS[s2])
                        if s_ > 0.0:
                            resS[s] -= s_; resS[s2] -= s_; cost += s_
                for f in range(KYp[kk], KYp[kk + 1]):
                    yy = KYy[f]
                    s_ = _route(resS[s], resy[yy])
                    if s_ > 0.0:
                        resS[s] -= s_; resy[yy] -= s_; cost += s_
        if phase == 0:
            # (3) detours of length two at cost 2: hubs in C and in U_i, then U_i -> demand on S
            for kk in range(t):
                pa = KXp[kk]; pb = KXp[kk + 1]; qa = KYp[kk]; qb = KYp[kk + 1]
                while pa < pb and qa < qb:
                    xx = KXx[pa]; yy = KYy[qa]
                    s_ = _route(resx[xx], resy[yy])
                    if s_ > 0.0:
                        resx[xx] -= s_; resy[yy] -= s_; cost += 2.0 * s_
                    if resx[xx] <= 0.0:
                        pa += 1
                    if resy[yy] <= 0.0:
                        qa += 1
            for xp in range(nx):
                if dHi[xp] == 0:
                    continue
                pa = UUp[xp]; pb = UUp[xp + 1]; qa = Hp[xp]; qb = Hp[xp + 1]
                while pa < pb and qa < qb:
                    xx = UUx[pa]; yy = Hy[qa]
                    s_ = _route(resx[xx], resy[yy])
                    if s_ > 0.0:
                        resx[xx] -= s_; resy[yy] -= s_; cost += 2.0 * s_
                    if resx[xx] <= 0.0:
                        pa += 1
                    if resy[yy] <= 0.0:
                        qa += 1
            xx = 0
            for s in range(ns):
                if rS[s] < 0.0:
                    while resS[s] > 0.0 and xx < nx:
                        s_ = _route(resx[xx], resS[s])
                        if s_ > 0.0:
                            resx[xx] -= s_; resS[s] -= s_; cost += 2.0 * s_
                        if resx[xx] <= 0.0:
                            xx += 1
    # (4b) remaining supply on S -> any remaining demand, cost at most 2
    yy = 0; s2 = 0
    for s in range(ns):
        if rS[s] <= 0.0:
            continue
        while resS[s] > 0.0:
            while s2 < ns and (rS[s2] >= 0.0 or resS[s2] <= 0.0):
                s2 += 1
            if s2 < ns:
                s_ = _route(resS[s], resS[s2])
                resS[s] -= s_; resS[s2] -= s_; cost += 2.0 * s_
                continue
            while yy < ny and resy[yy] <= 0.0:
                yy += 1
            if yy >= ny:
                break
            s_ = _route(resS[s], resy[yy])
            resS[s] -= s_; resy[yy] -= s_; cost += 2.0 * s_
    # (4c) remaining supply on U_i -> demand on S (cost 2), then -> U_j (cost 3)
    yy = 0; s2 = 0
    for xx in range(nx):
        while resx[xx] > 0.0:
            while s2 < ns and (rS[s2] >= 0.0 or resS[s2] <= 0.0):
                s2 += 1
            if s2 < ns:
                s_ = _route(resx[xx], resS[s2])
                resx[xx] -= s_; resS[s2] -= s_; cost += 2.0 * s_
                continue
            while yy < ny and resy[yy] <= 0.0:
                yy += 1
            if yy >= ny:
                break
            s_ = _route(resx[xx], resy[yy])
            resx[xx] -= s_; resy[yy] -= s_; cost += 3.0 * s_
    lower = 1.0 - cost
    # ---------- upper end: eta' = capped flows of i, j + componentwise caps
    # capped supply of x in U_i: neighbours with demand are i, common neighbours, U_j (cross edges)
    rmi = -rS[0] if rS[0] < 0.0 else 0.0
    rmj = -rS[1] if rS[1] < 0.0 else 0.0
    rpi = rS[0] if rS[0] > 0.0 else 0.0
    rpj = rS[1] if rS[1] > 0.0 else 0.0
    sumCm = 0.0; sumCp = 0.0
    for kk in range(t):
        if rS[2 + kk] < 0.0:
            sumCm -= rS[2 + kk]
        else:
            sumCp += rS[2 + kk]
    sumUj = 0.0
    for yy in range(ny):
        sumUj += ry[yy]
    sumUi = 0.0
    for xx in range(nx):
        sumUi += rx[xx]
    # mass on edges at i or j: flow F on (i, j) plus the other edges at i and at j, each capped by the
    # residual of its endpoint and by the opposite residual of the other neighbours (a concave function of F)
    ari = rpi + rmi
    arj = rpj + rmj
    if rS[0] > 0.0:
        Oi = sumCm
    else:
        Oi = sumCp + sumUi
    if rS[1] > 0.0:
        Oj = sumCm + sumUj
    else:
        Oj = sumCp
    if (rS[0] > 0.0 and rS[1] < 0.0) or (rS[0] < 0.0 and rS[1] > 0.0):
        Fm = ari if ari < arj else arj
        capij = 0.0
        for kb in range(4):
            if kb == 0:
                F = 0.0
            elif kb == 1:
                F = ari - Oi
            elif kb == 2:
                F = arj - Oj
            else:
                F = Fm
            if F < 0.0:
                F = 0.0
            if F > Fm:
                F = Fm
            tot = F + _route(Oi, ari - F) + _route(Oj, arj - F)
            if tot > capij:
                capij = tot
    else:
        capij = _route(ari, Oi) + _route(arj, Oj)
    # local graph of supply-demand edges outside {i, j}: nodes U_i (0..nx-1), C (nx..), U_j (nx+t..)
    NN = nx + t + ny
    for z in range(NN + 1):
        gdeg[z] = 0
    for xx in range(nx):
        for f in range(Hp[xx], Hp[xx + 1]):
            gdeg[xx] += 1; gdeg[nx + t + Hy[f]] += 1
        for f in range(XCp[xx], XCp[xx + 1]):
            if rS[2 + XCk[f]] < 0.0:
                gdeg[xx] += 1; gdeg[nx + XCk[f]] += 1
    for kk in range(t):
        if rS[2 + kk] > 0.0:
            for f in range(KYp[kk], KYp[kk + 1]):
                gdeg[nx + kk] += 1; gdeg[nx + t + KYy[f]] += 1
            for f in range(CCp[kk], CCp[kk + 1]):
                if rS[2 + CCk[f]] < 0.0:
                    gdeg[nx + kk] += 1; gdeg[nx + CCk[f]] += 1
    gptr[0] = 0
    for z in range(NN):
        gptr[z + 1] = gptr[z] + gdeg[z]
    for z in range(NN):
        cur[z] = gptr[z]
    for xx in range(nx):
        for f in range(Hp[xx], Hp[xx + 1]):
            o = nx + t + Hy[f]
            gadj[cur[xx]] = o; cur[xx] += 1; gadj[cur[o]] = xx; cur[o] += 1
        for f in range(XCp[xx], XCp[xx + 1]):
            if rS[2 + XCk[f]] < 0.0:
                o = nx + XCk[f]
                gadj[cur[xx]] = o; cur[xx] += 1; gadj[cur[o]] = xx; cur[o] += 1
    for kk in range(t):
        if rS[2 + kk] > 0.0:
            me = nx + kk
            for f in range(KYp[kk], KYp[kk + 1]):
                o = nx + t + KYy[f]
                gadj[cur[me]] = o; cur[me] += 1; gadj[cur[o]] = me; cur[o] += 1
            for f in range(CCp[kk], CCp[kk + 1]):
                if rS[2 + CCk[f]] < 0.0:
                    o = nx + CCk[f]
                    gadj[cur[me]] = o; cur[me] += 1; gadj[cur[o]] = me; cur[o] += 1
    # capped supply / demand of every node (all neighbours, as in the formal statement)
    for xx in range(nx):
        nb = rmi
        for f in range(Hp[xx], Hp[xx + 1]):
            nb += ry[Hy[f]]
        for f in range(XCp[xx], XCp[xx + 1]):
            if rS[2 + XCk[f]] < 0.0:
                nb -= rS[2 + XCk[f]]
        capv[xx] = _route(rx[xx], nb)
    for xx in range(nx):
        for f in range(Hp[xx], Hp[xx + 1]):
            supNbY[Hy[f]] += rx[xx]
    for kk in range(t):
        if rS[2 + kk] > 0.0:
            for f in range(KYp[kk], KYp[kk + 1]):
                supNbY[KYy[f]] += rS[2 + kk]
    for yy in range(ny):
        capv[nx + t + yy] = _route(ry[yy], supNbY[yy] + rpj)
    for kk in range(t):
        r = rS[2 + kk]
        if r > 0.0:
            nb = rmi + rmj
            for f in range(KYp[kk], KYp[kk + 1]):
                nb += ry[KYy[f]]
            for f in range(CCp[kk], CCp[kk + 1]):
                if rS[2 + CCk[f]] < 0.0:
                    nb -= rS[2 + CCk[f]]
            capv[nx + kk] = _route(r, nb)
        elif r < 0.0:
            nb = rpi + rpj
            for f in range(KXp[kk], KXp[kk + 1]):
                nb += rx[KXx[f]]
            for f in range(CCp[kk], CCp[kk + 1]):
                if rS[2 + CCk[f]] > 0.0:
                    nb += rS[2 + CCk[f]]
            capv[nx + kk] = _route(-r, nb)
        else:
            capv[nx + kk] = 0.0
    # components by breadth-first search
    for z in range(NN):
        comp[z] = -1
    eta = capij
    for z0 in range(NN):
        if comp[z0] >= 0 or gptr[z0 + 1] == gptr[z0]:
            continue
        head = 0; tail = 0; queue[tail] = z0; tail += 1; comp[z0] = z0
        sS = 0.0; sD = 0.0
        while head < tail:
            z = queue[head]; head += 1
            if z < nx:
                sS += capv[z]
            elif z < nx + t:
                if rS[2 + z - nx] > 0.0:
                    sS += capv[z]
                else:
                    sD += capv[z]
            else:
                sD += capv[z]
            for f in range(gptr[z], gptr[z + 1]):
                o = gadj[f]
                if comp[o] < 0:
                    comp[o] = z0; queue[tail] = o; tail += 1
        eta += sS if sS < sD else sD
    # certified distance-3 mass
    rX = 0.0
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
                rX += rx[xx]
    for kk in range(t):
        if KXp[kk + 1] > KXp[kk]:
            for f in range(KYp[kk], KYp[kk + 1]):
                badY[KYy[f]] = 1
    rY = 0.0
    for yy in range(ny):
        if dHj[yy] == 0 and badY[yy] == 0 and deg[Uj[yy]] == 1 + dCj[yy]:
            rY += ry[yy]
    D3 = 0.0
    if rX - rSm > D3:
        D3 = rX - rSm
    if rY - rSp > D3:
        D3 = rY - rSp
    upper = 1.0 - 2.0 * tau + eta - D3
    out[0] = lower; out[1] = upper; out[2] = tau; out[3] = cost


@njit(cache=True, parallel=True)
def tq_general(indptr, indices, eu, ev, idle_mode, p, wmode, out, nchunks):
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
    D = dmax + 3; L = S2 + 3
    for c in prange(nchunks):
        lo = c * m // nchunks; hi = (c + 1) * m // nchunks
        stamp = np.zeros(n, np.int64); tag = np.zeros(n, np.int8); lidx = np.zeros(n, np.int64)
        Ui = np.empty(D, np.int64); Uj = np.empty(D, np.int64); Cc = np.empty(D, np.int64)
        Hp = np.empty(D, np.int64); Hy = np.empty(L, np.int64); UUp = np.empty(D, np.int64); UUx = np.empty(L, np.int64)
        XCp = np.empty(D, np.int64); XCk = np.empty(L, np.int64); KYp = np.empty(D, np.int64); KYy = np.empty(L, np.int64)
        CCp = np.empty(D, np.int64); CCk = np.empty(L, np.int64); KXp = np.empty(D, np.int64); KXx = np.empty(L, np.int64)
        cur = np.empty(3 * D, np.int64)
        dHi = np.empty(D, np.int64); dHj = np.empty(D, np.int64); dCU = np.empty(D, np.int64); dCj = np.empty(D, np.int64)
        nout = np.empty(D, np.int64)
        rS = np.empty(D, np.float64); rx = np.empty(D, np.float64); ry = np.empty(D, np.float64)
        resS = np.empty(D, np.float64); resx = np.empty(D, np.float64); resy = np.empty(D, np.float64)
        order = np.empty(D, np.int64); cnt = np.empty(D + 2, np.int64)
        supNbY = np.empty(D, np.float64); badY = np.empty(D, np.int8)
        gptr = np.empty(3 * D + 1, np.int64); gadj = np.empty(4 * L, np.int64); gdeg = np.empty(3 * D + 1, np.int64)
        comp = np.empty(3 * D, np.int64); queue = np.empty(3 * D, np.int64); capv = np.empty(3 * D, np.float64)
        o = np.empty(NOUT, np.float64)
        for e in range(lo, hi):
            _edge(indptr, indices, deg, eu[e], ev[e], e + 1, idle_mode, p, wmode,
                  stamp, tag, lidx, Ui, Uj, Cc, Hp, Hy, UUp, UUx, XCp, XCk, KYp, KYy, CCp, CCk, KXp, KXx, cur,
                  dHi, dHj, dCU, dCj, nout, rS, rx, ry, resS, resx, resy, order, cnt, supNbY, badY,
                  gptr, gadj, gdeg, comp, queue, capv, o)
            for z in range(NOUT):
                out[e, z] = o[z]
