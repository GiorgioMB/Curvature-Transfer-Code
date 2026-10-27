"""Numba kernels.

Two families of per-edge kernels, each with a parallel driver over edges:

* ``edge_stats``: the 2-hop statistics that parameterize BF curvature and all
  combinatorial bounds (triangles, unique neighborhoods, 4-cycle counts, and
  the maximum matching of the cross-edge graph, via Hopcroft--Karp).
* ``exact``: exact Wasserstein-1 distances between one-step measures, for the
  lazy measures of the paper (OR curvature), the non-lazy neighbor laws, and
  the uniform idleness p = 1/(deg_max + 1), which yields the Lin--Lu--Yau
  curvature exactly by the idleness linearity of Bourne et al. (2018); plus
  the TLW dual certificate, which shares all of its work with the exact solve.

Exact transport
---------------
For an edge (i, j) split S = N(i) u N(j) into U_i = N(i) \\ N[j],
U_j = N(j) \\ N[i], and C = {i, j} u (N(i) & N(j)). For all three measures the
net mass mu_i - mu_j is positive on U_i, negative on U_j, and constant on each
of {i}, {j}, and the triangle vertices; P and N denote the vertices of positive
and negative net mass. Every pair of S is within distance 2 (through i or j)
except possibly a cross pair (x, y) in U_i x U_j, which is at distance 1 if
adjacent, 2 if it has a common neighbour anywhere in the graph, and 3 otherwise.

W1 only depends on mu - nu, and the transport problem from P to N with costs
c in {1, 2, 3} has an integral optimal dual with u in {1, 2, 3} on P and
v in {0, 1, 2} on N (shift the Kantorovich potential, whose range is at most 3).
With p_x = [u_x >= 2], q_x = [u_x >= 3], r_y = [v_y >= 1], t_y = [v_y >= 2],
the constraints u_x - v_y <= c are the implications q => p, t => r, and
p_x => r_y, q_x => t_y if c = 1; q_x => r_y if c = 2 (t => r is implied by
the others for the cut value and is not materialized). So the dual is a
maximum-weight closure (Picard 1976) on 2|P| + 2|N| nodes, solved by one s-t
min cut, and W1 = 3 A - maxflow with A the total positive mass. Implications
q_x => r_y for pairs within distance 2 go through hubs where whole blocks are
at distance <= 2 (pairs touching C, fully covered rows and columns), and are
listed explicitly only for the remaining covered cross pairs.

Fast path: if every cross pair is within distance 2 (typical in dense
neighbourhoods), every transport plan pays 1 on adjacent pairs and 2 elsewhere,
so W1 = 2 A - maxflow(G1), a single bipartite flow on the adjacency G1 between
P and N, with no distance information at all. Otherwise some optimal plan still
contains a maximum G1 flow (moving a unit onto G1 saves 1 and forces at most 1
unit onto a distance-3 pair), so on larger token networks that flow, loaded on
both levels, warm-starts the min cut.

Distances are never materialized. Adjacency to the columns U_j u C and the
common-neighbour coverage of the cross pairs are computed with bitsets over
the columns: one bitset per middle vertex (the columns it is adjacent to), and
each row of U_i ORs the bitsets of its neighbours (with early exit once the row
is fully covered). Rows with identical patterns are merged (hashing), which
collapses, for example, the leaves of a hub. Endpoints are oriented so that the
larger neighbourhood is on the row side. Max flow is Dinic on a CSR network
with an iterative DFS. All arithmetic is exact (integer masses); all
temporaries come from per-thread arenas, so steady state allocates nothing.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

INF = np.int64(1) << np.int64(60)

# ----------------------------------------------------------------- utilities


@njit(cache=True, inline="always")
def _contains(arr, lo, hi, x):
    """Binary search for x in sorted arr[lo:hi]."""
    while lo < hi:
        mid = (lo + hi) >> 1
        v = arr[mid]
        if v < x:
            lo = mid + 1
        elif v > x:
            hi = mid
        else:
            return True
    return False


@njit(cache=True, inline="always")
def _find(arr, x):
    """Index of x in sorted arr, or -1."""
    lo = 0
    hi = arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        v = arr[mid]
        if v < x:
            lo = mid + 1
        elif v > x:
            hi = mid
        else:
            return mid
    return -1


@njit(cache=True)
def _count_intersect(a, alo, ahi, b, blo, bhi):
    """|a[alo:ahi] & b[blo:bhi]| for sorted arrays (merge or galloping)."""
    na = ahi - alo
    nb = bhi - blo
    c = 0
    if na * 8 < nb:
        for p in range(alo, ahi):
            if _contains(b, blo, bhi, a[p]):
                c += 1
        return c
    if nb * 8 < na:
        for p in range(blo, bhi):
            if _contains(a, alo, ahi, b[p]):
                c += 1
        return c
    p = alo
    q = blo
    while p < ahi and q < bhi:
        if a[p] < b[q]:
            p += 1
        elif a[p] > b[q]:
            q += 1
        else:
            c += 1
            p += 1
            q += 1
    return c


# --------------------------------------------------------------- matching


@njit(cache=True)
def hopcroft_karp(nl, nr, ptr, adj):
    """Maximum bipartite matching size (left nl, right nr, CSR adjacency)."""
    big = np.int64(1) << np.int64(40)
    matchL = np.full(nl, -1, np.int64)
    matchR = np.full(nr, -1, np.int64)
    res = 0
    for u in range(nl):
        for p in range(ptr[u], ptr[u + 1]):
            v = adj[p]
            if matchR[v] == -1:
                matchL[u] = v
                matchR[v] = u
                res += 1
                break
    dist = np.empty(nl, np.int64)
    queue = np.empty(nl, np.int64)
    it = np.empty(nl, np.int64)
    stack = np.empty(nl + 1, np.int64)
    vstack = np.empty(nl + 1, np.int64)
    while True:
        qh = 0
        qt = 0
        for u in range(nl):
            if matchL[u] == -1:
                dist[u] = 0
                queue[qt] = u
                qt += 1
            else:
                dist[u] = big
        found = False
        while qh < qt:
            u = queue[qh]
            qh += 1
            for p in range(ptr[u], ptr[u + 1]):
                w = matchR[adj[p]]
                if w == -1:
                    found = True
                elif dist[w] == big:
                    dist[w] = dist[u] + 1
                    queue[qt] = w
                    qt += 1
        if not found:
            break
        for u in range(nl):
            it[u] = ptr[u]
        for s in range(nl):
            if matchL[s] != -1:
                continue
            top = 0
            stack[0] = s
            while top >= 0:
                u = stack[top]
                if it[u] == ptr[u + 1]:
                    dist[u] = big
                    top -= 1
                    if top >= 0:
                        it[stack[top]] += 1
                    continue
                v = adj[it[u]]
                w = matchR[v]
                if w == -1:
                    vstack[top] = v
                    for t in range(top + 1):
                        matchL[stack[t]] = vstack[t]
                        matchR[vstack[t]] = stack[t]
                    res += 1
                    break
                elif dist[w] == dist[u] + 1:
                    vstack[top] = v
                    top += 1
                    stack[top] = w
                else:
                    it[u] += 1
    return res


# ---------------------------------------------------------- edge statistics

N_STATS = 8  # tri, nUi, nUj, Xi, varpi, varpi_legacy, matching, xi_i


@njit(cache=True)
def edge_stats_one(indptr, indices, i, j, legacy, matching=True):
    """2-hop statistics of edge (i, j).

    Returns int64[8]: triangles, |U_i|, |U_j|, Xi, varpi_max (paper/Topping
    definition: closers of k in U_i inside U_j), varpi_max as computed by the
    preliminary code (over N(i)\\{j} into N(j)\\{i}; only if ``legacy``),
    maximum matching of the cross-edge graph B_ij, and |xi_i|.
    """
    out = np.zeros(N_STATS, np.int64)
    a0 = indptr[i]
    a1 = indptr[i + 1]
    b0 = indptr[j]
    b1 = indptr[j + 1]
    # unique neighborhoods
    Ui = np.empty(a1 - a0, np.int64)
    Uj = np.empty(b1 - b0, np.int64)
    nui = 0
    nuj = 0
    tri = 0
    p = a0
    q = b0
    while p < a1 or q < b1:
        if q >= b1 or (p < a1 and indices[p] < indices[q]):
            x = indices[p]
            if x != j:
                Ui[nui] = x
                nui += 1
            p += 1
        elif p >= a1 or indices[q] < indices[p]:
            y = indices[q]
            if y != i:
                Uj[nuj] = y
                nuj += 1
            q += 1
        else:
            tri += 1
            p += 1
            q += 1
    out[0] = tri
    out[1] = nui
    out[2] = nuj
    # cross edges U_i -- U_j
    ptr = np.zeros(nui + 1, np.int64)
    for a in range(nui):
        k = Ui[a]
        ptr[a + 1] = ptr[a] + _count_intersect(indices, indptr[k], indptr[k + 1], Uj, 0, nuj)
    adj = np.empty(ptr[nui], np.int64)
    rcnt = np.zeros(nuj, np.int64)
    varpi = 0
    xi_i = 0
    for a in range(nui):
        k = Ui[a]
        c = ptr[a + 1] - ptr[a]
        if c > 0:
            xi_i += 1
            if c > varpi:
                varpi = c
            pos = ptr[a]
            k0 = indptr[k]
            k1 = indptr[k + 1]
            if (k1 - k0) <= nuj:
                for t in range(k0, k1):
                    b = _find(Uj[:nuj], indices[t])
                    if b >= 0:
                        adj[pos] = b
                        pos += 1
                        rcnt[b] += 1
            else:
                for b in range(nuj):
                    if _contains(indices, k0, k1, Uj[b]):
                        adj[pos] = b
                        pos += 1
                        rcnt[b] += 1
    xi_j = 0
    for b in range(nuj):
        if rcnt[b] > 0:
            xi_j += 1
            if rcnt[b] > varpi:
                varpi = rcnt[b]
    out[3] = xi_i + xi_j
    out[4] = varpi
    out[7] = xi_i
    if matching and ptr[nui] > 0:
        out[6] = hopcroft_karp(nui, nuj, ptr, adj)
    if legacy:
        # max_k in N(i)\{j} |N(k) & (N(j)\{i})| and symmetrically
        best = 0
        for t in range(a0, a1):
            k = indices[t]
            if k == j:
                continue
            c = _count_intersect(indices, indptr[k], indptr[k + 1], indices, b0, b1)
            if _contains(indices, indptr[k], indptr[k + 1], i):
                c -= 1
            if c > best:
                best = c
        for t in range(b0, b1):
            w = indices[t]
            if w == i:
                continue
            c = _count_intersect(indices, indptr[w], indptr[w + 1], indices, a0, a1)
            if _contains(indices, indptr[w], indptr[w + 1], j):
                c -= 1
            if c > best:
                best = c
        out[5] = best
    return out


@njit(cache=True, parallel=True)
def edge_stats_all(indptr, indices, edges, legacy, matching=True):
    M = edges.shape[0]
    out = np.zeros((M, N_STATS), np.int64)
    for e in prange(M):
        out[e] = edge_stats_one(indptr, indices, edges[e, 0], edges[e, 1], legacy, matching)
    return out


# ===================================================== exact transport

U0 = np.uint64(0)
U1 = np.uint64(1)
N_EXACT = 5  # c_OR, c_OR0, c_LLY, TLW dual (lazy), TLW dual (non-lazy)
_BLOCK_BUDGET = 1 << 22  # words of middle-vertex bitsets held at once (32 MB)
_WARM_MIN_ARCS = 300  # token networks at least this large get the G1 warm start


@njit(cache=True, inline="always")
def _take_i(buf, off, n):
    """Carve n int64 from the arena ``buf`` at ``off`` (grow it if needed;
    views carved earlier stay valid)."""
    if off + n > buf.shape[0]:
        buf = np.empty(max(2 * buf.shape[0], 2 * n + 64), np.int64)
        off = 0
    return buf, off + n, buf[off:off + n]


@njit(cache=True, inline="always")
def _take_u(buf, off, n):
    if off + n > buf.shape[0]:
        buf = np.empty(max(2 * buf.shape[0], 2 * n + 64), np.uint64)
        off = 0
    return buf, off + n, buf[off:off + n]


@njit(cache=True, inline="always")
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, inline="always")
def _ctz(x):
    """Index of the lowest set bit of x != 0."""
    return _popcount((x & (~x + U1)) - U1)


@njit(cache=True, inline="always")
def _setbit(arr, base, k):
    arr[base + (k >> 6)] |= U1 << np.uint64(k & 63)


@njit(cache=True, inline="always")
def _getbit(arr, base, k):
    return ((arr[base + (k >> 6)] >> np.uint64(k & 63)) & U1) != U0


# ------------------------------------------------------------- max flow


@njit(cache=True)
def _csr(nV, nA, tail, head, capv, bi, oi):
    """CSR residual network of the arcs (tail[k], head[k], capv[k]), k < nA:
    arcs of a node contiguous, paired reverse arcs; pos[k] is the position of
    arc k. Returns (start, to, cap, rev, pos, arena)."""
    bi, oi, start = _take_i(bi, oi, nV + 1)
    bi, oi, fill = _take_i(bi, oi, nV)
    bi, oi, to = _take_i(bi, oi, 2 * nA)
    bi, oi, cap = _take_i(bi, oi, 2 * nA)
    bi, oi, rev = _take_i(bi, oi, 2 * nA)
    bi, oi, pos = _take_i(bi, oi, nA)
    for v in range(nV + 1):
        start[v] = 0
    for k in range(nA):
        start[tail[k] + 1] += 1
        start[head[k] + 1] += 1
    for v in range(nV):
        start[v + 1] += start[v]
    for v in range(nV):
        fill[v] = start[v]
    for k in range(nA):
        u = tail[k]
        v = head[k]
        a = fill[u]
        fill[u] = a + 1
        b = fill[v]
        fill[v] = b + 1
        to[a] = v
        cap[a] = capv[k]
        rev[a] = b
        to[b] = u
        cap[b] = 0
        rev[b] = a
        pos[k] = a
    return start, to, cap, rev, pos, bi, oi


@njit(cache=True)
def _dinic(nV, start, to, cap, rev, s, t, bi, oi):
    """Dinic from the current residual capacities (any feasible flow), with
    an iterative DFS and current-arc pointers. Returns (added flow, arena)."""
    bi, oi, level = _take_i(bi, oi, nV)
    bi, oi, it = _take_i(bi, oi, nV)
    bi, oi, q = _take_i(bi, oi, nV)
    bi, oi, path = _take_i(bi, oi, nV)
    flow = np.int64(0)
    while True:
        for v in range(nV):
            level[v] = -1
        level[s] = 0
        q[0] = s
        qh = 0
        qt = 1
        while qh < qt:
            u = q[qh]
            qh += 1
            lu = level[u] + 1
            for a in range(start[u], start[u + 1]):
                if cap[a] > 0:
                    v = to[a]
                    if level[v] < 0:
                        level[v] = lu
                        q[qt] = v
                        qt += 1
            if level[t] >= 0:
                break  # BFS is level by level: everything below t's level is labeled
        if level[t] < 0:
            break
        lt = level[t]
        for v in range(nV):
            it[v] = start[v]
            if level[v] >= lt and v != t:
                level[v] = -1  # nodes at or beyond t's level cannot reach t
        top = 0
        u = s
        while True:
            if u == t:
                f = INF
                for p in range(top):
                    c = cap[path[p]]
                    if c < f:
                        f = c
                for p in range(top):
                    a = path[p]
                    cap[a] -= f
                    cap[rev[a]] += f
                flow += f
                p0 = 0
                while cap[path[p0]] > 0:
                    p0 += 1
                top = p0
                u = s if top == 0 else to[path[top - 1]]
                continue
            a = it[u]
            end = start[u + 1]
            lu = level[u] + 1
            while a < end:
                if cap[a] > 0 and level[to[a]] == lu:
                    break
                a += 1
            it[u] = a
            if a < end:
                path[top] = a
                top += 1
                u = to[a]
            else:
                level[u] = -1
                if top == 0:
                    break
                top -= 1
                u = to[rev[path[top]]]
                it[u] += 1
    return flow, bi, oi


@njit(cache=True)
def _maxflow(nV, nA, tail, head, capv, s, t, bi, oi):
    """Max flow of the network with arcs (tail[k], head[k], capv[k])."""
    start, to, cap, rev, pos, bi, oi = _csr(nV, nA, tail, head, capv, bi, oi)
    return _dinic(nV, start, to, cap, rev, s, t, bi, oi)


# ------------------------------------------------------ twin merging


@njit(cache=True)
def _dedupe(keys, nrows, KW, bi, oi):
    """Group equal rows of the (nrows, KW) uint64 key matrix (flat) by
    hashing with open addressing. Returns (group id per row, number of
    groups, leader row of each group, arena)."""
    tsize = 4
    while tsize < 2 * nrows:
        tsize <<= 1
    mask = np.uint64(tsize - 1)
    bi, oi, table = _take_i(bi, oi, tsize)
    bi, oi, gid = _take_i(bi, oi, nrows)
    bi, oi, leader = _take_i(bi, oi, nrows)
    for x in range(tsize):
        table[x] = -1
    ng = 0
    for r in range(nrows):
        base = r * KW
        h = np.uint64(1469598103934665603)
        for x in range(KW):
            h = (h ^ keys[base + x]) * np.uint64(1099511628211)
        h ^= h >> np.uint64(29)
        pos = np.int64(h & mask)
        while True:
            g = table[pos]
            if g < 0:
                table[pos] = ng
                leader[ng] = r
                gid[r] = ng
                ng += 1
                break
            lb = leader[g] * KW
            same = True
            for x in range(KW):
                if keys[lb + x] != keys[base + x]:
                    same = False
                    break
            if same:
                gid[r] = g
                break
            pos = (pos + 1) & (tsize - 1)
    return gid, ng, leader, bi, oi


# ------------------------------------------------- one measure on one edge


@njit(cache=True, inline="always")
def _rmass(r, nUi, mU, m_i, m_j, m_T):
    """Net mass of row r (rows: U_i, then i, j, triangle vertices)."""
    if r < nUi:
        return mU
    if r == nUi:
        return m_i
    if r == nUi + 1:
        return m_j
    return m_T


@njit(cache=True, inline="always")
def _cmass(k, nUj, mV, m_i, m_j, m_T):
    """Demand of column k (columns: U_j, then i, j, triangle vertices);
    positive iff the column is in N."""
    if k < nUj:
        return mV
    if k == nUj:
        return -m_i
    if k == nUj + 1:
        return -m_j
    return -m_T


@njit(cache=True)
def _measure(Adj, Cov, fullr, colcov, colfull, UjMask, nUi, nUj, nT, W,
             mU, mV, m_i, m_j, m_T, need_w1, need_tlw, bi, oi, bu, ou):
    """W1 (scaled by the common denominator of the masses) and the TLW dual
    certificate for the measure with net masses mU on U_i, -mV on U_j, m_i on
    i, m_j on j, m_T on each triangle vertex (mU, mV > 0). Returns
    (W1, TLW, arenas)."""
    nC = 2 + nT
    nr = nUi + nC
    nc = nUj + nC
    A = nUi * mU + max(m_i, 0) + max(m_j, 0) + nT * max(m_T, 0)
    if A == 0:
        return np.int64(0), np.int64(0), bi, bu
    bu, ou, Nm = _take_u(bu, ou, W)  # columns in N
    for x in range(W):
        Nm[x] = UjMask[x]
    if m_i < 0:
        _setbit(Nm, 0, nUj)
    if m_j < 0:
        _setbit(Nm, 0, nUj + 1)
    if m_T < 0:
        for t in range(nT):
            _setbit(Nm, 0, nUj + 2 + t)
    nNC = (1 if m_i < 0 else 0) + (1 if m_j < 0 else 0) + (nT if m_T < 0 else 0)
    nPC = (1 if m_i > 0 else 0) + (1 if m_j > 0 else 0) + (nT if m_T > 0 else 0)

    tlw = np.int64(0)
    if need_tlw:
        # sum_P w min_N d  and  sum_N |w| min_P d, from adjacency and coverage
        bu, ou, colP = _take_u(bu, ou, W)
        for x in range(W):
            colP[x] = U0
        sP = np.int64(0)
        for r in range(nr):
            mr = _rmass(r, nUi, mU, m_i, m_j, m_T)
            if mr <= 0:
                continue
            has = False
            for x in range(W):
                z = Adj[r * W + x] & Nm[x]
                colP[x] |= z
                if z != U0:
                    has = True
            if has:
                d = 1
            elif r >= nUi or nNC > 0:
                d = 2
            else:
                d = 3
                for x in range(W):
                    if Cov[r * W + x] != U0:
                        d = 2
                        break
            sP += mr * d
        sN = np.int64(0)
        for k in range(nc):
            mk = _cmass(k, nUj, mV, m_i, m_j, m_T)
            if mk <= 0:
                continue
            if _getbit(colP, 0, k):
                d = 1
            elif k >= nUj or nPC > 0 or _getbit(colcov, 0, k):
                d = 2
            else:
                d = 3
            sN += mk * d
        tlw = max(sP, sN)
    if not need_w1:
        return np.int64(0), tlw, bi, bu

    nfull = 0
    for r in range(nUi):
        nfull += fullr[r]

    if nfull == nUi:
        # -------- fast path: every P x N pair is within distance 2
        bu, ou, keys = _take_u(bu, ou, nr * W)
        bi, oi, rid = _take_i(bi, oi, nr)
        ncand = 0
        for r in range(nr):
            if _rmass(r, nUi, mU, m_i, m_j, m_T) <= 0:
                continue
            nz = False
            for x in range(W):
                z = Adj[r * W + x] & Nm[x]
                keys[ncand * W + x] = z
                if z != U0:
                    nz = True
            if nz:
                rid[ncand] = r
                ncand += 1
        if ncand == 0:
            return 2 * A, tlw, bi, bu
        gid, ng, leader, bi, oi = _dedupe(keys, ncand, W, bi, oi)
        bi, oi, gm = _take_i(bi, oi, ng)
        for g in range(ng):
            gm[g] = 0
        for c in range(ncand):
            gm[gid[c]] += _rmass(rid[c], nUi, mU, m_i, m_j, m_T)
        bi, oi, cnode = _take_i(bi, oi, nc)
        for k in range(nc):
            cnode[k] = -1
        nV = 2 + ng
        nA = ng
        for g in range(ng):
            lb = leader[g] * W
            for x in range(W):
                z = keys[lb + x]
                while z != U0:
                    k = x * 64 + _ctz(z)
                    if cnode[k] < 0:
                        cnode[k] = nV
                        nV += 1
                        nA += 1
                    nA += 1
                    z &= z - U1
        bi, oi, tl = _take_i(bi, oi, nA)
        bi, oi, hd = _take_i(bi, oi, nA)
        bi, oi, cp = _take_i(bi, oi, nA)
        m = 0
        for g in range(ng):
            tl[m] = 0
            hd[m] = 2 + g
            cp[m] = gm[g]
            m += 1
            lb = leader[g] * W
            for x in range(W):
                z = keys[lb + x]
                while z != U0:
                    k = x * 64 + _ctz(z)
                    tl[m] = 2 + g
                    hd[m] = cnode[k]
                    cp[m] = INF
                    m += 1
                    z &= z - U1
        for k in range(nc):
            if cnode[k] >= 0:
                tl[m] = cnode[k]
                hd[m] = 1
                cp[m] = _cmass(k, nUj, mV, m_i, m_j, m_T)
                m += 1
        if m != nA:
            raise RuntimeError("internal error: bipartite network arc count")
        F, bi, oi = _maxflow(nV, nA, tl, hd, cp, 0, 1, bi, oi)
        return 2 * A - F, tlw, bi, bu

    # -------- general path: token network (p, q per row; r, t per column)
    KW = 2 * W + 1
    bu, ou, keys = _take_u(bu, ou, nr * KW)
    bi, oi, rid = _take_i(bi, oi, nr)
    ncand = 0
    for r in range(nr):
        if _rmass(r, nUi, mU, m_i, m_j, m_T) <= 0:
            continue
        b0 = ncand * KW
        for x in range(W):
            keys[b0 + x] = Adj[r * W + x] & Nm[x]
        if r < nUi:
            for x in range(W):
                keys[b0 + W + x] = Cov[r * W + x]
            keys[b0 + 2 * W] = U0
        else:
            for x in range(W):
                keys[b0 + W + x] = U0
            keys[b0 + 2 * W] = U1
        rid[ncand] = r
        ncand += 1
    gid, ng, leader, bi, oi = _dedupe(keys, ncand, KW, bi, oi)
    bi, oi, gm = _take_i(bi, oi, ng)
    bi, oi, gkind = _take_i(bi, oi, ng)  # 0 U_i row, 1 full U_i row, 2 C row
    for g in range(ng):
        gm[g] = 0
        L = leader[g]
        if keys[L * KW + 2 * W] == U1:
            gkind[g] = 2
        else:
            gkind[g] = 1 if fullr[rid[L]] == 1 else 0
    for c in range(ncand):
        gm[gid[c]] += _rmass(rid[c], nUi, mU, m_i, m_j, m_T)
    bi, oi, cidx = _take_i(bi, oi, nc)
    nN = 0
    for k in range(nc):
        if _cmass(k, nUj, mV, m_i, m_j, m_T) > 0:
            cidx[k] = nN
            nN += 1
        else:
            cidx[k] = -1
    n_rowfull = 0
    n_rowpart = 0
    n_rowC = 0
    for g in range(ng):
        if gkind[g] == 1:
            n_rowfull += 1
        elif gkind[g] == 0:
            n_rowpart += 1
        else:
            n_rowC += 1
    n_colfull = 0
    for x in range(W):
        n_colfull += _popcount(colfull[x])
    use_h4 = n_colfull > 0 and n_rowpart > 0
    # nodes: s=0, t=1, hubs 2..5, rows (p, q), columns (r, t)
    H1 = 2  # every P row -> columns of N in C
    H2 = 3  # rows of P in C -> every column
    H3 = 4  # fully covered U_i rows -> every U_j column
    H4 = 5  # every partial U_i row -> fully covered U_j columns
    cbase = 6 + 2 * ng
    nV = cbase + 2 * nN
    nA = 3 * ng + 2 * nN
    for g in range(ng):
        lb = leader[g] * KW
        for x in range(W):
            nA += 2 * _popcount(keys[lb + x])
            if gkind[g] == 0:
                nA += _popcount(keys[lb + W + x] & ~keys[lb + x] & ~colfull[x])
    if nNC > 0:
        nA += ng + nNC
    if nPC > 0:
        nA += n_rowC + nN
    if n_rowfull > 0:
        nA += n_rowfull + nUj
    if use_h4:
        nA += n_rowpart + n_colfull
    # warm start: a maximum flow on the adjacency G1 alone (one layer, small).
    # Some optimal plan contains a maximum G1 flow (moving a unit onto G1 saves
    # 1 and forces at most 1 unit onto a distance-3 pair), and that flow loaded
    # on both layers (p -> r and q -> t) is a feasible token flow of value 2 F1,
    # so Dinic on the token network only has to add the rest. On small
    # networks the extra flow costs more than the Dinic phases it saves.
    warm = nA >= _WARM_MIN_ARCS
    nG1 = 0
    for g in range(ng):
        lb = leader[g] * KW
        for x in range(W):
            nG1 += _popcount(keys[lb + x])
    nV1 = 2 + ng + nN if warm else 2  # an empty stub when not warm
    nA1 = ng + nG1 + nN if warm else 0
    bi, oi, tl1 = _take_i(bi, oi, nA1)
    bi, oi, hd1 = _take_i(bi, oi, nA1)
    bi, oi, cp1 = _take_i(bi, oi, nA1)
    if warm:
        m = 0
        for g in range(ng):
            tl1[m] = 0; hd1[m] = 2 + g; cp1[m] = gm[g]; m += 1
        for g in range(ng):
            lb = leader[g] * KW
            for x in range(W):
                z = keys[lb + x]
                while z != U0:
                    tl1[m] = 2 + g; hd1[m] = 2 + ng + cidx[x * 64 + _ctz(z)]; cp1[m] = INF; m += 1
                    z &= z - U1
        for k in range(nc):
            if cidx[k] >= 0:
                tl1[m] = 2 + ng + cidx[k]; hd1[m] = 1; cp1[m] = _cmass(k, nUj, mV, m_i, m_j, m_T); m += 1
    st1, to1, ca1, rv1, ps1, bi, oi = _csr(nV1, nA1, tl1, hd1, cp1, bi, oi)
    F1 = np.int64(0)
    if warm:
        F1, bi, oi = _dinic(nV1, st1, to1, ca1, rv1, 0, 1, bi, oi)

    bi, oi, tl = _take_i(bi, oi, nA)
    bi, oi, hd = _take_i(bi, oi, nA)
    bi, oi, cp = _take_i(bi, oi, nA)
    bi, oi, tsp = _take_i(bi, oi, ng)    # token index of s -> p_g (s -> q_g follows)
    bi, oi, tg1 = _take_i(bi, oi, nG1)   # token index of p_g -> r_c (q_g -> t_c follows)
    bi, oi, tcol = _take_i(bi, oi, nN)   # token index of r_c -> t (t_c -> t follows)
    m = 0
    jj = 0
    for g in range(ng):
        pn = 6 + 2 * g
        qn = pn + 1
        tsp[g] = m
        tl[m] = 0; hd[m] = pn; cp[m] = gm[g]; m += 1
        tl[m] = 0; hd[m] = qn; cp[m] = gm[g]; m += 1
        tl[m] = qn; hd[m] = pn; cp[m] = INF; m += 1
        lb = leader[g] * KW
        for x in range(W):
            z = keys[lb + x]
            while z != U0:  # adjacent pairs: both implications
                c = cidx[x * 64 + _ctz(z)]
                tg1[jj] = m
                jj += 1
                tl[m] = pn; hd[m] = cbase + 2 * c; cp[m] = INF; m += 1
                tl[m] = qn; hd[m] = cbase + 2 * c + 1; cp[m] = INF; m += 1
                z &= z - U1
            if gkind[g] == 0:  # covered, non-adjacent, column not fully covered
                z = keys[lb + W + x] & ~keys[lb + x] & ~colfull[x]
                while z != U0:
                    c = cidx[x * 64 + _ctz(z)]
                    tl[m] = qn; hd[m] = cbase + 2 * c; cp[m] = INF; m += 1
                    z &= z - U1
        if nNC > 0:
            tl[m] = qn; hd[m] = H1; cp[m] = INF; m += 1
        if gkind[g] == 2 and nPC > 0:
            tl[m] = qn; hd[m] = H2; cp[m] = INF; m += 1
        if gkind[g] == 1:
            tl[m] = qn; hd[m] = H3; cp[m] = INF; m += 1
        if gkind[g] == 0 and use_h4:
            tl[m] = qn; hd[m] = H4; cp[m] = INF; m += 1
    for k in range(nc):
        c = cidx[k]
        if c < 0:
            continue
        rn = cbase + 2 * c
        tn = rn + 1
        mk = _cmass(k, nUj, mV, m_i, m_j, m_T)
        tcol[c] = m
        tl[m] = rn; hd[m] = 1; cp[m] = mk; m += 1
        tl[m] = tn; hd[m] = 1; cp[m] = mk; m += 1
        # no t -> r arc: flow q -> t -> r can always be rerouted q -> p -> r
        if nNC > 0 and k >= nUj:
            tl[m] = H1; hd[m] = rn; cp[m] = INF; m += 1
        if nPC > 0:
            tl[m] = H2; hd[m] = rn; cp[m] = INF; m += 1
        if n_rowfull > 0 and k < nUj:
            tl[m] = H3; hd[m] = rn; cp[m] = INF; m += 1
        if use_h4 and k < nUj and _getbit(colfull, 0, k):
            tl[m] = H4; hd[m] = rn; cp[m] = INF; m += 1
    if m != nA or jj != nG1:
        raise RuntimeError("internal error: token network arc count")
    start, to, cap, rev, pos, bi, oi = _csr(nV, nA, tl, hd, cp, bi, oi)
    for g in range(ng if warm else 0):  # load the G1 flow on both layers
        f = ca1[rv1[ps1[g]]]
        if f > 0:
            for d in range(2):
                e = pos[tsp[g] + d]
                cap[e] -= f
                cap[rev[e]] += f
    for x in range(nG1 if warm else 0):
        f = ca1[rv1[ps1[ng + x]]]
        if f > 0:
            for d in range(2):
                e = pos[tg1[x] + d]
                cap[e] -= f
                cap[rev[e]] += f
    for c in range(nN if warm else 0):
        f = ca1[rv1[ps1[ng + nG1 + c]]]
        if f > 0:
            for d in range(2):
                e = pos[tcol[c] + d]
                cap[e] -= f
                cap[rev[e]] += f
    F2, bi, oi = _dinic(nV, start, to, cap, rev, 0, 1, bi, oi)
    return 3 * A - (2 * F1 + F2), tlw, bi, bu


# --------------------------------------------------------- one edge


@njit(cache=True)
def _edge(indptr, indices, i0, j0, mode, want_or, want_or0, want_lly, tau, tol,
          slot, sstamp, epoch, bi, bu, out, budget):
    """All requested quantities for edge (i0, j0), written into ``out``.

    mode 0: exact curvatures (``want_*``) and the TLW duals that come with
    them, laid out as ``N_EXACT``; mode 1: certify-then-solve for
    {c_OR < tau}: out[1] = TLW upper bound, out[0] = exact c_OR unless the
    bound decides the edge; mode 2: out[0] = TLW upper bound on c_OR.
    ``slot``/``sstamp`` are O(n) per-thread scratch stamped with ``epoch``;
    ``budget`` caps the words of middle-vertex bitsets held at once (columns
    are processed in blocks beyond it). Returns the (possibly grown) arenas.
    """
    # orient so that the larger neighbourhood is on the row side
    if indptr[i0 + 1] - indptr[i0] >= indptr[j0 + 1] - indptr[j0]:
        i = i0
        j = j0
    else:
        i = j0
        j = i0
    di = np.int64(indptr[i + 1] - indptr[i])
    dj = np.int64(indptr[j + 1] - indptr[j])
    oi = 0
    ou = 0
    bi, oi, Ui = _take_i(bi, oi, di)
    bi, oi, Uj = _take_i(bi, oi, dj)
    bi, oi, Tt = _take_i(bi, oi, dj)
    nUi = 0
    nUj = 0
    nT = 0
    p = indptr[i]
    pe = indptr[i + 1]
    q = indptr[j]
    qe = indptr[j + 1]
    while p < pe or q < qe:
        if q >= qe or (p < pe and indices[p] < indices[q]):
            x = indices[p]
            if x != j:
                Ui[nUi] = x
                nUi += 1
            p += 1
        elif p >= pe or indices[q] < indices[p]:
            y = indices[q]
            if y != i:
                Uj[nUj] = y
                nUj += 1
            q += 1
        else:
            Tt[nT] = indices[p]
            nT += 1
            p += 1
            q += 1
    nC = 2 + nT
    nr = nUi + nC
    nc = nUj + nC
    W = (nc + 63) >> 6
    bi, oi, colv = _take_i(bi, oi, nc)
    bi, oi, rowv = _take_i(bi, oi, nr)
    for k in range(nUj):
        colv[k] = Uj[k]
    for r in range(nUi):
        rowv[r] = Ui[r]
    colv[nUj] = i
    colv[nUj + 1] = j
    rowv[nUi] = i
    rowv[nUi + 1] = j
    for t in range(nT):
        colv[nUj + 2 + t] = Tt[t]
        rowv[nUi + 2 + t] = Tt[t]

    # one slot per middle vertex (any vertex adjacent to a column)
    ns = 0
    for k in range(nc):
        b = colv[k]
        for e in range(indptr[b], indptr[b + 1]):
            w = indices[e]
            if sstamp[w] != epoch:
                sstamp[w] = epoch
                slot[w] = ns
                ns += 1
    WB = W
    if ns * W > budget:
        WB = max(1, budget // max(ns, 1))

    bu, ou, UjMask = _take_u(bu, ou, W)
    for x in range(W):
        lo = x * 64
        if lo + 64 <= nUj:
            UjMask[x] = ~U0
        elif lo < nUj:
            UjMask[x] = (U1 << np.uint64(nUj - lo)) - U1
        else:
            UjMask[x] = U0
    bu, ou, Adj = _take_u(bu, ou, nr * W)  # columns adjacent to each row
    bu, ou, Cov = _take_u(bu, ou, nUi * W)  # U_j columns within distance 2
    bu, ou, R = _take_u(bu, ou, ns * WB)  # columns adjacent to each middle vertex
    for b0 in range(0, W, WB):
        wb = min(WB, W - b0)
        for z in range(ns * WB):
            R[z] = U0
        k_lo = b0 * 64
        k_hi = min(nc, (b0 + wb) * 64)
        for k in range(k_lo, k_hi):
            kk = k - k_lo
            wq = kk >> 6
            bit = U1 << np.uint64(kk & 63)
            b = colv[k]
            for e in range(indptr[b], indptr[b + 1]):
                R[slot[indices[e]] * WB + wq] |= bit
        for r in range(nr):
            a = rowv[r]
            base = r * W + b0
            if sstamp[a] == epoch:
                sb = slot[a] * WB
                for x in range(wb):
                    Adj[base + x] = R[sb + x]
            else:
                for x in range(wb):
                    Adj[base + x] = U0
        if k_lo >= nUj:
            for r in range(nUi):
                for x in range(wb):
                    Cov[r * W + b0 + x] = U0
            continue
        for r in range(nUi):
            cb = r * W + b0
            done = True
            for x in range(wb):
                Cov[cb + x] = Adj[cb + x]
                if (Cov[cb + x] & UjMask[b0 + x]) != UjMask[b0 + x]:
                    done = False
            if done:
                continue
            xv = rowv[r]
            for e in range(indptr[xv], indptr[xv + 1]):
                w = indices[e]
                if sstamp[w] != epoch:
                    continue
                sb = slot[w] * WB
                done = True
                for x in range(wb):
                    c = Cov[cb + x] | R[sb + x]
                    Cov[cb + x] = c
                    if (c & UjMask[b0 + x]) != UjMask[b0 + x]:
                        done = False
                if done:
                    break
    # mask coverage to U_j; full rows; columns covered by some / every row
    bi, oi, fullr = _take_i(bi, oi, nUi)
    bu, ou, colcov = _take_u(bu, ou, W)
    bu, ou, colfull = _take_u(bu, ou, W)
    for x in range(W):
        colcov[x] = U0
        colfull[x] = UjMask[x] if nUi > 0 else U0
    for r in range(nUi):
        f = 1
        for x in range(W):
            c = Cov[r * W + x] & UjMask[x]
            Cov[r * W + x] = c
            colcov[x] |= c
            colfull[x] &= c
            if c != UjMask[x]:
                f = 0
        fullr[r] = f

    if mode == 0:
        if want_or:
            L = (di + 1) * (dj + 1)
            Wd, T, bi, bu = _measure(Adj, Cov, fullr, colcov, colfull, UjMask, nUi, nUj, nT, W,
                                     dj + 1, di + 1, dj - di, dj - di, dj - di,
                                     True, True, bi, oi, bu, ou)
            out[0] = 1.0 - Wd / L
            out[3] = 1.0 - T / L
        if want_or0:
            L = di * dj
            Wd, T, bi, bu = _measure(Adj, Cov, fullr, colcov, colfull, UjMask, nUi, nUj, nT, W,
                                     dj, di, -di, dj, dj - di,
                                     True, True, bi, oi, bu, ou)
            out[1] = 1.0 - Wd / L
            out[4] = 1.0 - T / L
        if want_lly:
            dmax = di if di > dj else dj
            L = (dmax + 1) * di * dj
            Wd, T, bi, bu = _measure(Adj, Cov, fullr, colcov, colfull, UjMask, nUi, nUj, nT, W,
                                     dmax * dj, dmax * di, di * dj - dmax * di, dmax * dj - di * dj,
                                     dmax * (dj - di), True, False, bi, oi, bu, ou)
            kp = 1.0 - Wd / L
            out[2] = kp * (dmax + 1) / dmax
        return bi, bu
    L = (di + 1) * (dj + 1)
    Wd, T, bi, bu = _measure(Adj, Cov, fullr, colcov, colfull, UjMask, nUi, nUj, nT, W,
                             dj + 1, di + 1, dj - di, dj - di, dj - di,
                             False, True, bi, oi, bu, ou)
    tlw = 1.0 - T / L
    if mode == 2:
        out[0] = tlw
        return bi, bu
    out[1] = tlw
    if tlw + tol < tau:
        return bi, bu
    Wd, T, bi, bu = _measure(Adj, Cov, fullr, colcov, colfull, UjMask, nUi, nUj, nT, W,
                             dj + 1, di + 1, dj - di, dj - di, dj - di,
                             True, False, bi, oi, bu, ou)
    out[0] = 1.0 - Wd / L
    return bi, bu


# ------------------------------------------------------------- drivers


@njit(cache=True)
def _chunk(indptr, indices, edges, ids, lo, hi, mode, want_or, want_or0, want_lly, tau, tol, out, budget):
    """Edges ids[lo:hi] on one thread. The arenas are reassigned when they
    grow, so they must live here and not in the body of a ``prange`` loop:
    the parfor pass hoists loop-invariant allocations out of the parallel
    loop, and reassigning a hoisted array breaks its reference count."""
    n = indptr.shape[0] - 1
    slot = np.empty(n, np.int64)
    sstamp = np.full(n, -1, np.int64)
    bi = np.empty(1 << 12, np.int64)
    bu = np.empty(1 << 10, np.uint64)
    for k in range(lo, hi):
        e = ids[k]
        bi, bu = _edge(indptr, indices, edges[e, 0], edges[e, 1], mode, want_or, want_or0,
                       want_lly, tau, tol, slot, sstamp, k, bi, bu, out[k], budget)


@njit(cache=True, parallel=True)
def _run(indptr, indices, edges, ids, mode, want_or, want_or0, want_lly, tau, tol, width, nchunks,
         budget=_BLOCK_BUDGET):
    K = ids.shape[0]
    out = np.full((K, width), np.nan)
    if K == 0:
        return out
    nchunks = max(1, min(nchunks, K))
    step = (K + nchunks - 1) // nchunks
    for c in prange(nchunks):
        lo = c * step
        hi = min(K, lo + step)
        if lo < hi:
            _chunk(indptr, indices, edges, ids, lo, hi, mode, want_or, want_or0, want_lly, tau, tol, out, budget)
    return out


# The public drivers are Python functions on purpose: Numba launches its
# thread pool when a ``parallel=True`` function is dispatched from Python, and
# a parallel function first entered from inside compiled code runs on an
# uninitialized pool. They also fix the argument types, so that every caller
# reuses one compiled specialization of ``_run``.

def _ids(ids):
    return np.ascontiguousarray(ids, dtype=np.int64)


def exact_ids(indptr, indices, edges, ids, want_or, want_or0, want_lly, nchunks):
    """Exact curvatures for edges[ids] as a (len(ids), ``N_EXACT``) array:
    c_OR, c_OR0, c_LLY, TLW dual (lazy), TLW dual (non-lazy); entries not
    requested are NaN. Parallel over ``nchunks`` chunks with per-chunk scratch."""
    return _run(indptr, indices, edges, _ids(ids), 0, bool(want_or), bool(want_or0), bool(want_lly),
                0.0, 0.0, N_EXACT, int(nchunks))


def exact_or_ids(indptr, indices, edges, ids, nchunks):
    """Exact lazy c_OR only, for edges[ids] (same arithmetic as :func:`exact_ids`)."""
    return _run(indptr, indices, edges, _ids(ids), 0, True, False, False, 0.0, 0.0, N_EXACT,
                int(nchunks))[:, 0].copy()


def screen_ids(indptr, indices, edges, ids, tau, tol, nchunks):
    """Last stage of certify-then-solve for {c_OR < tau}: per edge (exact c_OR,
    or NaN if the TLW certificate decided it with tolerance ``tol``; TLW)."""
    return _run(indptr, indices, edges, _ids(ids), 1, True, False, False, float(tau), float(tol), 2,
                int(nchunks))


def tlw_lazy_ids(indptr, indices, edges, ids, nchunks):
    """TLW dual upper bound on the lazy curvature (no transport solve)."""
    return _run(indptr, indices, edges, _ids(ids), 2, True, False, False, 0.0, 0.0, 1,
                int(nchunks))[:, 0].copy()
