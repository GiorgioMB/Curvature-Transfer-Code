"""Numba kernels of the bounds: the 2-hop statistics of every edge.

``edge_stats``: the 2-hop statistics that parameterize BF curvature and all
combinatorial bounds (triangles, unique neighborhoods, 4-cycle counts, and the
maximum matching of the cross-edge graph, via Hopcroft--Karp), and the numerator
of the lazy TLW dual certificate (one stamp-based pass, no local distance
structure).

Exact curvatures come from the solver package ``orcut`` (pinned to 1.0.0, the
solver used for every exact value of the paper): its drivers are re-exported
here (``exact_ids``, ``exact_or_ids``, ``screen_ids``, ``tlw_lazy_ids_local``),
so ``_kernels.exact_ids`` and the rest of this package are unchanged.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# Exact transport (orcut 1.0.0). ``_take_i`` is the arena allocator shared by
# the statistics kernel; ``_nchunks``/``_ids`` normalize the driver arguments.
from orcut._kernels import (  # noqa: F401  (re-exported)
    N_EXACT,
    U0,
    U1,
    _ids,
    _nchunks,
    _run,
    _take_i,
    exact_ids,
    exact_or_ids,
    screen_ids,
    tlw_lazy_ids_local,
)

INF = np.int64(1) << np.int64(60)



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

N_STATS = 9  # tri, nUi, nUj, Xi, varpi, varpi_legacy, matching, xi_i, TLW numerator (lazy)




@njit(cache=True)
def edge_stats_one(indptr, indices, i, j, legacy, matching=True):
    """2-hop statistics of edge (i, j): the direct implementation (sorted-list
    intersections, allocations per edge), kept as the test oracle of the
    production kernel :func:`_stats_edge` (column 8 is always 0 here).

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


# ================================================ 2-hop statistics (fast)
#
# Production kernel for the 2-hop statistics (BF curvature and every
# combinatorial bound) and for the lazy TLW dual certificate. Per edge, each
# neighbour list of the scanned side is read once, with O(1) membership tests
# against per-thread vertex stamps (instead of one sorted-list intersection per
# vertex, followed by a second pass of binary searches), and nothing is
# allocated in steady state (arena scratch, as in the exact solver).
#
# Lazy TLW dual without the local distance structure. Orient (i, j) as the exact
# solver does (deg i >= deg j, ties keep the stored order) and scale the masses
# by L = (rho_i + 1)(rho_j + 1): +(rho_j + 1) on U_i, -(rho_i + 1) on U_j and
# rho_j - rho_i <= 0 on i, j and each triangle vertex. The certificate of
# ``_measure`` is max(sP, sN) with sP = sum_P mass * dist(x, N) and sN = sum_N
# |mass| * dist(y, P). Every pair of S = N(i) u N(j) is within distance 2 except
# pairs in U_i x U_j, which are at distance 1 (cross edge), 2 (a common
# neighbour anywhere in the graph) or 3; so the certificate needs, per vertex of
# U_i and U_j, only whether it has a cross edge and, if not, whether it has a
# neighbour in N(U_j), respectively N(U_i):
#   rho_i > rho_j: every x in U_i is adjacent to i in N, so sP = |U_i|(rho_j + 1);
#     sN = (rho_i + 1) sum_{y in U_j} d(y, U_i) + (rho_i - rho_j)(1 + 2 + sum_t d_t),
#     with d(i, U_i) = 1, d(j, U_i) = 2, d_t = 1 if t is adjacent to U_i, else 2;
#   rho_i = rho_j: C carries no mass; sP = (rho + 1) sum_{x in U_i} d(x, U_j),
#     sN = (rho + 1) sum_{y in U_j} d(y, U_i).
# The vertices adjacent to U_i are stamped while U_i is scanned for cross edges,
# so the certificate costs one probe per vertex of U_j and of T on top of the
# statistics (plus one scan of U_j when the degrees are equal). Tested bit for
# bit against the certificate of the exact solver.

_STATS_BLOCK = 64  # consecutive edges per block; blocks are dealt to the chunks cyclically
_HUB = 8  # a neighbour list longer than _HUB x |other side| is intersected by binary search


@njit(cache=True)
def _grow_i(buf, off, arr, n, need):
    """A view of length max(2 len(arr), need), carved from the arena, holding arr[:n]."""
    buf, off, new = _take_i(buf, off, max(2 * arr.shape[0], need))
    for t in range(n):
        new[t] = arr[t]
    return buf, off, new


@njit(cache=True)
def _hk(nl, nr, ptr, adj, bound, bi, oi):
    """Size of a maximum matching of the bipartite graph (left nl, right nr,
    CSR ptr/adj), by Hopcroft--Karp on arena scratch (same algorithm as
    :func:`hopcroft_karp`). Stops as soon as the matching reaches ``bound``, an
    upper bound on its size (e.g. the number of left vertices with an edge)."""
    big = np.int64(1) << np.int64(40)
    bi, oi, matchL = _take_i(bi, oi, nl)
    bi, oi, matchR = _take_i(bi, oi, nr)
    for u in range(nl):
        matchL[u] = -1
    for v in range(nr):
        matchR[v] = -1
    res = 0
    for u in range(nl):
        for p in range(ptr[u], ptr[u + 1]):
            v = adj[p]
            if matchR[v] == -1:
                matchL[u] = v
                matchR[v] = u
                res += 1
                break
    if res >= bound:
        return res, bi, oi
    bi, oi, dist = _take_i(bi, oi, nl)
    bi, oi, queue = _take_i(bi, oi, nl)
    bi, oi, it = _take_i(bi, oi, nl)
    bi, oi, stack = _take_i(bi, oi, nl + 1)
    bi, oi, vstack = _take_i(bi, oi, nl + 1)
    while res < bound:
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
    return res, bi, oi


@njit(cache=True)
def _stats_edge(indptr, indices, i0, j0, legacy, matching, tlw, ep, sO, pO, mS, mO, bi, out):
    """2-hop statistics of edge (i0, j0) written into ``out`` (the layout of
    :func:`edge_stats_one`, in the orientation (i0, j0)), plus out[8] = the
    numerator of the lazy TLW dual certificate (over (rho_i + 1)(rho_j + 1)) if
    ``tlw``. ``sO``/``pO`` (membership in and position on the side that is not
    scanned) and ``mS``/``mO`` (vertices adjacent to the scanned / the other
    side) are O(n) per-thread stamps, valid where they equal ``ep``. Returns the
    (possibly grown) arena."""
    # orient as the exact solver: i has the larger degree (ties: i0)
    if indptr[i0 + 1] - indptr[i0] >= indptr[j0 + 1] - indptr[j0]:
        i = i0
        j = j0
        flip = False
    else:
        i = j0
        j = i0
        flip = True
    di = indptr[i + 1] - indptr[i]
    dj = indptr[j + 1] - indptr[j]
    e32 = np.int32(ep)
    oi = 0
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
    while p < pe and q < qe:
        x = indices[p]
        y = indices[q]
        if x < y:
            if x != j:
                Ui[nUi] = x
                nUi += 1
            p += 1
        elif y < x:
            if y != i:
                Uj[nUj] = y
                nUj += 1
            q += 1
        else:
            Tt[nT] = x
            nT += 1
            p += 1
            q += 1
    while p < pe:
        x = indices[p]
        if x != j:
            Ui[nUi] = x
            nUi += 1
        p += 1
    while q < qe:
        y = indices[q]
        if y != i:
            Uj[nUj] = y
            nUj += 1
        q += 1
    out[0] = nT
    out[1] = nUj if flip else nUi
    out[2] = nUi if flip else nUj
    # scanned side: U_i if the certificate is wanted (it needs the vertices
    # adjacent to U_i), otherwise the side with the smaller neighbourhood volume
    scan_i = True
    if not tlw:
        vi = 0
        for a in range(nUi):
            vi += indptr[Ui[a] + 1] - indptr[Ui[a]]
        vj = 0
        for b in range(nUj):
            vj += indptr[Uj[b] + 1] - indptr[Uj[b]]
        scan_i = vi <= vj
    if scan_i:
        S = Ui
        nS = nUi
        O = Uj
        nO = nUj
    else:
        S = Uj
        nS = nUj
        O = Ui
        nO = nUi
    for b in range(nO):
        w = O[b]
        sO[w] = e32
        pO[w] = np.int32(b)
    bi, oi, ptr = _take_i(bi, oi, nS + 1)
    bi, oi, rcnt = _take_i(bi, oi, nO)
    # cross-edge buffer, doubled on demand: memory O(number of cross edges)
    bi, oi, adj = _take_i(bi, oi, 2 * (nS + nO) + 64)
    for b in range(nO):
        rcnt[b] = 0
    # cross edges U_i -- U_j, as a CSR adjacency from the scanned side
    ptr[0] = 0
    m = 0
    nxS = 0  # scanned-side vertices with a cross edge
    varpi = 0
    for a in range(nS):
        k0 = indptr[S[a]]
        k1 = indptr[S[a] + 1]
        # room for every cross edge of this vertex, reserved here and not per hit: an
        # array that may be reassigned inside the inner loop becomes a loop-carried
        # variable, and Numba then pays atomic reference counts on every hit
        need = m + min(k1 - k0, nO)
        if need > adj.shape[0]:
            bi, oi, adj = _grow_i(bi, oi, adj, m, need)
        if tlw:
            for t in range(k0, k1):
                w = indices[t]
                mS[w] = e32
                if sO[w] == e32:
                    b = pO[w]
                    adj[m] = b
                    m += 1
                    rcnt[b] += 1
        elif k1 - k0 <= _HUB * nO:
            for t in range(k0, k1):
                w = indices[t]
                if sO[w] == e32:
                    b = pO[w]
                    adj[m] = b
                    m += 1
                    rcnt[b] += 1
        else:
            for b in range(nO):
                if _contains(indices, k0, k1, O[b]):
                    adj[m] = b
                    m += 1
                    rcnt[b] += 1
        c = m - ptr[a]
        ptr[a + 1] = m
        if c > 0:
            nxS += 1
            if c > varpi:
                varpi = c
    nxO = 0
    for b in range(nO):
        c = rcnt[b]
        if c > 0:
            nxO += 1
            if c > varpi:
                varpi = c
    out[3] = nxS + nxO
    out[4] = varpi
    xi_i = nxS if scan_i else nxO  # |xi| of the oriented i
    out[7] = (nxS + nxO - xi_i) if flip else xi_i
    if matching and m > 0:
        res, bi, oi = _hk(nS, nO, ptr, adj, min(nxS, nxO), bi, oi)
        out[6] = res
    if legacy:
        # max_k in N(i)\{j} |N(k) & (N(j)\{i})| and symmetrically (symmetric in i, j)
        a0 = indptr[i]
        a1 = indptr[i + 1]
        b0 = indptr[j]
        b1 = indptr[j + 1]
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
    if tlw and nUi > 0:  # (nUi = 0: identical measures, numerator 0)
        # here S = U_i, O = U_j, and mS stamps the vertices adjacent to U_i
        sy = 0
        for b in range(nUj):
            if rcnt[b] > 0:
                sy += 1
            else:
                y = Uj[b]
                d = 3
                for t in range(indptr[y], indptr[y + 1]):
                    if mS[indices[t]] == e32:
                        d = 2
                        break
                sy += d
        if di > dj:
            st = 0
            for t in range(nT):
                st += 1 if mS[Tt[t]] == e32 else 2
            sP = nUi * (dj + 1)
            sN = (di + 1) * sy + (di - dj) * (3 + st)
        else:
            sx = nxS
            if nxS < nUi:
                for b in range(nUj):
                    y = Uj[b]
                    for t in range(indptr[y], indptr[y + 1]):
                        mO[indices[t]] = e32
                for a in range(nUi):
                    if ptr[a + 1] == ptr[a]:
                        x = Ui[a]
                        d = 3
                        for t in range(indptr[x], indptr[x + 1]):
                            if mO[indices[t]] == e32:
                                d = 2
                                break
                        sx += d
            sP = (dj + 1) * sx
            sN = (di + 1) * sy
        out[8] = sP if sP > sN else sN
    return bi


@njit(cache=True)
def _stats_chunk(indptr, indices, edges, c, nch, block, legacy, matching, tlw, out):
    """Blocks c, c + nch, c + 2 nch, ... of ``block`` consecutive edges on one
    thread, with its own stamps and arena (see :func:`_chunk`)."""
    n = indptr.shape[0] - 1
    sO = np.full(n, -1, np.int32)
    pO = np.empty(n, np.int32)
    mS = np.full(n, -1, np.int32)
    mO = np.full(n, -1, np.int32)
    bi = np.empty(1 << 12, np.int64)
    K = edges.shape[0]
    nb = (K + block - 1) // block
    for bk in range(c, nb, nch):
        lo = bk * block
        hi = min(K, lo + block)
        for k in range(lo, hi):
            bi = _stats_edge(indptr, indices, edges[k, 0], edges[k, 1], legacy, matching, tlw,
                             k, sO, pO, mS, mO, bi, out[k])


@njit(cache=True, parallel=True)
def _stats_run(indptr, indices, edges, legacy, matching, tlw, nchunks, block):
    K = edges.shape[0]
    out = np.zeros((K, N_STATS), np.int64)
    if K == 0:
        return out
    nb = (K + block - 1) // block
    nchunks = max(1, min(nchunks, nb))
    for c in prange(nchunks):
        _stats_chunk(indptr, indices, edges, c, nchunks, block, legacy, matching, tlw, out)
    return out


def edge_stats_all(indptr, indices, edges, legacy, matching=True, tlw=False, nchunks=None):
    """2-hop statistics of every edge of ``edges`` (a (K, 2) array) as a
    (K, ``N_STATS``) int64 array: the columns of :func:`edge_stats_one`, plus
    column 8 = the numerator of the lazy TLW dual certificate if ``tlw`` (see
    :func:`tlw_lazy_from_stats`), else 0. ``matching=False`` skips the maximum
    matching (column 6 is then 0). Parallel over at most ``nchunks`` chunks
    (default and maximum: one per thread)."""
    E = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    if E.shape[0] >= np.iinfo(np.int32).max:
        raise ValueError("at most 2^31 - 2 edges per call (the vertex stamps are int32)")
    return _stats_run(indptr, indices, E, bool(legacy), bool(matching), bool(tlw), _nchunks(nchunks),
                      _STATS_BLOCK)


def tlw_lazy_from_stats(raw, di, dj):
    """TLW dual upper bound on the lazy curvature from column 8 of
    :func:`edge_stats_all` (``tlw=True``) and the degrees: the same arithmetic
    as the exact solver, 1 - numerator / ((rho_i + 1)(rho_j + 1))."""
    return 1.0 - np.asarray(raw)[:, 8] / ((np.asarray(di, float) + 1.0) * (np.asarray(dj, float) + 1.0))


def tlw_lazy_ids(indptr, indices, edges, ids, nchunks):
    """TLW dual upper bound on the lazy curvature for edges[ids] (no transport
    solve and no local distance structure: the 2-hop kernel with ``tlw=True``;
    equal bit for bit to the certificate that the exact solver returns)."""
    E = np.ascontiguousarray(np.asarray(edges)[_ids(ids)], dtype=np.int64).reshape(-1, 2)
    raw = edge_stats_all(indptr, indices, E, False, False, True, nchunks)
    deg = indptr[1:] - indptr[:-1]
    return tlw_lazy_from_stats(raw, deg[E[:, 0]], deg[E[:, 1]])

