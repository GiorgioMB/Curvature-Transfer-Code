"""Numba kernels.

Two families of per-edge kernels, each with a parallel driver over all edges:

* ``edge_stats``: the 2-hop statistics that parameterize BF curvature and all
  combinatorial bounds (triangles, unique neighborhoods, 4-cycle counts, and
  the maximum matching of the cross-edge graph, via Hopcroft--Karp).
* ``exact``: exact Wasserstein-1 distances between one-step measures, for the
  lazy measures of the paper (OR curvature), the non-lazy neighbor laws, and
  the uniform idleness p = 1/(deg_max + 1), which yields the Lin--Lu--Yau
  curvature exactly by the idleness linearity of Bourne et al. (2018).

Exact transport
---------------
For an edge (i, j) let S = N(i) u N(j) (this already contains i and j). All
measures are supported on S, and every pairwise distance on S is in {0,1,2,3}:
d(x, y) <= 2 within N(i) or within N(j), and <= 3 across, through the edge.
Distances are therefore computed from adjacency alone: 1 for adjacent pairs,
2 for pairs with a common neighbor (found by grouping the 2-paths through
each middle vertex), and 3 otherwise. No BFS is needed.

By Kantorovich--Rubinstein duality, W1(mu, nu) = max sum_v (mu - nu)(v) phi(v)
over 1-Lipschitz phi on supp(mu - nu). The constraint matrix of this LP is a
network matrix and the distances are integers, so an integral optimal phi
exists (cf. Bourne et al. 2018, Lemma 3.3); since all distances are at most
three, phi can be shifted into {0, 1, 2, 3}. Writing phi(v) = sum_k [phi(v) >= k]
for k = 1, 2, 3 turns the dual into a maximum-weight *closure* problem with the
implications [phi(u) >= k] => [phi(u) >= k-1] and [phi(u) >= k] =>
[phi(v) >= k - d(u, v)], which is solved by one s-t min cut (Picard 1976). With
masses scaled to integers this is exact integer arithmetic. Before solving,
vertices of the same sign that are twins (identical distances to everything
else) are merged; this is exact because an optimal potential can be taken
constant on such a class, and it collapses, for example, all the leaves of a hub.
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


# ------------------------------------------------------- local distances


@njit(cache=True)
def local_support(indptr, indices, i, j):
    """Sorted S = N(i) u N(j) (contains i and j since they are adjacent)."""
    a0 = indptr[i]
    a1 = indptr[i + 1]
    b0 = indptr[j]
    b1 = indptr[j + 1]
    S = np.empty((a1 - a0) + (b1 - b0), np.int64)
    s = 0
    p = a0
    q = b0
    while p < a1 or q < b1:
        if q >= b1 or (p < a1 and indices[p] < indices[q]):
            S[s] = indices[p]
            p += 1
        elif p >= a1 or indices[q] < indices[p]:
            S[s] = indices[q]
            q += 1
        else:
            S[s] = indices[p]
            p += 1
            q += 1
        s += 1
    return S[:s]


@njit(cache=True)
def new_scratch(n):
    """Per-thread scratch: local index of each global vertex and grouping
    arrays for 2-paths, all stamped so they never need clearing."""
    pos = np.full(n, -1, np.int64)
    pos_stamp = np.full(n, -1, np.int64)
    ghead = np.full(n, -1, np.int64)
    g_stamp = np.full(n, -1, np.int64)
    return pos, pos_stamp, ghead, g_stamp


@njit(cache=True)
def local_distances(indptr, indices, S, scratch, epoch):
    """Graph distances (capped at 3) between all pairs of the sorted set S,
    from adjacency only: 1 for edges, 2 for pairs with a common neighbor
    (grouping the 2-paths by middle vertex), 3 otherwise."""
    pos, pos_stamp, ghead, g_stamp = scratch
    s = S.shape[0]
    for a in range(s):
        pos[S[a]] = a
        pos_stamp[S[a]] = epoch
    D = np.full((s, s), 3, np.int8)
    tot = 0
    for a in range(s):
        D[a, a] = 0
        tot += indptr[S[a] + 1] - indptr[S[a]]
    gnext = np.empty(tot, np.int64)
    gwho = np.empty(tot, np.int64)
    touched = np.empty(tot, np.int64)
    nt = 0
    m = 0
    for a in range(s):
        u = S[a]
        for t in range(indptr[u], indptr[u + 1]):
            w = indices[t]
            if pos_stamp[w] == epoch:
                D[a, pos[w]] = 1
            if g_stamp[w] != epoch:
                g_stamp[w] = epoch
                ghead[w] = -1
                touched[nt] = w
                nt += 1
            gwho[m] = a
            gnext[m] = ghead[w]
            ghead[w] = m
            m += 1
    for k in range(nt):
        w = touched[k]
        x = ghead[w]
        while x != -1:
            a = gwho[x]
            y = gnext[x]
            while y != -1:
                b = gwho[y]
                if D[a, b] > 2:
                    D[a, b] = 2
                    D[b, a] = 2
                y = gnext[y]
            x = gnext[x]
    return D


# -------------------------------------------------------- max-flow (Dinic)


@njit(cache=True)
def _dfs_block(u, t, f, nxt, to, cap, level, it):
    """Blocking-flow DFS: push up to f units from u to t along level arcs."""
    if u == t:
        return f
    pushed = 0
    e = it[u]
    while e != -1:
        v = to[e]
        if cap[e] > 0 and level[v] == level[u] + 1:
            r = f - pushed
            if cap[e] < r:
                r = cap[e]
            d = _dfs_block(v, t, r, nxt, to, cap, level, it)
            if d > 0:
                cap[e] -= d
                cap[e ^ 1] += d
                pushed += d
                if pushed == f:
                    it[u] = e
                    return pushed
        e = nxt[e]
    it[u] = -1
    if pushed == 0:
        level[u] = -1
    return pushed


@njit(cache=True)
def _dinic(V, head, nxt, to, cap, s, t):
    flow = 0
    level = np.empty(V, np.int64)
    it = np.empty(V, np.int64)
    q = np.empty(V, np.int64)
    while True:
        for v in range(V):
            level[v] = -1
        level[s] = 0
        qh = 0
        qt = 1
        q[0] = s
        while qh < qt:
            u = q[qh]
            qh += 1
            e = head[u]
            while e != -1:
                if cap[e] > 0 and level[to[e]] < 0:
                    level[to[e]] = level[u] + 1
                    q[qt] = to[e]
                    qt += 1
                e = nxt[e]
        if level[t] < 0:
            return flow
        for v in range(V):
            it[v] = head[v]
        while True:
            f = _dfs_block(s, t, INF, nxt, to, cap, level, it)
            if f == 0:
                break
            flow += f


# ------------------------------------------------ W1 by maximum closure


@njit(cache=True)
def _merge_twins(D, w, diag):
    """Merge same-sign vertices whose rows coincide once the diagonal is set
    to ``diag`` (such vertices are twins at mutual distance ``diag``)."""
    s = w.shape[0]
    if s <= 2:
        return D, w
    h = np.empty(s, np.int64)
    for a in range(s):
        x = np.int64(1469598103934665603)
        for b in range(s):
            v = np.int64(diag) if a == b else np.int64(D[a, b])
            x = (x ^ (v + 1)) * np.int64(1099511628211)
        sg = np.int64(1) if w[a] > 0 else np.int64(2)
        h[a] = x * np.int64(31) + sg
    order = np.argsort(h)
    rep = np.full(s, -1, np.int64)
    nrep = 0
    newid = np.empty(s, np.int64)
    leaders = np.empty(s, np.int64)
    k = 0
    while k < s:
        k1 = k + 1
        while k1 < s and h[order[k1]] == h[order[k]]:
            k1 += 1
        # within the hash bucket, verify exact row equality against leaders
        first = nrep
        for x in range(k, k1):
            a = order[x]
            placed = False
            for L in range(first, nrep):
                b = leaders[L]
                if (w[a] > 0) != (w[b] > 0):
                    continue
                same = True
                for c in range(s):
                    va = diag if c == a else D[a, c]
                    vb = diag if c == b else D[b, c]
                    if va != vb:
                        same = False
                        break
                if same:
                    newid[a] = L
                    placed = True
                    break
            if not placed:
                leaders[nrep] = a
                newid[a] = nrep
                nrep += 1
        k = k1
    if nrep == s:
        return D, w
    w2 = np.zeros(nrep, np.int64)
    for a in range(s):
        w2[newid[a]] += w[a]
    D2 = np.empty((nrep, nrep), np.int8)
    for L in range(nrep):
        for M in range(nrep):
            D2[L, M] = D[leaders[L], leaders[M]]
    return D2, w2


@njit(cache=True)
def w1_closure(D, w):
    """W1 * scale for integer net masses w (sum zero) on vertices with metric
    D (entries in {0,1,2,3}); vertices of weight zero act as relays. Exact.

    Implication arcs are added for adjacent pairs, and for pairs at distance
    two only when they have no common neighbor among the variables (otherwise
    the constraint is implied by transitivity of the closure)."""
    s = w.shape[0]
    if s == 0:
        return 0
    D, w = _merge_twins(D, w, 2)
    D, w = _merge_twins(D, w, 1)
    s = w.shape[0]
    pos_total = 0
    for a in range(s):
        if w[a] > 0:
            pos_total += 3 * w[a]
    if pos_total == 0:
        return 0
    # adjacency bitsets of the (merged) problem
    nw = (s + 63) >> 6
    bits = np.zeros((s, nw), np.uint64)
    for a in range(s):
        for b in range(s):
            if D[a, b] == 1:
                bits[a, b >> 6] |= np.uint64(1) << np.uint64(b & 63)
    A = 0
    for a in range(s):
        A += 2
        if w[a] != 0:
            A += 3
        for b in range(s):
            if a == b:
                continue
            d = D[a, b]
            if d == 1:
                A += 2
            elif d == 2:
                implied = False
                for q in range(nw):
                    if bits[a, q] & bits[b, q]:
                        implied = True
                        break
                if not implied:
                    A += 1
    V = 3 * s + 2
    src = 3 * s
    snk = 3 * s + 1
    head = np.full(V, -1, np.int64)
    nxt = np.empty(2 * A, np.int64)
    to = np.empty(2 * A, np.int64)
    cap = np.empty(2 * A, np.int64)
    m = 0
    for a in range(s):
        if w[a] != 0:
            for k in range(3):
                node = 3 * a + k
                if w[a] > 0:
                    u = src
                    v = node
                    c = w[a]
                else:
                    u = node
                    v = snk
                    c = -w[a]
                to[m] = v; cap[m] = c; nxt[m] = head[u]; head[u] = m; m += 1
                to[m] = u; cap[m] = 0; nxt[m] = head[v]; head[v] = m; m += 1
        for k in range(1, 3):  # level k+1 => level k
            u = 3 * a + k
            v = 3 * a + k - 1
            to[m] = v; cap[m] = INF; nxt[m] = head[u]; head[u] = m; m += 1
            to[m] = u; cap[m] = 0; nxt[m] = head[v]; head[v] = m; m += 1
        for b in range(s):
            if a == b:
                continue
            d = D[a, b]
            if d == 1:
                # phi(a) >= 2 => phi(b) >= 1 ; phi(a) >= 3 => phi(b) >= 2
                for k in range(1, 3):
                    u = 3 * a + k
                    v = 3 * b + k - 1
                    to[m] = v; cap[m] = INF; nxt[m] = head[u]; head[u] = m; m += 1
                    to[m] = u; cap[m] = 0; nxt[m] = head[v]; head[v] = m; m += 1
            elif d == 2:
                implied = False
                for q in range(nw):
                    if bits[a, q] & bits[b, q]:
                        implied = True
                        break
                if not implied:
                    u = 3 * a + 2
                    v = 3 * b
                    to[m] = v; cap[m] = INF; nxt[m] = head[u]; head[u] = m; m += 1
                    to[m] = u; cap[m] = 0; nxt[m] = head[v]; head[v] = m; m += 1
    flow = _dinic(V, head, nxt, to, cap, src, snk)
    return pos_total - flow


@njit(cache=True)
def _tlw_dual(D, w, idx):
    """sum_P w_x min_{y in N} d(x, y) and sum_N |w_y| min_{x in P} d(x, y)."""
    a = 0
    b = 0
    n = idx.shape[0]
    for p in range(n):
        x = idx[p]
        if w[p] > 0:
            mn = 99
            for q in range(n):
                if w[q] < 0 and D[x, idx[q]] < mn:
                    mn = D[x, idx[q]]
            a += w[p] * mn
        elif w[p] < 0:
            mn = 99
            for q in range(n):
                if w[q] > 0 and D[idx[q], x] < mn:
                    mn = D[idx[q], x]
            b += (-w[p]) * mn
    return max(a, b)


N_EXACT = 5  # c_OR, c_OR0, c_LLY, TLW dual (lazy), TLW dual (non-lazy)


@njit(cache=True)
def _net(S, indptr, indices, i, j, mi_self, mi_nb, mj_self, mj_nb):
    """Integer net masses on S for measures (self mass, per-neighbor mass)."""
    s = S.shape[0]
    w = np.zeros(s, np.int64)
    for a in range(s):
        x = S[a]
        if x == i:
            w[a] += mi_self
        elif _contains(indices, indptr[i], indptr[i + 1], x):
            w[a] += mi_nb
        if x == j:
            w[a] -= mj_self
        elif _contains(indices, indptr[j], indptr[j + 1], x):
            w[a] -= mj_nb
    return w


@njit(cache=True)
def _w1_measure(D, w):
    nz = 0
    for a in range(w.shape[0]):
        if w[a] != 0:
            nz += 1
    idx = np.empty(nz, np.int64)
    k = 0
    for a in range(w.shape[0]):
        if w[a] != 0:
            idx[k] = a
            k += 1
    wsub = np.empty(nz, np.int64)
    for p in range(nz):
        wsub[p] = w[idx[p]]
    return w1_closure(D.copy(), w.copy()), _tlw_dual(D, wsub, idx)


@njit(cache=True)
def exact_one(indptr, indices, i, j, want_or, want_or0, want_lly, scratch, epoch):
    """Exact curvatures of edge (i, j). Returns float64[5]
    (c_OR, c_OR0, c_LLY, TLW dual upper lazy, TLW dual upper non-lazy);
    entries not requested are NaN."""
    out = np.full(N_EXACT, np.nan)
    S = local_support(indptr, indices, i, j)
    D = local_distances(indptr, indices, S, scratch, epoch)
    di = np.int64(indptr[i + 1] - indptr[i])
    dj = np.int64(indptr[j + 1] - indptr[j])
    if want_or:
        # m_u uniform on the closed neighborhood: scale (di+1)(dj+1)
        L = (di + 1) * (dj + 1)
        w = _net(S, indptr, indices, i, j, dj + 1, dj + 1, di + 1, di + 1)
        W, T = _w1_measure(D, w)
        out[0] = 1.0 - W / L
        out[3] = 1.0 - T / L
    if want_or0:
        L = di * dj
        w = _net(S, indptr, indices, i, j, 0, dj, 0, di)
        W, T = _w1_measure(D, w)
        out[1] = 1.0 - W / L
        out[4] = 1.0 - T / L
    if want_lly:
        # uniform idleness p = 1/(dmax+1): kappa_p = (1-p) c_LLY
        dmax = di if di > dj else dj
        L = (dmax + 1) * di * dj
        w = _net(S, indptr, indices, i, j, di * dj, dmax * dj, di * dj, dmax * di)
        W, T = _w1_measure(D, w)
        kp = 1.0 - W / L
        out[2] = kp * (dmax + 1) / dmax
    return out


@njit(cache=True, parallel=True)
def exact_ids(indptr, indices, edges, ids, want_or, want_or0, want_lly, nchunks):
    """Exact curvatures for edges[ids], parallel over chunks; each chunk owns
    an O(n) scratch space."""
    n = indptr.shape[0] - 1
    K = ids.shape[0]
    out = np.full((K, N_EXACT), np.nan)
    if K == 0:
        return out
    nchunks = max(1, min(nchunks, K))
    step = (K + nchunks - 1) // nchunks
    for c in prange(nchunks):
        lo = c * step
        hi = min(K, lo + step)
        if lo >= hi:
            continue
        scratch = new_scratch(n)
        for k in range(lo, hi):
            e = ids[k]
            out[k] = exact_one(indptr, indices, edges[e, 0], edges[e, 1],
                               want_or, want_or0, want_lly, scratch, k)
    return out


@njit(cache=True)
def tlw_lazy_one(indptr, indices, i, j, scratch, epoch):
    """TLW dual upper bound on the lazy curvature (distances only, no flow)."""
    S = local_support(indptr, indices, i, j)
    D = local_distances(indptr, indices, S, scratch, epoch)
    di = np.int64(indptr[i + 1] - indptr[i])
    dj = np.int64(indptr[j + 1] - indptr[j])
    w = _net(S, indptr, indices, i, j, dj + 1, dj + 1, di + 1, di + 1)
    nz = 0
    for a in range(w.shape[0]):
        if w[a] != 0:
            nz += 1
    idx = np.empty(nz, np.int64)
    wsub = np.empty(nz, np.int64)
    k = 0
    for a in range(w.shape[0]):
        if w[a] != 0:
            idx[k] = a
            wsub[k] = w[a]
            k += 1
    return 1.0 - _tlw_dual(D, wsub, idx) / ((di + 1) * (dj + 1))


@njit(cache=True, parallel=True)
def tlw_lazy_ids(indptr, indices, edges, ids, nchunks):
    n = indptr.shape[0] - 1
    K = ids.shape[0]
    out = np.full(K, np.nan)
    if K == 0:
        return out
    nchunks = max(1, min(nchunks, K))
    step = (K + nchunks - 1) // nchunks
    for c in prange(nchunks):
        lo = c * step
        hi = min(K, lo + step)
        if lo >= hi:
            continue
        scratch = new_scratch(n)
        for k in range(lo, hi):
            e = ids[k]
            out[k] = tlw_lazy_one(indptr, indices, edges[e, 0], edges[e, 1], scratch, k)
    return out
