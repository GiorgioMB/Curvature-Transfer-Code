"""Reference exact solver (the previous implementation), kept as a test oracle.

For an edge (i, j) it builds the full local distance matrix on S = N(i) u N(j)
(entries in {0, 1, 2, 3}, from adjacency and 2-paths), merges twin vertices,
and solves the Kantorovich dual as a maximum-weight closure on three levels
per vertex (one s-t min cut, recursive Dinic). It is exact and independent of
the production solver in :mod:`curvature_transfer._kernels` (different
formulation, different distance computation, different max-flow code), and it
is verified against linear programming in the tests; the tests check that the
two solvers return identical values. It is slower, and it is not used by the
package.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

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
def exact_ids_reference(indptr, indices, edges, ids, want_or, want_or0, want_lly, nchunks):
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
