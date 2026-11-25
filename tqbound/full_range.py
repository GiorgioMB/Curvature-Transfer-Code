"""Full range of Theorem 5.2: the minimum and the maximum of kappa_mu(i, j) over all graphs consistent with
the building blocks of (i, j).

  lower end   the curvature of the graph of listed edges (every unknown pair of U_i x U_j at distance 3),
              computed exactly by orcut on the adversarial local graph;
  upper end   the integer program over admissible sets F of pairs {u, y}, y in U_j, u in (U_j u Z) - {y},
              with every vertex v in at most dbar_v pairs (scipy.optimize.milp, HiGHS);
  relaxation  the same without the free degrees, by orcut on the local graph with one extra vertex joined
              to every open vertex of U_i and U_j (an upper bound on the upper end).

Measures: 'lazy', a number p (uniform idleness), 'wj' (uniform idleness w_j of each edge, Corollary 5.4),
'degree' (lazy idleness, neighbours weighted by 1/d_v).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import orcut
from orcut.measures import Walk


# ----------------------------------------------------------------------------------------------- measures
def masses(adj, deg, measure, i, j):
    """mu_i and mu_j as dicts on N[i] and N[j] (floats)."""
    out = []
    b = max(deg[i], deg[j])
    for u in (i, j):
        d = deg[u]
        if measure == 'lazy' or measure == 'degree':
            al = 1.0 / (d + 1)
        elif measure == 'wj':
            al = 1.0 / (b + 1)
        else:
            al = float(measure)
        if measure == 'degree':
            Z = sum(1.0 / deg[v] for v in adj[u])
            m = {v: (1 - al) * (1.0 / deg[v]) / Z for v in adj[u]}
        else:
            m = {v: (1 - al) / d for v in adj[u]}
        m[u] = m.get(u, 0.0) + al
        out.append(m)
    return out


# ------------------------------------------------------------------------------------------ building blocks
def edge_blocks(adj, deg, u, v):
    i, j = (u, v) if deg[u] <= deg[v] else (v, u)
    Ni, Nj = adj[i], adj[j]
    C = Ni & Nj; Ui = Ni - Nj - {j}; Uj = Nj - Ni - {i}
    inside = Ni | Nj | {i, j}
    H = {(x, y) for x in Ui for y in adj[x] if y in Uj}
    known2 = set()
    for x in Ui:
        for w in adj[x]:
            if w in C or w in Ui:
                for y in adj[w]:
                    if y in Uj:
                        known2.add((x, y))
    Z = {x: [z for z in adj[x] if z not in inside] for x in Ui}
    listed = Ui | C
    sy = {y: deg[y] - 1 - sum(1 for w in adj[y] if w in listed) for y in Uj}
    sz = {}
    for x in Ui:
        for z in Z[x]:
            if z not in sz:
                sz[z] = deg[z] - sum(1 for w in adj[z] if w in listed)
    return dict(i=i, j=j, C=C, Ui=Ui, Uj=Uj, H=H, known2=known2, Z=Z, sy=sy, sz=sz, inside=inside)


def _dist_known(adj, B, p, q):
    """Distance of a supply-demand pair outside U_i x U_j (Lemma 4.1): 1 if listed as an edge, else 2."""
    return 1 if q in adj[p] else 2


# ------------------------------------------------------------------------------------------- upper end (MILP)
def upper_end(adj, deg, measure, u, v, time_limit=60.0):
    """kappa-bar* of Theorem 5.2 for the edge (u, v); returns (value, solved)."""
    B = edge_blocks(adj, deg, u, v)
    i, j, Ui, Uj = B['i'], B['j'], B['Ui'], B['Uj']
    mi, mj = masses(adj, deg, measure, i, j)
    keys = set(mi) | set(mj)
    r = {w: mi.get(w, 0.0) - mj.get(w, 0.0) for w in keys}
    P = [w for w in keys if r[w] > 1e-15]; Q = [w for w in keys if r[w] < -1e-15]
    if not P:
        return 1.0, True
    sup = np.array([r[w] for w in P]); dem = np.array([-r[w] for w in Q])
    # arcs: (p, q, cost) always allowed; gated cost-2 arcs for unknown pairs of U_i x U_j
    arcs_p, arcs_q, arcs_c, gated = [], [], [], []
    for a_, p in enumerate(P):
        for b_, q in enumerate(Q):
            if p in Ui and q in Uj:
                if (p, q) in B['H']:
                    c = 1
                elif (p, q) in B['known2']:
                    c = 2
                else:
                    c = 3
                    gated.append((len(arcs_p) + 0, a_, b_, p, q))   # its cost-2 twin is added below
            else:
                c = _dist_known(adj, B, p, q)
            arcs_p.append(a_); arcs_q.append(b_); arcs_c.append(c)
    # candidate pairs {w, y} of F: w in Z or U_j adjacent to a vertex x with an unknown pair to y
    open_y = {y for y in Uj if B['sy'][y] > 0}
    hub_of = {}                                            # (x, y) -> list of candidate pair keys
    cand = {}
    for (_, a_, b_, x, y) in gated:
        if y not in open_y:
            continue
        ks = []
        for z in B['Z'][x]:
            if B['sz'][z] > 0:
                ks.append(('z', z, y))
        for (xx, yp) in B['H']:
            if xx == x and yp != y and yp in open_y:
                ks.append(('f', min(yp, y), max(yp, y)))
        if ks:
            hub_of[(x, y)] = ks
            for k in ks:
                cand.setdefault(k, len(cand))
    nA = len(arcs_p)
    twins = [(g, a_, b_, x, y) for (g, a_, b_, x, y) in gated if (x, y) in hub_of]
    nT = len(twins)
    if nT == 0:                                            # nothing can be opened: the upper end is the lower end
        val = _transport(sup, dem, arcs_p, arcs_q, arcs_c, len(P), len(Q))
        return (1.0 - val, True) if val is not None else (np.nan, False)
    # dominance: opening a pair only adds options, so a pair whose two endpoints have a free degree at least
    # their number of candidate pairs is open in some optimum; only pairs at a binding vertex need a binary
    load = {}
    for k in cand:
        ends = (('y', k[2]), ('z', k[1])) if k[0] == 'z' else (('y', k[1]), ('y', k[2]))
        for e_ in ends:
            load[e_] = load.get(e_, 0) + 1
    def free(e_):
        return B['sy'][e_[1]] if e_[0] == 'y' else B['sz'][e_[1]]
    def ends_of(k):
        return (('y', k[2]), ('z', k[1])) if k[0] == 'z' else (('y', k[1]), ('y', k[2]))
    binding = {e_ for e_, c_ in load.items() if free(e_) < c_}
    var_keys = [k for k in cand if any(e_ in binding for e_ in ends_of(k))]
    vk = {k: t for t, k in enumerate(var_keys)}
    nG = len(var_keys)
    nV = nA + nT + nG
    c = np.concatenate([np.array(arcs_c, float), np.full(nT, 2.0), np.zeros(nG)])
    nP, nQ = len(P), len(Q)
    tw_p = np.array([t[1] for t in twins], np.int64); tw_q = np.array([t[2] for t in twins], np.int64)
    ap = np.array(arcs_p, np.int64); aq = np.array(arcs_q, np.int64)
    rows = list(np.concatenate([ap, nP + aq, tw_p, nP + tw_q]))
    cols = list(np.concatenate([np.arange(nA), np.arange(nA), nA + np.arange(nT), nA + np.arange(nT)]))
    vals = [1.0] * len(rows)
    lo = list(sup) + list(dem); hi = list(sup) + list(dem)
    R = nP + nQ
    for t_, (g, aa, bb, x, y) in enumerate(twins):       # a twin carries mass only through an open pair
        ks = hub_of[(x, y)]
        if any(k not in vk for k in ks):                 # one of its pairs is open in some optimum
            continue
        cap = float(min(sup[aa], dem[bb]))
        rows.append(R); cols.append(nA + t_); vals.append(1.0)
        for k in ks:
            rows.append(R); cols.append(nA + nT + vk[k]); vals.append(-cap)
        lo.append(-np.inf); hi.append(0.0); R += 1
    for e_ in binding:                                    # free degrees of the binding vertices
        qs = [vk[k] for k in var_keys if e_ in ends_of(k)]
        for q in qs:
            rows.append(R); cols.append(nA + nT + q); vals.append(1.0)
        lo.append(-np.inf); hi.append(float(free(e_))); R += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(R, nV))
    integ = np.concatenate([np.zeros(nA + nT), np.ones(nG)])
    ub = np.concatenate([np.full(nA + nT, np.inf), np.ones(nG)])
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nV), ub), options={'time_limit': time_limit, 'mip_rel_gap': 1e-9})
    if res.status != 0:
        return np.nan, False
    return 1.0 - res.fun, True


def _transport(sup, dem, ap, aq, ac, nP, nQ):
    from scipy.optimize import linprog
    nA = len(ap)
    ap = np.asarray(ap, np.int64); aq = np.asarray(aq, np.int64)
    A = sp.csr_matrix((np.ones(2 * nA), (np.concatenate([ap, nP + aq]), np.concatenate([np.arange(nA)] * 2))),
                      shape=(nP + nQ, nA))
    res = linprog(np.array(ac, float), A_eq=A, b_eq=np.concatenate([sup, dem]), bounds=(0, None), method='highs')
    return res.fun if res.status == 0 else None


# ------------------------------------------------------------------ lower end and relaxation (orcut, batched)
def _local_edges(adj, deg, B, friendly):
    i, j, Ui, Uj, C, inside = B['i'], B['j'], B['Ui'], B['Uj'], B['C'], B['inside']
    E = set()
    for w in [i, j] + list(Ui) + list(C):
        for x in adj[w]:
            if x in inside:
                E.add((min(w, x), max(w, x)))
    extra = []
    if friendly:
        sy = B['sy']
        oy = [y for y in Uj if sy[y] > 0]
        ox = [x for x in Ui if any(z not in inside for z in adj[x]) or any(w in Uj and sy[w] > 0 for w in adj[x])]
        if ox and oy:
            extra = ox + oy
    return E, extra


def local_batch(adj, deg, edges, friendly):
    """Disjoint union of the adversarial (or, with ``friendly``, slot-free friendly) local graphs of ``edges``:
    returns (orcut graph, ids of the designated edges, original vertex of every union vertex)."""
    allE = []; des = []; orig = []; off = 0
    for (u, v) in edges:
        B = edge_blocks(adj, deg, u, v)
        E, extra = _local_edges(adj, deg, B, friendly)
        verts = sorted({w for ab in E for w in ab})
        lab = {w: off + t for t, w in enumerate(verts)}
        allE.extend((lab[a], lab[b]) for a, b in E)
        nloc = len(verts)
        if extra:
            allE.extend((lab[w], off + nloc) for w in extra)
            nloc += 1
        orig.extend(verts + ([None] if extra else []))
        des.append((lab[B['i']], lab[B['j']]))
        off += nloc
    g = orcut.Graph.from_edges(np.array(allE, np.int64), num_nodes=off)
    ge = np.asarray(g.edges)
    pos = {(int(a), int(b)): t for t, (a, b) in enumerate(ge)}
    ids = np.array([pos[(min(a, b), max(a, b))] for a, b in des], np.int64)
    return g, ids, orig


def batch_measure(g, orig, deg, measure, key=None):
    """The orcut measure of ``measure`` on a batch (``key``: the idleness w_j of the group for 'wj')."""
    if measure == 'lazy':
        return 'lazy'
    if measure == 'wj':
        return key
    if measure == 'degree':
        ind = np.asarray(g.indices)
        dg = np.array([deg[w] if w is not None else 1 for w in orig], float)
        return Walk(weights=1.0 / dg[ind], idleness='lazy')
    return float(measure)


def chunks_by_size(adj, deg, edges, cap=2_000_000):
    """Split ``edges`` into consecutive chunks whose local graphs have at most about ``cap`` list entries
    (estimated by d_j plus the degrees of the neighbours of i), so that batches fit in memory."""
    out, cur, size = [], [], 0
    for e, (u, v) in enumerate(edges):
        i, j = (u, v) if deg[u] <= deg[v] else (v, u)
        est = int(deg[j]) + sum(int(deg[w]) for w in adj[i])
        if cur and size + est > cap:
            out.append(cur); cur, size = [], 0
        cur.append(e); size += est
    if cur:
        out.append(cur)
    return out


def lower_and_relaxed(adj, deg, edges, measure, n_jobs=None, exact=False):
    """(kappa_lower*, kappa_relaxed) for every edge, by orcut on the adversarial local graphs and on the
    friendly local graphs without free degrees, batched as disjoint unions (one batch per idleness value).
    With ``exact`` (lazy or rational idleness), also the exact W1 fractions of both: (lo, hi, lo_frac, hi_frac),
    each fraction an (m, 2) object array of (numerator, denominator)."""
    groups = {}
    for e, (u, v) in enumerate(edges):
        key = 1.0 / (max(deg[u], deg[v]) + 1) if measure == 'wj' else None
        groups.setdefault(key, []).append(e)
    m = len(edges)
    lo = np.full(m, np.nan); hi = np.full(m, np.nan)
    fr = [np.empty((m, 2), object), np.empty((m, 2), object)]
    pieces = []
    for key, idx in groups.items():
        sub = [edges[e] for e in idx]
        for ch in chunks_by_size(adj, deg, sub):
            pieces.append((key, np.array([idx[c] for c in ch])))
    for key, idx in pieces:
        for t_, (friendly, outv) in enumerate(((False, lo), (True, hi))):
            g, ids, orig = local_batch(adj, deg, [edges[e] for e in idx], friendly)
            ms = batch_measure(g, orig, deg, measure, key)
            if exact:
                r = orcut.curvature(g, measure=ms, edges=ids, n_jobs=n_jobs, exact_fraction=True)
                outv[idx] = r['curvature']
                fr[t_][idx, 0] = r['w1_num'].astype(object); fr[t_][idx, 1] = r['w1_den'].astype(object)
            else:
                outv[idx] = orcut.curvature(g, measure=ms, edges=ids, n_jobs=n_jobs)
    return (lo, hi, fr[0], fr[1]) if exact else (lo, hi)
