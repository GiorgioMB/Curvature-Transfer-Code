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
    nT = len(twins); nG = len(cand)
    if nT == 0:                                            # nothing can be opened: the upper end is the lower end
        val = _transport(sup, dem, arcs_p, arcs_q, arcs_c, len(P), len(Q))
        return (1.0 - val, True) if val is not None else (np.nan, False)
    nV = nA + nT + nG
    c = np.concatenate([np.array(arcs_c, float), np.full(nT, 2.0), np.zeros(nG)])
    nP, nQ = len(P), len(Q)
    tw_p = np.array([t[1] for t in twins], np.int64); tw_q = np.array([t[2] for t in twins], np.int64)
    ap = np.array(arcs_p, np.int64); aq = np.array(arcs_q, np.int64)
    # supply rows 0..nP-1 and demand rows nP..nP+nQ-1, for the arcs and their twins
    rows = list(np.concatenate([ap, nP + aq, tw_p, nP + tw_q]))
    cols = list(np.concatenate([np.arange(nA), np.arange(nA), nA + np.arange(nT), nA + np.arange(nT)]))
    vals = [1.0] * len(rows)
    lo = list(sup) + list(dem); hi = list(sup) + list(dem)
    R = nP + nQ
    cap = float(max(sup.max(), dem.max()))
    for t_, (g, aa, bb, x, y) in enumerate(twins):       # a twin carries mass only through an open pair
        rows.append(R); cols.append(nA + t_); vals.append(1.0)
        for k in hub_of[(x, y)]:
            rows.append(R); cols.append(nA + nT + cand[k]); vals.append(-cap)
        lo.append(-np.inf); hi.append(0.0); R += 1
    slots = {}
    for k, q in cand.items():
        if k[0] == 'z':
            slots.setdefault(('y', k[2]), []).append(q); slots.setdefault(('z', k[1]), []).append(q)
        else:
            slots.setdefault(('y', k[1]), []).append(q); slots.setdefault(('y', k[2]), []).append(q)
    for (kind, w), qs in slots.items():
        bound = B['sy'][w] if kind == 'y' else B['sz'][w]
        for q in qs:
            rows.append(R); cols.append(nA + nT + q); vals.append(1.0)
        lo.append(-np.inf); hi.append(float(bound)); R += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(R, nV))
    integ = np.concatenate([np.zeros(nA + nT), np.ones(nG)])
    ub = np.concatenate([np.full(nA + nT, np.inf), np.ones(nG)])
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nV), ub), options={'time_limit': time_limit})
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


def lower_and_relaxed(adj, deg, edges, measure, n_jobs=None):
    """(kappa_lower*, kappa_relaxed) for every edge: orcut on the adversarial local graphs and on the
    friendly local graphs without free degrees, batched as disjoint unions (one call per idleness value)."""
    groups = {}
    for e, (u, v) in enumerate(edges):
        key = 1.0 / (max(deg[u], deg[v]) + 1) if measure == 'wj' else None
        groups.setdefault(key, []).append(e)
    lo = np.full(len(edges), np.nan); hi = np.full(len(edges), np.nan)
    for key, idx in groups.items():
        for friendly, outv in ((False, lo), (True, hi)):
            allE = []; des = []; orig = []; off = 0
            for e in idx:
                B = edge_blocks(adj, deg, *edges[e])
                E, extra = _local_edges(adj, deg, B, friendly)
                verts = sorted({w for ab in E for w in ab})
                lab = {w: off + t for t, w in enumerate(verts)}
                allE.extend((lab[a], lab[b]) for a, b in E)
                nloc = len(verts)
                if extra:
                    zl = off + nloc
                    allE.extend((lab[w], zl) for w in extra)
                    nloc += 1
                orig.extend(verts + ([None] if extra else []))
                des.append((lab[B['i']], lab[B['j']]))
                off += nloc
            g = orcut.Graph.from_edges(allE)
            ge = np.asarray(g.edges)
            pos = {(int(a), int(b)): t for t, (a, b) in enumerate(ge)}
            ids = np.array([pos[(min(a, b), max(a, b))] for a, b in des])
            if measure == 'lazy':
                ms = 'lazy'
            elif measure == 'wj':
                ms = key
            elif measure == 'degree':
                ind = np.asarray(g.indices)
                dg = np.array([deg[w] if w is not None else 1 for w in orig], float)
                ms = Walk(weights=1.0 / dg[ind], idleness='lazy')
            else:
                ms = float(measure)
            k = orcut.curvature(g, measure=ms, edges=ids, n_jobs=n_jobs)
            outv[np.array(idx)] = k
    return lo, hi
