#!/usr/bin/env python3
"""
Edgewise comparison of combinatorial Ollivier-Ricci (OR) bounds against exact OR.

Compares, on every edge (or a uniform edge sample for dense graphs):
  - non-lazy OR  c_OR-0 = 1 - W1(nu_i, nu_j)       (nu_u uniform on N(u))
      lower: Jost-Liu (2014), Tian-Lubberts-Weber (2025, Thm 9.2),
             ours = Theorem (quadrangle-augmented Jost-Liu, matching term)
      upper: Jost-Liu (2014) triangle bound, Tian-Lubberts-Weber (2025, Thm 9.1, dual),
             ours = lazy transport envelope evaluated at zero idleness
  - lazy OR    c_OR = 1 - W1(m_i, m_j), m_u uniform on the closed neighbourhood (alpha_u = 1/(deg u + 1))
      lower: Jost-Liu vs ours, both lifted with the lazy/non-lazy transfer (computable idleness choice)
      upper: best prior = min(Tian dual bound, zero-cost-mass bound) vs ours (lazy transport envelope)
  - ranking fidelity of cheap surrogates for exact lazy OR: Spearman rho and top-10% recall of the most
    negatively curved edges, for BF, ORC-A (mean of Jost-Liu bounds, Tian et al.) and our bound midpoint.

Usage
  python compare_bounds.py                      # regenerated synthetic suite (seeded) + Karate
  python compare_bounds.py --edgelist jazz.csv --name Jazz   # any CSV edge list (u,v per line, no header)
Outputs a CSV (bound_comparison.csv) and LaTeX rows (bound_comparison_rows.tex).
"""

import argparse, math, time, sys, csv, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)                     
sys.path.insert(0, os.path.dirname(_script_dir))    

import numpy as np, networkx as nx
from scipy.stats import spearmanr
import torch
from torch_geometric.data import Data
from pyg_curvature import CurvatureEngine
import models

TOL = 1e-9

def local_dist(A, S):
    AS = A[S].astype(np.float32)
    common = (AS @ AS.T) > 0.5
    D = np.full((len(S), len(S)), 3.0)
    D[common] = 2.0
    D[A[np.ix_(S, S)]] = 1.0
    np.fill_diagonal(D, 0.0)
    return D

def jl_type_lower(di, dj, t, S=0.0):
    K = 1.0 - 1.0 / di - 1.0 / dj
    zmax = t / np.maximum(di, dj)
    zmin = t / np.minimum(di, dj)
    return zmax - np.maximum(K - zmax - S, 0.0) - np.maximum(K - zmin - S, 0.0)

def envelope_upper(di, dj, t, nUi, nUj, Xi, ai, aj):
    wi, wj = (1.0 - ai) / di, (1.0 - aj) / dj
    wmin = np.minimum(wi, wj)
    Sig = di / (1.0 - ai) + dj / (1.0 - aj)
    zi, zj = np.minimum(ai, wj), np.minimum(aj, wi)
    ends = np.abs(ai - wj) + np.abs(aj - wi)
    mUU = np.minimum(np.minimum(nUi * wi, nUj * wj), Xi / Sig)
    mT = np.minimum(t * np.abs(wi - wj), nUi * wi + nUj * wj)
    return -1.0 + 2.0 * (zi + zj) + ends + 2.0 * t * wmin + mUU + mT

def zero_cost_upper(di, dj, t, ai, aj):
    """c_OR <= total zero-cost mass (the Jost-Liu upper-bound argument, lazy version)."""
    wi, wj = (1.0 - ai) / di, (1.0 - aj) / dj
    return np.minimum(ai, wj) + np.minimum(aj, wi) + t * np.minimum(wi, wj)

def lift_lower(L0, ai, aj):
    """Lazy lower bound from a non-lazy lower bound L0, with the computable idleness choice."""
    amin, amax = np.minimum(ai, aj), np.maximum(ai, aj)
    beta = np.where(L0 >= 0, amin, amax)
    return (1.0 - beta) * L0 - (amax - amin)

def tian_lower_nonlazy(di, dj, t, nUi, nUj):
    """Tian-Lubberts-Weber (2025), Thm 9.2, unweighted graph, alpha = 0 (uniform neighbour measures)."""
    mx, my = 1.0 / di, 1.0 / dj
    Lx = nUi * mx
    cc = t * np.abs(mx - my)
    cross = Lx - my - t * np.maximum(my - mx, 0.0)
    return 1.0 - nUi * mx - nUj * my - cc - np.abs(cross)

def tian_dual_upper(mu, nu, D):
    """Tian-Lubberts-Weber (2025), Thm 9.1: 1 - W1 lower bound via h = dist(., N) or dist(., P)."""
    diff = mu - nu
    P = diff > TOL; N = diff < -TOL
    if not P.any():
        return 1.0
    a = float((D[:, N][P].min(axis=1) * diff[P]).sum())
    b = float((D[:, P][N].min(axis=1) * (-diff[N])).sum())
    return 1.0 - max(a, b)

def evaluate(G, name, max_edges=None, seed=0):
    G_core = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G_core = nx.convert_node_labels_to_integers(G_core)
    n = G_core.number_of_nodes()
    A = nx.to_numpy_array(G_core, dtype=bool)
    
    # Build PyG Data and evaluate through CurvatureEngine
    edges = list(G_core.edges())
    ei = torch.tensor(edges, dtype=torch.long).T
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    data = Data(num_nodes=n, edge_index=ei)
    
    eng = CurvatureEngine(data, n_jobs=None)
    base = eng.compute_all()
    M = len(base["edges"])
    
    # Subsample
    if max_edges and M > max_edges:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, max_edges, replace=False)
    else:
        idx = np.arange(M)
        
    u_arr, v_arr = base["edges"][idx, 0], base["edges"][idx, 1]
    di, dj = base["deg_i"][idx], base["deg_j"][idx]
    t = base["triangle"][idx]
    Xi, C4 = base["Xi"][idx], base["C4"][idx]
    smax = base["smax"][idx]
    ex0, ex = base["c_OR0"][idx], base["c_OR"][idx]
    lo_BF = base["c_BF"][idx]
    
    nUi, nUj = np.maximum(0, di - 1 - t), np.maximum(0, dj - 1 - t)
    ai, aj = 1.0 / (di + 1.0), 1.0 / (dj + 1.0)
    
    # Vectorized Bounds
    lo_JL = jl_type_lower(di, dj, t, S=0.0)
    lo_ours = jl_type_lower(di, dj, t, S=smax)
    lo_Tian = tian_lower_nonlazy(di, dj, t, nUi, nUj)
    
    up_JL = t / np.maximum(di, dj)
    up_ours = envelope_upper(di, dj, t, nUi, nUj, Xi, 0.0, 0.0)
    
    Llo_JL = lift_lower(lo_JL, ai, aj)
    Llo_ours = lift_lower(lo_ours, ai, aj)
    
    Lup_ours = envelope_upper(di, dj, t, nUi, nUj, Xi, ai, aj)
    zc_upper = zero_cost_upper(di, dj, t, ai, aj)
    
    # Tian dual upper bound (requires distance matrix per edge)
    up_Tian = np.zeros(len(idx))
    Lup_prior = np.zeros(len(idx))
    
    for k, _ in enumerate(idx):
        i, j = u_arr[k], v_arr[k]
        Ni, Nj = set(G_core[i]), set(G_core[j])
        S = sorted(Ni | Nj | {i, j})
        pos = {v: c for c, v in enumerate(S)}
        D = local_dist(A, S)
        
        nu_i = np.zeros(len(S)); nu_j = np.zeros(len(S))
        nu_i[[pos[v] for v in Ni]] = 1.0 / di[k]
        nu_j[[pos[v] for v in Nj]] = 1.0 / dj[k]
        up_Tian[k] = tian_dual_upper(nu_i, nu_j, D)
        
        m_i = (1.0 - ai[k]) * nu_i; m_i[pos[i]] += ai[k]
        m_j = (1.0 - aj[k]) * nu_j; m_j[pos[j]] += aj[k]
        Lup_prior[k] = min(tian_dual_upper(m_i, m_j, D), zc_upper[k])
        
    rows = []
    for k in range(len(idx)):
        rows.append(dict(
            ex0=ex0[k], ex=ex[k],
            lo_JL=lo_JL[k], lo_Tian=lo_Tian[k], lo_BF=lo_BF[k], lo_ours=lo_ours[k],
            up_JL=up_JL[k], up_Tian=up_Tian[k], up_ours=up_ours[k],
            Llo_JL=Llo_JL[k], Llo_ours=Llo_ours[k],
            Lup_prior=Lup_prior[k], Lup_ours=Lup_ours[k]
        ))
        
    return summarize(name, M, rows, seed)

def topk_recall(exact, proxy, frac=0.1, seed=0):
    k = max(1, int(math.ceil(frac * len(exact))))
    rng = np.random.default_rng(seed)
    te = set(np.lexsort((rng.random(len(exact)), exact))[:k])
    tp = set(np.lexsort((rng.random(len(proxy)), proxy))[:k])
    return len(te & tp) / k

def summarize(name, nE, rows, seed):
    col = lambda k: np.array([r[k] for r in rows])
    ex0, ex = col("ex0"), col("ex")
    out = dict(graph=name, E=nE, sampled=len(rows))
    viol = 0
    for k in ["lo_JL", "lo_Tian", "lo_ours"]:
        out["gap_" + k] = float(np.mean(ex0 - col(k))); viol += int((col(k) > ex0 + 1e-7).sum())
    for k in ["up_JL", "up_Tian", "up_ours"]:
        out["gap_" + k] = float(np.mean(col(k) - ex0)); viol += int((col(k) < ex0 - 1e-7).sum())
    for k in ["Llo_JL", "Llo_ours"]:
        out["gap_" + k] = float(np.mean(ex - col(k))); viol += int((col(k) > ex + 1e-7).sum())
    for k in ["Lup_prior", "Lup_ours"]:
        out["gap_" + k] = float(np.mean(col(k) - ex)); viol += int((col(k) < ex - 1e-7).sum())
    best_lo = np.maximum(col("lo_JL"), col("lo_Tian"))
    best_up = np.minimum(col("up_JL"), col("up_Tian"))
    out["lo_tighter"] = float(np.mean(col("lo_ours") > best_lo + TOL))
    out["lo_looser"] = float(np.mean(col("lo_ours") < best_lo - TOL))
    out["up_tighter"] = float(np.mean(col("up_ours") < best_up - TOL))
    out["up_looser"] = float(np.mean(col("up_ours") > best_up + TOL))
    out["Llo_tighter"] = float(np.mean(col("Llo_ours") > col("Llo_JL") + TOL))
    out["Lup_tighter"] = float(np.mean(col("Lup_ours") < col("Lup_prior") - TOL))
    out["Lup_looser"] = float(np.mean(col("Lup_ours") > col("Lup_prior") + TOL))
    # sign certification of the lazy curvature (lower > 0 certifies positive, upper < 0 certifies negative)
    up_comb = np.minimum(col("Lup_ours"), col("Lup_prior"))
    cert_prior = (col("Llo_JL") > TOL) | (col("Lup_prior") < -TOL)
    cert_ours = (col("Llo_ours") > TOL) | (up_comb < -TOL)
    out["cert_prior"] = float(np.mean(cert_prior)); out["cert_ours"] = float(np.mean(cert_ours))
    out["certpos_prior"] = float(np.mean(col("Llo_JL") > TOL)); out["certpos_ours"] = float(np.mean(col("Llo_ours") > TOL))
    out["certneg_prior"] = float(np.mean(col("Lup_prior") < -TOL)); out["certneg_ours"] = float(np.mean(up_comb < -TOL))
    proxies = dict(BF=col("lo_BF"), ORCA=0.5 * (col("lo_JL") + col("up_JL")), ours=col("Llo_ours"),
                   midprior=0.5 * (col("Llo_JL") + col("Lup_prior")),     # midpoint of the best prior certified interval
                   mid=0.5 * (col("Llo_ours") + up_comb))                  # midpoint of the sharpened certified interval
    out["lo_relgain"] = float(1 - out["gap_lo_ours"] / max(min(out["gap_lo_JL"], out["gap_lo_Tian"]), 1e-12))
    for k, p in proxies.items():
        rho = spearmanr(ex, p).correlation if np.std(ex) > 0 and np.std(p) > 0 else float("nan")
        out["rho_" + k] = float(rho); out["rec_" + k] = topk_recall(ex, p, 0.1, seed)
    out["violations"] = viol
    return out

def graph_from_edges(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

def suite(seed=0):
    return [
        ("BA(800, 2)", graph_from_edges(*models.barabasi_albert(800, 2, seed=seed)), None),
        ("BA(800, 5)", graph_from_edges(*models.barabasi_albert(800, 5, seed=seed)), None),
        ("BA(1600, 2)", graph_from_edges(*models.barabasi_albert(1600, 2, seed=seed)), None),
        ("BA(1600, 5)", graph_from_edges(*models.barabasi_albert(1600, 5, seed=seed)), None),
        ("G(800, 0.010013)", graph_from_edges(*models.erdos_renyi(800, 0.010013, seed=seed)), None),
        ("G(1600, 0.005003)", graph_from_edges(*models.erdos_renyi(1600, 0.005003, seed=seed)), None),
        ("RGG(800, 0.056419)", graph_from_edges(*models.random_geometric(800, 0.056419, seed=seed)), None),
        ("RGG(1600, 0.039894)", graph_from_edges(*models.random_geometric(1600, 0.039894, seed=seed)), None),
        ("HRG(800, 5.0, 1.0, 0.0)", graph_from_edges(*models.make_hyperbolic_random_graph(800, 5.0, 1.0, 0.0, seed=seed, n_jobs=-1)), 2000),
        ("HRG(800, 5.0, 1.0, 0.5)", graph_from_edges(*models.make_hyperbolic_random_graph(800, 5.0, 1.0, 0.5, seed=seed, n_jobs=-1)), 1000),
        ("WS(800,10,0.05)", graph_from_edges(*models.watts_strogatz(800, 10, 0.05, seed=seed)), None),
        ("WS(800,10,0.2)", graph_from_edges(*models.watts_strogatz(800, 10, 0.2, seed=seed)), None),
        ("WS(1600,10,0.05)", graph_from_edges(*models.watts_strogatz(1600, 10, 0.05, seed=seed)), None),
        ("WS(1600,10,0.2)", graph_from_edges(*models.watts_strogatz(1600, 10, 0.2, seed=seed)), None),
        ("SBM(2x500, 0.012, 0.004)", graph_from_edges(*models.stochastic_block_model([500, 500], [[0.012018, 0.004006], [0.004006, 0.012018]], seed=seed)), None),
        ("Grid(40,40)", graph_from_edges(*models.grid_graph(40, 40)), None),
        ("Karate", nx.karate_club_graph(), None),
    ]

def load_edgelist_csv(filepath):
    edges = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                u, v = int(row[0].strip()), int(row[1].strip())
                edges.append((u, v))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def latex_row(o):
    f = lambda x: f"{x:.3f}"; p = lambda x: f"{100 * x:.1f}"
    a = (f"{o['graph']} & {o['E']} & {f(o['gap_lo_JL'])} & {f(o['gap_lo_Tian'])} & "
         f"\\textbf{{{f(o['gap_lo_ours'])}}} & {p(o['lo_tighter'])} & {f(o['gap_up_JL'])} & {f(o['gap_up_Tian'])} & "
         f"{f(o['gap_up_ours'])} & {p(o['up_tighter'])} / {p(o['up_looser'])} \\\\")
    b = (f"{o['graph']} & {f(o['gap_Llo_JL'])} & \\textbf{{{f(o['gap_Llo_ours'])}}} & {p(o['Llo_tighter'])} & "
         f"{f(o['gap_Lup_prior'])} & {f(o['gap_Lup_ours'])} & {p(o['Lup_tighter'])} & "
         f"{p(o['cert_prior'])} & {p(o['cert_ours'])} \\\\")
    c = (f"{o['graph']} & {o['rho_BF']:.2f} & {o['rho_ORCA']:.2f} & {o['rho_midprior']:.2f} & {o['rho_mid']:.2f} & "
         f"{o['rec_BF']:.2f} & {o['rec_ORCA']:.2f} & {o['rec_midprior']:.2f} & {o['rec_mid']:.2f} \\\\")
    return a, b, c

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgelist"); ap.add_argument("--name", default="custom")
    ap.add_argument("--extra-csvs", nargs='+', default=[], help="Format: 'Name=filepath.csv'")
    ap.add_argument("--max-edges", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="semicolon-separated subset of suite names")
    args = ap.parse_args()
    
    jobs = []
    if args.edgelist:
        G = load_edgelist_csv(args.edgelist)
        jobs = [(args.name, G, args.max_edges)]
    else:
        jobs = suite(args.seed)
        
    for item in args.extra_csvs:
        if "=" not in item:
            print(f"[ERROR] Invalid --extra-csvs argument: '{item}'")
            print("Ensure the entire Name=Path pair is enclosed in quotes if the path contains spaces.")
            sys.exit(1)
        name, path = item.split("=", 1)
        G = load_edgelist_csv(path)
        jobs.append((name, G, args.max_edges))
        
    if args.only:
        keep = set(s.strip() for s in args.only.split(";"))
        jobs = [j for j in jobs if j[0] in keep]
        
    results = []
    for name, G, me in jobs:
        t0 = time.time(); o = evaluate(G, name, me, args.seed); results.append(o)
        print(f"{name:26s} E={o['E']:6d} sampled={o['sampled']:5d} viol={o['violations']} "
              f"lo gaps JL/Tian/ours={o['gap_lo_JL']:.3f}/{o['gap_lo_Tian']:.3f}/{o['gap_lo_ours']:.3f} "
              f"up gaps JL/Tian/ours={o['gap_up_JL']:.3f}/{o['gap_up_Tian']:.3f}/{o['gap_up_ours']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    with open("bound_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    with open("bound_comparison_rows.tex", "w") as fh:
        fh.write("% Table A rows (non-lazy)\n" + "\n".join(latex_row(o)[0] for o in results) + "\n")
        fh.write("% Table B rows (lazy bounds + sign certification)\n" + "\n".join(latex_row(o)[1] for o in results) + "\n")
        fh.write("% Table C rows (ranking fidelity)\n" + "\n".join(latex_row(o)[2] for o in results) + "\n")
