import os
import sys

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def _w1(G, mu, nu):
    S, T = list(mu), list(nu)
    dist = {s: nx.single_source_shortest_path_length(G, s, cutoff=4) for s in S}
    c = np.array([dist[s][t] for s in S for t in T], float)
    m, n = len(S), len(T)
    A = np.zeros((m + n, m * n)); b = np.zeros(m + n)
    for a in range(m):
        A[a, a * n:(a + 1) * n] = 1; b[a] = mu[S[a]]
    for k in range(n):
        A[m + k, k::n] = 1; b[m + k] = nu[T[k]]
    return linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs").fun


def reference_curvature(G, i, j, kind):
    """LP reference for the lazy (paper idleness), non-lazy and LLY curvature."""
    di, dj = G.degree(i), G.degree(j)
    if kind == "c_OR":
        mu = {u: 1 / (di + 1) for u in list(G[i]) + [i]}
        nu = {u: 1 / (dj + 1) for u in list(G[j]) + [j]}
        return 1 - _w1(G, mu, nu)
    if kind == "c_OR0":
        return 1 - _w1(G, {u: 1 / di for u in G[i]}, {u: 1 / dj for u in G[j]})
    p = 1 / (max(di, dj) + 1)
    mu = {u: (1 - p) / di for u in G[i]}; mu[i] = p
    nu = {u: (1 - p) / dj for u in G[j]}; nu[j] = p
    return (1 - _w1(G, mu, nu)) / (1 - p)


def small_graphs():
    gs = [nx.gnp_random_graph(30, 0.18, seed=1), nx.barabasi_albert_graph(40, 3, seed=2),
          nx.watts_strogatz_graph(36, 6, 0.2, seed=3), nx.random_geometric_graph(40, 0.28, seed=4),
          nx.powerlaw_cluster_graph(40, 3, 0.6, seed=5), nx.complete_graph(6), nx.hypercube_graph(3),
          nx.grid_2d_graph(4, 5), nx.balanced_tree(2, 3), nx.cycle_graph(6), nx.star_graph(5),
          nx.path_graph(4), nx.petersen_graph(), nx.karate_club_graph()]
    return [nx.convert_node_labels_to_integers(g) for g in gs]


def random_graphs(n_seeds=6):
    out = []
    for s in range(n_seeds):
        out += [nx.gnp_random_graph(50, 0.12, seed=s), nx.barabasi_albert_graph(60, 1 + s % 4, seed=s),
                nx.watts_strogatz_graph(50, 4 + 2 * (s % 3), 0.15, seed=s), nx.random_geometric_graph(60, 0.22, seed=s),
                nx.powerlaw_cluster_graph(60, 3, 0.5, seed=s), nx.random_regular_graph(4 + s % 3, 30, seed=s),
                nx.stochastic_block_model([25, 25], [[0.3, 0.05], [0.05, 0.3]], seed=s)]
    return [nx.convert_node_labels_to_integers(g) for g in out]


@pytest.fixture(scope="session")
def ref():
    return reference_curvature
