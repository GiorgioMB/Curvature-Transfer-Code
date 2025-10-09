import numpy as np
import pyg_curvature as pc

def _neighbors_from_edges(n, edges):
    nbrs = [set() for _ in range(n)]
    for u, v in edges:
        nbrs[u].add(v); nbrs[v].add(u)
    return nbrs

def test_bfs_depth_cutoff_and_pairwise_matrix():
    # Path on 6 nodes: 0-1-2-3-4-5
    n = 6
    edges = [(i, i+1) for i in range(n-1)]
    nbrs = _neighbors_from_edges(n, edges)

    # From 0, nodes at distance <=3 are {1,2,3}; 4 and 5 are beyond the cutoff
    found = pc._bfs_dist_limited(0, set([1,2,3,4,5]), nbrs, max_depth=3)
    assert found[1] == 1 and found[2] == 2 and found[3] == 3
    assert 4 not in found and 5 not in found

    C = pc._pairwise_distances_between_sets([0], [3,4,5], nbrs, default_far=10)
    assert C.shape == (1,3)
    assert C[0,0] == 3.0 and C[0,1] == 10.0 and C[0,2] == 10.0
