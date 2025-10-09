import numpy as np
import pytest

def test_as_undirected_unique_edges_basic(pc, torch_mod):
    f = pc._as_undirected_unique_edges
    ei = torch_mod.tensor([[0,1,1,2,2,0],
                           [1,0,2,1,0,2]], dtype=torch_mod.long)  # triangle (directed)
    out = f(ei, num_nodes=3)
    pairs = out.numpy().T.tolist()
    assert pairs == [[0,1],[0,2],[1,2]]

def test_as_undirected_unique_edges_ignores_self_loops(pc, torch_mod):
    f = pc._as_undirected_unique_edges
    ei = torch_mod.tensor([[0,0,1,2,2,0],
                           [0,1,0,1,0,2]], dtype=torch_mod.long)  # includes (0,0)
    out = f(ei, num_nodes=3)
    pairs = out.numpy().T.tolist()
    assert [0,0] not in pairs
    assert pairs == [[0,1],[0,2],[1,2]]

def test_build_neighbors(pc, torch_mod):
    und = torch_mod.tensor([[0,0,1],
                            [1,2,2]], dtype=torch_mod.long)  # undirected u<v pairs
    neigh = pc._build_neighbors(num_nodes=3, undirected_edges=und)
    assert neigh[0] == {1,2}
    assert neigh[1] == {0,2}
    assert neigh[2] == {0,1}

def test_bfs_dist_limited(pc):
    # Path 0-1-2-3
    neighbors = [{1}, {0,2}, {1,3}, {2}]
    src = 0; targets = {3}
    dd2 = pc._bfs_dist_limited(src, targets, neighbors, max_depth=2)
    assert 3 not in dd2
    dd3 = pc._bfs_dist_limited(src, targets, neighbors, max_depth=3)
    assert dd3[3] == 3

def test_pairwise_distances_between_sets(pc):
    neighbors = [{1}, {0,2}, {1,3}, {2}]  # path 0-1-2-3
    A = [0,1]; B = [2,3]
    C = pc._pairwise_distances_between_sets(A,B,neighbors,default_far=10)
    # Expected:
    # d(0,2)=2, d(0,3)=3; d(1,2)=1, d(1,3)=2
    assert C.shape == (2,2)
    assert C[0,0] == 2 and C[0,1] == 3
    assert C[1,0] == 1 and C[1,1] == 2

def test_values_to_undirected_aggregation(pc, torch_mod):
    # Directed triangle values aggregated to undirected
    ei = torch_mod.tensor([[0,1,1,2,2,0],
                           [1,0,2,1,0,2]], dtype=torch_mod.long)
    vals = np.array([1,2,3,4,5,6], dtype=float)
    eng_like = type("X",(object,),{})()
    eng_like._original_edge_index = ei
    eng_like.edges = [(0,1),(0,2),(1,2)]
    arr_mean = pc.CurvatureEngine._values_to_undirected(eng_like, vals, ei, agg="mean")
    assert np.allclose(arr_mean, [1.5, 5.5, 3.5])
    arr_sum = pc.CurvatureEngine._values_to_undirected(eng_like, vals, ei, agg="sum")
    assert np.allclose(arr_sum, [3.0, 11.0, 7.0])
    arr_min = pc.CurvatureEngine._values_to_undirected(eng_like, vals, ei, agg="min")
    assert np.allclose(arr_min, [1.0, 5.0, 3.0])
    arr_max = pc.CurvatureEngine._values_to_undirected(eng_like, vals, ei, agg="max")
    assert np.allclose(arr_max, [2.0, 6.0, 4.0])
