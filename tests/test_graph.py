import numpy as np
import pytest

from curvature_transfer import Graph


def test_canonicalization():
    g = Graph.from_edges([(1, 0), (0, 1), (2, 2), (2, 1), (1, 2), (3, 0)])
    assert g.edges.tolist() == [[0, 1], [0, 3], [1, 2]]
    assert g.deg.tolist() == [2, 2, 1, 1]
    assert g.neighbors(1).tolist() == [0, 2]
    assert g.has_edge(2, 1) and not g.has_edge(2, 3)


def test_edge_ids_and_directed_aggregation():
    g = Graph.from_edges([(0, 1), (1, 2), (2, 3)])
    assert g.edge_id(2, 1) == 1 and g.edge_id(0, 3) == -1
    assert g.edge_ids(np.array([[1, 0], [3, 2], [0, 2]])).tolist() == [0, 2, -1]
    ei = g.to_edge_index()
    vals = np.arange(ei.shape[1], dtype=float)
    agg = g.directed_to_undirected(vals, ei, "mean")
    for e, (u, v) in enumerate(g.edges):
        mask = ((ei[0] == u) & (ei[1] == v)) | ((ei[0] == v) & (ei[1] == u))
        assert agg[e] == pytest.approx(vals[mask].mean())


def test_isolated_nodes_and_empty():
    g = Graph.from_edges([(0, 1)], num_nodes=5)
    assert g.num_nodes == 5 and g.deg.tolist() == [1, 1, 0, 0, 0]
    e = Graph.from_edges(np.empty((0, 2), int), num_nodes=3)
    assert e.num_edges == 0
    with pytest.raises(ValueError):
        Graph.from_edges([(0, 4)], num_nodes=3)


def test_with_edges():
    g = Graph.from_edges([(0, 1), (1, 2)])
    h = g.with_edges(add=[(2, 0)], remove=[(1, 0)])
    assert h.edges.tolist() == [[0, 2], [1, 2]]
