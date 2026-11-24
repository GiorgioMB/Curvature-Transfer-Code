import numpy as np
import pytest

from curvature_transfer import Graph


def test_canonicalization():
    g = Graph.from_edges([(1, 0), (0, 1), (2, 2), (2, 1), (1, 2), (3, 0)])
    assert g.edges.tolist() == [[0, 1], [0, 3], [1, 2]]
    assert g.deg.tolist() == [2, 2, 1, 1]
    assert g.neighbors(1).tolist() == [0, 2]
    assert g.has_edge(2, 1) and not g.has_edge(2, 3)


def test_edge_ids():
    g = Graph.from_edges([(0, 1), (1, 2), (2, 3)])
    assert g.edge_id(2, 1) == 1 and g.edge_id(0, 3) == -1


def test_isolated_nodes_and_empty():
    g = Graph.from_edges([(0, 1)], num_nodes=5)
    assert g.num_nodes == 5 and g.deg.tolist() == [1, 1, 0, 0, 0]
    e = Graph.from_edges(np.empty((0, 2), int), num_nodes=3)
    assert e.num_edges == 0
    with pytest.raises(ValueError):
        Graph.from_edges([(0, 4)], num_nodes=3)
