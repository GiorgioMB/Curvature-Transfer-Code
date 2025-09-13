import os
import sys
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="These tests require PyTorch (torch).")

# Ensure we can import pyg_curvature from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pyg_curvature as pc


class Data:
    """Minimal stub of a PyG Data object with only what CurvatureEngine needs."""
    def __init__(self, num_nodes, edge_index):
        self.num_nodes = int(num_nodes)
        self.edge_index = edge_index


def undirected_edge_index(num_nodes, edges_undirected):
    """Build a directed edge_index (2, 2E) LongTensor from undirected edges list of (u,v)."""
    rows = []
    cols = []
    for u, v in edges_undirected:
        assert 0 <= u < num_nodes and 0 <= v < num_nodes and u != v
        rows.extend([u, v])
        cols.extend([v, u])
    ei = torch.tensor([rows, cols], dtype=torch.long)
    return ei


def graph_catalog():
    """A small catalog of simple graphs used in the paper-style tests."""
    # Path on 4 nodes: 0-1-2-3
    path4_edges = [(0,1), (1,2), (2,3)]
    # 4-cycle: 0-1-2-3-0
    cycle4_edges = [(0,1), (1,2), (2,3), (3,0)]
    # Triangle K3
    tri3_edges = [(0,1), (1,2), (2,0)]
    # Star on 5 nodes centered at 0
    star5_edges = [(0,1), (0,2), (0,3), (0,4)]
    # 2x3 grid (0,1,2 on top; 3,4,5 on bottom)
    grid2x3_edges = [
        (0,1), (1,2), (3,4), (4,5),  # horizontal
        (0,3), (1,4), (2,5)          # vertical
    ]
    # Square with a tail (adds a leaf to a 4-cycle) to create deg variations with triangles=0 on some edges
    sq_tail_edges = cycle4_edges + [(0,4)]  # 4 is a leaf attached to 0

    return {
        "path4": (4, path4_edges),
        "cycle4": (4, cycle4_edges),
        "tri3": (3, tri3_edges),
        "star5": (5, star5_edges),
        "grid2x3": (6, grid2x3_edges),
        "cycle4_tail": (5, sq_tail_edges),
    }


@pytest.fixture(scope="module", params=sorted(graph_catalog().keys()))
def engine(request):
    """CurvatureEngine over each small test graph."""
    name = request.param
    n, undirected_edges = graph_catalog()[name]
    edge_index = undirected_edge_index(n, undirected_edges)
    data = Data(n, edge_index)
    eng = pc.CurvatureEngine(data)
    return eng


@pytest.fixture(scope="module")
def engines_dict():
    """A dict of named engines for tests that need to compare two graphs."""
    out = {}
    for name, (n, undirected_edges) in graph_catalog().items():
        ei = undirected_edge_index(n, undirected_edges)
        out[name] = pc.CurvatureEngine(Data(n, ei))
    return out


EPS = 1e-9
