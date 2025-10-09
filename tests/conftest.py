import os
import sys
import pytest


def pytest_configure(config):
    """Register custom markers used across the test suite."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')",
    )


# Hard dependency for these tests
torch = pytest.importorskip("torch", reason="These tests require PyTorch (torch).")

# Ensure project root is on sys.path so we can import pyg_curvature
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the target module (skip all tests if missing)
pc_mod = pytest.importorskip("pyg_curvature")


class Data:
    """Minimal stub of a PyG-like Data object with only what CurvatureEngine needs."""

    def __init__(self, num_nodes, edge_index):
        self.num_nodes = int(num_nodes)
        self.edge_index = edge_index


def undirected_edge_index(num_nodes, edges_undirected):
    """Build a directed edge_index (2, 2E) LongTensor from undirected edge list [(u,v), ...]."""
    rows, cols = [], []
    for u, v in edges_undirected:
        assert 0 <= u < num_nodes and 0 <= v < num_nodes and u != v
        rows.extend([u, v])
        cols.extend([v, u])
    return torch.tensor([rows, cols], dtype=torch.long)


def graph_catalog():
    """A small catalog of simple graphs used in paper-style tests."""
    path4_edges = [(0, 1), (1, 2), (2, 3)]  # Path on 4 nodes: 0-1-2-3
    cycle4_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # 4-cycle: 0-1-2-3-0
    tri3_edges = [(0, 1), (1, 2), (2, 0)]  # Triangle K3
    star5_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]  # Star on 5 nodes centered at 0
    grid2x3_edges = [  # 2x3 grid (0,1,2 on top; 3,4,5 on bottom)
        (0, 1), (1, 2), (3, 4), (4, 5),  # horizontal
        (0, 3), (1, 4), (2, 5),          # vertical
    ]
    # Square with a tail: 4-cycle plus a leaf attached to node 0
    sq_tail_edges = cycle4_edges + [(0, 4)]

    return {
        "path4": (4, path4_edges),
        "cycle4": (4, cycle4_edges),
        "tri3": (3, tri3_edges),
        "star5": (5, star5_edges),
        "grid2x3": (6, grid2x3_edges),
        "cycle4_tail": (5, sq_tail_edges),
    }


# -------------------- PyTest fixtures --------------------

@pytest.fixture(scope="session")
def pc():
    """Provide the pyg_curvature module to tests via a fixture."""
    return pc_mod


@pytest.fixture(scope="session")
def torch_mod():
    """Provide torch via a fixture when tests prefer injection."""
    return torch


@pytest.fixture(scope="session")
def tg_data():
    """Optional: expose torch_geometric.data if available (else skip tests that require it)."""
    return pytest.importorskip("torch_geometric.data")


@pytest.fixture
def Data_fixture():
    """Fixture that yields the minimal Data class used by CurvatureEngine."""
    return Data


@pytest.fixture
def make_triangle(torch_mod, Data_fixture):
    """K3: 0-1-2-0"""
    def _build():
        ei = torch_mod.tensor(
            [[0, 1, 1, 2, 2, 0],
             [1, 0, 2, 1, 0, 2]],
            dtype=torch_mod.long,
        )
        return Data_fixture(num_nodes=3, edge_index=ei)

    return _build


@pytest.fixture
def make_square(torch_mod, Data_fixture):
    """C4: 0-1-2-3-0"""
    def _build():
        ei = torch_mod.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0],
             [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch_mod.long,
        )
        return Data_fixture(num_nodes=4, edge_index=ei)

    return _build


@pytest.fixture
def make_path3(torch_mod, Data_fixture):
    """P3: 0-1-2"""
    def _build():
        ei = torch_mod.tensor(
            [[0, 1, 1, 2],
             [1, 0, 2, 1]],
            dtype=torch_mod.long,
        )
        return Data_fixture(num_nodes=3, edge_index=ei)

    return _build


@pytest.fixture
def make_star4(torch_mod, Data_fixture):
    """Star with center 0 connected to 1,2,3"""
    def _build():
        ei = torch_mod.tensor(
            [[0, 0, 0, 1, 2, 3],
             [1, 2, 3, 0, 0, 0]],
            dtype=torch_mod.long,
        )
        return Data_fixture(num_nodes=4, edge_index=ei)

    return _build


@pytest.fixture
def new_engine(pc):
    """Factory to build CurvatureEngine instances from Data."""
    def _eng(data, **kwargs):
        return pc.CurvatureEngine(data, **kwargs)

    return _eng


@pytest.fixture(scope="module", params=sorted(graph_catalog().keys()))
def engine(request, pc):
    """CurvatureEngine over each small test graph from the catalog."""
    name = request.param
    n, undirected_edges = graph_catalog()[name]
    edge_index = undirected_edge_index(n, undirected_edges)
    data = Data(n, edge_index)
    return pc.CurvatureEngine(data)


@pytest.fixture(scope="module")
def engines_dict(pc):
    """A dict of named engines for tests that need to compare two graphs."""
    out = {}
    for name, (n, undirected_edges) in graph_catalog().items():
        ei = undirected_edge_index(n, undirected_edges)
        out[name] = pc.CurvatureEngine(Data(n, ei))
    return out


# Tolerance used across tests
EPS = 1e-9
