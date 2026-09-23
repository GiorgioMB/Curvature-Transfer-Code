import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Curvature rewiring requires PyTorch.")
tg_data = pytest.importorskip("torch_geometric.data", reason="Curvature rewiring requires PyG.")
rewiring = pytest.importorskip("curvature_rewiring")

def _clone_data(data):
    """Helper to clone the mock Data object from conftest.py."""
    return type(data)(num_nodes=data.num_nodes, edge_index=data.edge_index.clone())

def test_fast_pyg_edge_index(torch_mod):
    """Ensure edge sets correctly resolve to canonically ordered, bidirectional PyG tensors."""
    edges = {(1, 2), (0, 1)}
    device = torch_mod.device('cpu')
    ei = rewiring._fast_pyg_edge_index(edges, device)
    
    assert ei.shape == (2, 4), "Should contain 2 undirected edges (4 directed edges)"
    
    expected = torch_mod.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch_mod.long)
    assert torch_mod.equal(ei, expected), "Edge index is not canonically ordered"

    # Edge case: Empty set
    ei_empty = rewiring._fast_pyg_edge_index(set(), device)
    assert ei_empty.shape == (2, 0)
    assert ei_empty.dtype == torch_mod.long

def test_evaluate_metric_all(make_triangle, new_engine):
    """Test optimized vector extraction of exact and bounded metrics."""
    eng = new_engine(make_triangle(), n_jobs=1)
    
    bf_vals = rewiring._evaluate_metric_all(eng, "c_BF", n_jobs=1)
    assert bf_vals.shape == (3,)
    assert np.allclose(bf_vals, 1.5)

    or_vals = rewiring._evaluate_metric_all(eng, "c_OR", n_jobs=1)
    assert np.allclose(or_vals, 1.0)
    
    bound_vals = rewiring._evaluate_metric_all(eng, "c_OR_lower_from_c_BF", n_jobs=1)
    assert bound_vals.shape == (3,)
    
    with pytest.raises(ValueError, match="Unsupported metric"):
        rewiring._evaluate_metric_all(eng, "invalid_metric")

def test_evaluate_metric_single(make_triangle, new_engine):
    """Test optimized O(1) single-edge metric extraction."""
    eng = new_engine(make_triangle(), n_jobs=1)
    
    single_bf = rewiring._evaluate_metric_single(eng, "c_BF", 0)
    assert np.isclose(single_bf, 1.5)

    single_bound = rewiring._evaluate_metric_single(eng, "c_OR_lower_from_c_BF", 0)
    all_bounds = rewiring._evaluate_metric_all(eng, "c_OR_lower_from_c_BF", n_jobs=1)
    assert np.isclose(single_bound, all_bounds[0]), "Single extraction deviated from vector extraction"

    with pytest.raises(ValueError, match="Unsupported metric"):
        rewiring._evaluate_metric_single(eng, "invalid_metric", 0)

def test_sdrf_rewiring_adds_edges(make_square):
    """Test SDRF identifies lowest curvature and adds a cycle-closing edge."""
    data = make_square() 
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.SDRFRewiring(
        metric="c_BF", 
        max_iters=1, 
        max_candidates=2, 
        remove_edges=False, 
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_data.num_nodes == 4
    assert new_edges == original_edges + 1, "SDRF failed to add exactly one diagonal to the square"
    
    assert new_data.edge_index.shape[0] == 2
    # Verify canonical sort manually via numpy
    order = np.lexsort((new_data.edge_index[1].numpy(), new_data.edge_index[0].numpy()))
    assert torch.equal(new_data.edge_index, new_data.edge_index[:, order])

def test_sdrf_rewiring_removes_edges(make_square):
    """Test SDRF removal functionality enforces connectivity constraints."""
    data = make_square()
    
    transform = rewiring.SDRFRewiring(
        metric="c_OR_lower_from_c_BF", 
        max_iters=3, 
        max_candidates=3, 
        remove_edges=True, 
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_edges >= 3, "SDRF pruned too aggressively and broke components"

def test_borf_rewiring_structure(make_square):
    """Test BORF batched additions and removals execute without exceptions."""
    data = make_square()
    
    transform = rewiring.BORFRewiring(
        metric="c_OR", 
        max_iters=2, 
        batch_add=2, 
        batch_remove=1, 
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    
    assert new_data.edge_index.shape[0] == 2
    assert new_data.edge_index.shape[1] % 2 == 0, "BORF produced asymmetric (directed) edges"
    assert new_data.num_nodes == 4

def test_borf_rewiring_empty_graph(torch_mod):
    """Ensure BORF safely handles degenerate graphs."""
    empty_ei = torch_mod.empty((2, 0), dtype=torch_mod.long)
    data = tg_data.Data(num_nodes=5, edge_index=empty_ei)
    
    transform = rewiring.BORFRewiring(max_iters=2, batch_add=1, batch_remove=1, n_jobs=1)
    new_data = transform(_clone_data(data))
    
    assert new_data.edge_index.shape[1] == 0, "Should handle empty edge sets gracefully"

def test_sdrf_empty_graph(torch_mod):
    """Ensure SDRF safely handles degenerate graphs."""
    empty_ei = torch_mod.empty((2, 0), dtype=torch_mod.long)
    data = tg_data.Data(num_nodes=5, edge_index=empty_ei)
    
    transform = rewiring.SDRFRewiring(max_iters=2, n_jobs=1)
    new_data = transform(_clone_data(data))
    
    assert new_data.edge_index.shape[1] == 0, "Should handle empty edge sets gracefully"

def test_sdrf_path3_forms_triangle(make_path3):
    """
    On a P3 graph (0-1-2), the lowest curvature edges are the only edges.
    SDRF should predictably find the missing edge (0,2) to form a triangle.
    """
    data = make_path3()
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.SDRFRewiring(
        metric="c_BF", 
        max_iters=1, 
        max_candidates=3, 
        remove_edges=False, 
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_data.num_nodes == 3
    assert original_edges == 2
    assert new_edges == 3, "SDRF must close the P3 into a Triangle (K3)"


def test_borf_path3_forms_triangle(make_path3):
    """BORF on P3 should also deterministically find the only missing edge (0,2)."""
    data = make_path3()
    
    transform = rewiring.BORFRewiring(
        metric="c_OR_lower_from_c_BF", 
        max_iters=1, 
        batch_add=1, 
        batch_remove=0, 
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_edges == 3, "BORF must close the P3 into a Triangle (K3)"


def test_borf_star4_protects_leaves(make_star4):
    """
    On a Star graph, all peripheral nodes are leaves (degree 1). 
    BORF should add edges between leaves (forming triangles) but must NEVER 
    remove the structural hub edges, even if we request aggressive removal.
    """
    data = make_star4()
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.BORFRewiring(
        metric="c_OR", 
        max_iters=1, 
        batch_add=1, 
        batch_remove=10, # Request aggressive removal
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    # It should add 1 edge between leaves, but 0 edges should be removed 
    # because the degree constraint (deg > 1) protects the leaves.
    assert new_edges == original_edges + 1, "BORF removed an edge it wasn't supposed to, or failed to add one"


def test_sdrf_fully_connected_terminates(make_triangle):
    """
    SDRF should gracefully terminate and do nothing if the graph is fully connected 
    and no further candidate edges can be generated.
    """
    data = make_triangle()
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.SDRFRewiring(
        metric="c_BF", 
        max_iters=5, 
        remove_edges=False, 
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_edges == original_edges, "SDRF should not modify a fully connected graph when removal is False"


def test_sdrf_protects_bridges(torch_mod, tg_data):
    """
    Custom graph: Triangle with a tail (Kite/Paw graph).
    Edges: (0,1), (1,2), (2,0) [Triangle] and (0,3) [Tail].
    Ensures highest curvature edges inside the triangle might be removed, 
    but the bridge (0,3) is strictly protected.
    """
    ei = torch_mod.tensor([
        [0, 1, 1, 2, 2, 0, 0, 3],
        [1, 0, 2, 1, 0, 2, 3, 0]
    ], dtype=torch_mod.long)
    data = tg_data.Data(num_nodes=4, edge_index=ei)
    
    transform = rewiring.SDRFRewiring(
        metric="c_BF", 
        max_iters=1, 
        max_candidates=0, # Disable additions to isolate removal logic
        remove_edges=True,
        n_jobs=1
    )
    
    new_data = transform(_clone_data(data))
    
    # Extract the undirected edges present after transformation
    edges = set()
    for i in range(new_data.edge_index.shape[1]):
        u = new_data.edge_index[0, i].item()
        v = new_data.edge_index[1, i].item()
        edges.add(tuple(sorted((u, v))))
        
    assert tuple(sorted((0, 3))) in edges, "The bridge to the leaf (0,3) was illegally removed"

def test_fosr_empty_graph(torch_mod):
    """Ensure FoSR safely handles degenerate graphs with no edges."""
    empty_ei = torch_mod.empty((2, 0), dtype=torch_mod.long)
    data = tg_data.Data(num_nodes=5, edge_index=empty_ei)
    
    transform = rewiring.FoSRRewiring(max_iters=2, remove_edges=True)
    new_data = transform(_clone_data(data))
    
    assert new_data.edge_index.shape[1] == 0, "Should handle empty edge sets gracefully without crashing"

def test_fosr_path3_forms_triangle(make_path3):
    """
    On a P3 graph (0-1-2), nodes 0 and 2 will have the maximum difference 
    in the Fiedler vector. FoSR must connect them, forming a K3 triangle.
    """
    data = make_path3()
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.FoSRRewiring(max_iters=1, remove_edges=False)
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_data.num_nodes == 3
    assert original_edges == 2
    assert new_edges == 3, "FoSR failed to maximize algebraic connectivity by closing the P3 into a Triangle"

def test_fosr_square_adds_diagonal(make_square):
    """Test FoSR adds a cycle-closing diagonal edge to a square graph."""
    data = make_square() 
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.FoSRRewiring(max_iters=1, remove_edges=False)
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_data.num_nodes == 4
    assert new_edges == original_edges + 1, "FoSR failed to add exactly one diagonal to the square"

def test_fosr_fully_connected_terminates(make_triangle):
    """
    FoSR should gracefully terminate and do nothing if the graph is fully 
    connected (clique) and no further candidate edges can be added.
    """
    data = make_triangle()
    original_edges = data.edge_index.shape[1] // 2
    
    transform = rewiring.FoSRRewiring(max_iters=5, remove_edges=False)
    new_data = transform(_clone_data(data))
    new_edges = new_data.edge_index.shape[1] // 2
    
    assert new_edges == original_edges, "FoSR mutated a fully connected graph"

def test_fosr_protects_bridges(torch_mod, tg_data):
    """
    Custom graph: Triangle with a tail (Kite/Paw graph).
    Edges: (0,1), (1,2), (2,0) [Triangle] and (0,3) [Tail].
    Verifies that the connected components check strictly prevents removing 
    the bridge (0,3), which would disconnect the graph.
    """
    ei = torch_mod.tensor([
        [0, 1, 1, 2, 2, 0, 0, 3],
        [1, 0, 2, 1, 0, 2, 3, 0]
    ], dtype=torch_mod.long)
    data = tg_data.Data(num_nodes=4, edge_index=ei)
    
    # max_iters=5 to give it ample opportunity to remove edges
    transform = rewiring.FoSRRewiring(max_iters=5, remove_edges=True)
    new_data = transform(_clone_data(data))
    
    edges = set()
    for i in range(new_data.edge_index.shape[1]):
        u = new_data.edge_index[0, i].item()
        v = new_data.edge_index[1, i].item()
        edges.add(tuple(sorted((u, v))))
        
    assert tuple(sorted((0, 3))) in edges, "FoSR illegally removed a bridge and disconnected the graph"

def test_fosr_preserves_edge_attr(torch_mod, tg_data):
    """
    Verifies that FoSR preserves multidimensional PyG edge attributes for 
    existing edges, and initializes strictly zeroed attributes for newly added edges.
    """
    ei = torch_mod.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch_mod.long)
    
    # 2 undirected edges (4 directed), arbitrary 2D features
    attr = torch_mod.tensor([
        [0.5, -0.1], 
        [0.5, -0.1], 
        [1.0, 2.2], 
        [1.0, 2.2]
    ], dtype=torch_mod.float)
    
    data = tg_data.Data(num_nodes=3, edge_index=ei, edge_attr=attr)
    transform = rewiring.FoSRRewiring(max_iters=1, remove_edges=False)
    
    # Use PyG's native clone() to ensure edge_attr is passed to the transform
    new_data = transform(data.clone())
    
    assert new_data.edge_attr is not None
    assert new_data.edge_attr.shape[0] == 6, "Expected 3 undirected edges (6 directed edges) after P3 -> Triangle rewiring"
    assert new_data.edge_attr.shape[1] == 2, "Edge attribute dimensionality mismatch"
    
    # Identify the newly added structural edges (0,2) and (2,0)
    edges = [(new_data.edge_index[0, i].item(), new_data.edge_index[1, i].item()) for i in range(6)]
    new_idx1 = edges.index((0, 2))
    new_idx2 = edges.index((2, 0))
    
    assert torch_mod.equal(new_data.edge_attr[new_idx1], torch_mod.zeros(2)), "Added edge attributes must be zero-initialized"
    assert torch_mod.equal(new_data.edge_attr[new_idx2], torch_mod.zeros(2)), "Added edge attributes must be zero-initialized"
