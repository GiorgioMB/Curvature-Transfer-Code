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
