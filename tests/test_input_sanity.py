import numpy as np
import pytest
import torch
from conftest import Data
import pyg_curvature as pc

def test_undirected_unique_edge_extraction_and_degrees():
    # Duplicates, reversed duplicates, and a self-loop (should be ignored or trigger)
    # Build edge_index with duplicates
    rows = [0,1,1,0, 2,3, 3,2]
    cols = [1,0,1,0, 3,2, 2,3]
    ei = torch.tensor([rows, cols], dtype=torch.long)
    eng = pc.CurvatureEngine(Data(4, ei))
    # Undirected unique should be exactly {(0,1),(2,3)}
    assert set(map(tuple, eng.undirected_edges.T.tolist())) == {(0,1),(2,3)}
    # Degrees should be 1,1,1,1
    assert np.all(eng.deg == np.array([1,1,1,1]))

def test_n_jobs_validation_errors():
    import torch
    ei = torch.tensor([[0,1],[1,0]], dtype=torch.long)
    eng = pc.CurvatureEngine(Data(2, ei))
    with pytest.raises(ValueError):
        eng.compute_all(n_jobs=0)
    with pytest.raises(TypeError):
        eng._validate_n_jobs(2.5)  # must be int or None
