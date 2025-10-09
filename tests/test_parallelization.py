import os
import sys
import numpy as np
import pytest

@pytest.mark.slow
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Process-based parallelism is flaky on Windows CI")
def test_parallel_matches_sequential(make_square, new_engine):
    # Use a slightly larger graph by duplicating cycle edges to reach M >= 4; force n_jobs=2 anyway
    data = make_square()
    eng_seq = new_engine(data, n_jobs=1)
    base_seq = eng_seq.compute_all(n_jobs=1)

    eng_par = new_engine(data, n_jobs=2)
    base_par = eng_par.compute_all(n_jobs=2)

    keys = ["edges","c_BF","c_OR","c_OR0","C4","deg_i","deg_j","triangle"]
    for k in keys:
        assert np.allclose(base_seq[k], base_par[k]), f"Mismatch for key {k}"
