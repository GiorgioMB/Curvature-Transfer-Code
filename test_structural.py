import numpy as np
import pytest
from conftest import EPS

def test_structural_Xi_bounds(engine):
    base = engine.compute_all()
    Xi = base["Xi"]
    deg_i = base["deg_i"]
    deg_j = base["deg_j"]
    tri = base["triangle"]
    sho = base["sho_max"]
    # Corollary: Xi <= deg_i + deg_j - 2 - 2*triangle
    assert np.all(Xi <= deg_i + deg_j - 2 - 2*tri + 1e-12)
    # Lemma/Box bound: Xi <= sho_max
    assert np.all(Xi <= sho + 1e-12)


def test_coverage_monotonicity_example(engines_dict):
    # Compare path4 interior edge (deg 2-2, tri=0, Xi=0) vs cycle4 edge (deg 2-2, tri=0, Xi>0)
    eng_path = engines_dict["path4"]
    eng_cyc = engines_dict["cycle4"]

    base_path = eng_path.compute_all()
    base_cyc = eng_cyc.compute_all()

    # find a 2-2 edge with tri=0 in path4: the middle edge (1,2)
    idx_path = None
    for eidx, (u,v) in enumerate(base_path["edges"]):
        if base_path["deg_i"][eidx]==2 and base_path["deg_j"][eidx]==2 and base_path["triangle"][eidx]==0:
            idx_path = eidx
            break
    assert idx_path is not None, "Did not find interior path edge"

    # find any 2-2, tri=0 edge in cycle4: all qualify
    idx_cyc = 0

    env_path = eng_path.lazy_transport_envelope(idx_path)
    env_cyc = eng_cyc.lazy_transport_envelope(idx_cyc)

    # Xi in cycle should exceed Xi in path
    Xi_path = base_path["Xi"][idx_path]
    Xi_cyc = base_cyc["Xi"][idx_cyc]
    assert Xi_cyc >= Xi_path, "4-cycle coverage Xi must be at least as large in the cycle"
    # The lazy transport envelope should be (weakly) larger when coverage is larger
    assert env_cyc["cOR_upper"] >= env_path["cOR_upper"] - EPS, "Envelope should not shrink as coverage grows"