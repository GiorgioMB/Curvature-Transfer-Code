import numpy as np
import pytest
from conftest import EPS

def test_lazy_transport_envelope_upper_bound(engine):
    base = engine.compute_all()
    c_or = base["c_OR"]
    for eidx in range(len(base["edges"])):
        env = engine.lazy_transport_envelope(eidx)
        assert c_or[eidx] <= env["cOR_upper"] + EPS, "Lazy transport envelope must upper-bound c_OR"


def test_theta_alpha_monotone_envelope(engine):
    base = engine.compute_all()
    c_or = base["c_OR"]
    tri = base["triangle"]
    cons = base["Theta_Const"]
    slope = base["Theta_Slope"]

    # Slope must be strictly positive and envelope must upper-bound c_OR at t = triangle(i,j)
    assert np.all(slope > 0), "Proposition claims Theta_alpha has strictly positive slope"
    theta_at_t = cons + slope * tri
    assert np.all(c_or <= theta_at_t + EPS), "c_OR must be below Theta_alpha(triangle) envelope"

    # Monotonicity: Theta_alpha(t+1) > Theta_alpha(t) for all admissible t
    edges = base["edges"]
    for eidx, (i, j) in enumerate(edges):
        # admissible t in [0, min(deg_i,deg_j)-1]
        deg_i = int(base["deg_i"][eidx])
        deg_j = int(base["deg_j"][eidx])
        tmax = min(deg_i, deg_j) - 1
        C, S = float(cons[eidx]), float(slope[eidx])
        if tmax >= 1:
            t0 = 0
            t1 = min(1, tmax)
            assert C + S * t1 > C + S * t0, "Theta_alpha must be strictly increasing in t"
