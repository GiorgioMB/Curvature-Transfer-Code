import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import stats  # noqa: E402


def _entry(rng, mean, reps=10, K=5, shuffle=False):
    runs = [{"rep": r, "seed": 1000 + r, "score": 0.0,
             "folds": list(mean + rng.normal(0, 0.01, K) + 0.02 * np.sin(r + np.arange(K)))} for r in range(reps)]
    for x in runs:
        x["score"] = float(np.mean(x["folds"]))
    if shuffle:
        rng.shuffle(runs)
    return {"runs": runs}


def test_pairing_is_by_repetition_not_position():
    rng = np.random.default_rng(0)
    a = _entry(rng, 0.8, shuffle=True)
    b = {"runs": [dict(r, folds=[f - 0.001 for f in r["folds"]]) for r in sorted(a["runs"], key=lambda r: r["rep"])]}
    d, K, r = stats.fold_differences(a, b)
    assert np.allclose(d, 0.001) and K == 5 and r == 10


def test_refuses_unpairable_results():
    with pytest.raises(ValueError):
        stats.fold_differences({"scores": [0.1, 0.2]}, {"scores": [0.1, 0.2]})


def test_corrected_is_more_conservative_than_paired():
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 0.02, 50)
    p_c, _, lcb_c = stats.noninferiority(d, 5, 0.01, test="corrected")
    p_p, _, lcb_p = stats.noninferiority(d, 5, 0.01, test="paired")
    assert p_c >= p_p and lcb_c <= lcb_p


def test_decision_matches_confidence_bound():
    rng = np.random.default_rng(2)
    for _ in range(200):
        d = rng.normal(rng.uniform(-0.02, 0.02), 0.02, 40)
        p, m, lcb = stats.noninferiority(d, 5, 0.01, alpha=0.05)
        assert (p < 0.05) == (lcb > -0.01)


def test_holm():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_cli(tmp_path):
    rng = np.random.default_rng(3)
    res = {"ZINC_GCN": {"BORF (OR)": _entry(rng, 0.5), "BORF (Bounds)": _entry(rng, 0.5)}}
    path = tmp_path / "gnn_results.json"
    path.write_text(json.dumps(res))
    out = subprocess.run([sys.executable, stats.__file__, str(path), "--delta", "0.05"],
                         capture_output=True, text=True, check=True).stdout
    assert "BORF (Bounds)" in out and "p_holm" in out
