"""Guarantees the experiment pipeline relies on: one interval definition,
resumable computations that give the same result as uninterrupted ones, and a
graph registry that agrees with the file names used by the manuscript."""
import os
import random
import sys

import networkx as nx
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from conftest import random_graphs  # noqa: E402
from curvature_transfer import CurvatureEngine, Graph, _io, intervals, rewiring  # noqa: E402
from curvature_transfer import _kernels as K  # noqa: E402


def test_interval_compute_equals_from_arrays():
    for G in random_graphs(2):
        g = Graph.from_edges(np.array(G.edges()), G.number_of_nodes())
        eng = CurvatureEngine(g)
        stored = intervals.from_arrays(eng.bounds(), eng.prior_upper())
        ex = eng.exact(("c_OR",))
        prior_from_exact = np.minimum(ex["tlw_dual_lazy"], eng.bounds()["zero_cost_upper"])
        assert np.allclose(prior_from_exact, eng.prior_upper(), atol=1e-12, rtol=0)
        for kind in intervals.KINDS:
            lo, up = intervals.compute(g, kind)
            assert np.allclose(lo, stored[kind][0], atol=1e-12, rtol=0)
            assert np.allclose(up, stored[kind][1], atol=1e-12, rtol=0)
            c = ex["c_OR"]
            assert np.all(lo <= c + 1e-9) and np.all(c <= up + 1e-9)


def test_chunk_store_resumes(tmp_path):
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(120, 3, seed=1))
    g = Graph.from_edges(np.array(G.edges()), G.number_of_nodes())
    ids = np.arange(g.num_edges, dtype=np.int64)
    f = lambda sl: K.exact_ids(g.indptr, g.indices, g.edges, np.ascontiguousarray(sl), True, True, True, 4)
    full = f(ids)
    store = _io.ChunkStore(str(tmp_path / "chunks"), 37)
    calls = {"n": 0}

    def flaky(sl):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt  # a "kill" after three chunks were saved
        return f(sl)

    with pytest.raises(KeyboardInterrupt):
        store.run(ids, flaky)
    saved = len(os.listdir(tmp_path / "chunks"))
    assert saved == 3
    recomputed = {"n": 0}

    def counting(sl):
        recomputed["n"] += 1
        return f(sl)

    out = store.run(ids, counting)
    assert recomputed["n"] == -(-g.num_edges // 37) - saved
    assert np.array_equal(out, full, equal_nan=True)


class _Kill(Exception):
    pass


@pytest.mark.parametrize("algo", ["sdrf", "borf", "fosr"])
def test_rewiring_resume_equals_uninterrupted(tmp_path, monkeypatch, algo):
    G = nx.convert_node_labels_to_integers(nx.barabasi_albert_graph(70, 2, seed=4))
    E = np.array(G.edges())

    def call(checkpoint=None, info=None):
        if algo == "sdrf":
            return rewiring.sdrf(70, E, "bounds", max_iters=6, rng=random.Random(3), checkpoint=checkpoint, info=info)
        if algo == "borf":
            return rewiring.borf(70, E, "bounds", max_iters=4, checkpoint=checkpoint, info=info)
        return rewiring.fosr(70, E, max_iters=5, checkpoint=checkpoint, info=info)

    ref = call()
    ck = str(tmp_path / "state.pkl")
    orig = rewiring._IterCheckpoint.save
    n = {"k": 0}

    def save_then_kill(self, *a, **kw):
        orig(self, *a, **kw)
        n["k"] += 1
        if n["k"] == 2:
            raise _Kill

    monkeypatch.setattr(rewiring._IterCheckpoint, "save", save_then_kill)
    with pytest.raises(_Kill):
        call(checkpoint=ck)
    monkeypatch.setattr(rewiring._IterCheckpoint, "save", orig)
    info = {}  # for SDRF the RNG state comes from the checkpoint, not from the fresh Random(3)
    resumed = call(checkpoint=ck, info=info)
    assert np.array_equal(resumed, ref)
    assert info["compute_time"] >= 0


def test_registry_matches_manuscript_file_names():
    import suite
    import main
    tags = {s.file_tag for s in suite.REGISTRY}
    for f in main.FIGURES:
        t = f.split("__", 1)[1].replace("_2d_Hist.png", "").replace(".png", "")
        assert t in tags, f"figure {f} refers to a graph that is not in the registry"
    assert all(q in suite.BY_NAME for q in suite.QUICK)
    assert {s.group for s in suite.REGISTRY} <= set(suite.GROUPS)
    assert all(name in suite.BY_NAME for name in main.FIG_GRAPHS.values())


def test_number_formatting():
    import compare_bounds as cb
    assert cb.fnum(-1e-19) == "0.000" and cb.fnum(-0.0004) == "0.000" and cb.fnum(-0.0006) == "-0.001"
    assert cb.fnum(float("nan")) == "--" and cb.fpct(float("nan")) == "--"
    assert cb._bold(["0.100", "--", "0.050"], "min") == ["0.100", "--", "\\textbf{0.050}"]


def test_atomic_writes(tmp_path):
    p = str(tmp_path / "a.json")
    _io.atomic_json({"x": np.float64(1.5), "y": np.arange(3)}, p)
    assert _io.read_json(p) == {"x": 1.5, "y": [0, 1, 2]}
    q = str(tmp_path / "b.npz")
    _io.atomic_npz(q, a=np.arange(4), fp=np.array("abc"))
    z = _io.read_npz(q)
    assert np.array_equal(z["a"], np.arange(4)) and str(z["fp"]) == "abc"
    assert not [f for f in os.listdir(tmp_path) if ".tmp." in f]
    assert _io.read_json(str(tmp_path / "missing.json"), 7) == 7
