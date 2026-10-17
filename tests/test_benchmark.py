"""The runtime benchmark: generators, chunked computation, resume, and skip guards."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import benchmark as BM  # noqa: E402
import timing  # noqa: E402
from curvature_transfer import Graph, intervals  # noqa: E402


def _simple(E, n):
    E = np.asarray(E)
    assert np.all(E[:, 0] != E[:, 1]) and np.all((E >= 0) & (E < n))
    key = np.minimum(E[:, 0], E[:, 1]) * n + np.maximum(E[:, 0], E[:, 1])
    assert len(np.unique(key)) == len(key)


@pytest.mark.parametrize("family", BM.FAMILIES)
def test_generators_are_simple_deterministic_and_follow_the_law(family):
    n = 3000
    p1, E1 = BM.make_graph(family, n, 7)
    p2, E2 = BM.make_graph(family, n, 7)
    assert p1 == p2 and np.array_equal(E1, E2)
    _simple(E1, n)
    d = 2 * len(E1) / n
    tol = {"ER": 0.1, "BA": 0.05, "WS": 0.05, "RGG": 0.2}[family]  # RGG: boundary effect of the unit square
    assert abs(d - BM.target_degree(n)) / BM.target_degree(n) < tol + (0.2 if family == "BA" else 0)


def test_er_pair_decoding_is_exact():
    rng = np.random.default_rng(0)
    E = BM.gen_er(40, 1.0, rng)  # p = 1: every pair exactly once
    assert len(E) == 40 * 39 // 2
    _simple(E, 40)
    assert np.all(E[:, 0] < E[:, 1])


def test_chunked_intervals_equal_whole():
    _, E = BM.make_graph("BA", 400, 1)
    g = Graph.from_edges(E, 400)
    for kind in intervals.KINDS:
        lo, up = intervals.compute(g, kind)
        parts = [intervals.compute(g, kind, ids=np.arange(k, min(g.num_edges, k + 97))) for k in range(0, g.num_edges, 97)]
        assert np.array_equal(np.concatenate([p[0] for p in parts]), lo)
        assert np.array_equal(np.concatenate([p[1] for p in parts]), up)


def _args(**kw):
    a = dict(chunk_above=100, chunk=150, budget_hours=12.0, no_mem_guard=False, max_attempts=2, min_total=0.0,
             max_reps=1, legacy_dir=None, legacy_sample=10)
    a.update(kw)
    return SimpleNamespace(**a)


class _Kill(Exception):
    pass


def test_resume_after_a_kill_inside_a_chunked_phase(tmp_path, monkeypatch):
    timing.warmup()
    a = _args()
    orig = timing.PHASES["exact"]
    calls = {"n": 0}

    def flaky(g, ids=None, n_jobs=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise _Kill  # killed during the third chunk
        return orig[0](g, ids, n_jobs)

    monkeypatch.setitem(timing.PHASES, "exact", (flaky, True))
    with pytest.raises(_Kill):
        BM.run_size("ER", 300, 0, None, 1, str(tmp_path), [], a, 64.0)
    part = BM._io.read_json(str(tmp_path / "n300.partial.json"))
    assert len(part["chunks"]["exact"]) == 2 and part["status"]["bf"] == "measured"
    seen = []
    monkeypatch.setitem(timing.PHASES, "exact", (lambda g, ids=None, n_jobs=None: seen.append(ids[0]) or orig[0](g, ids, n_jobs), True))
    row = BM.run_size("ER", 300, 0, None, 1, str(tmp_path), [], a, 64.0)
    assert row["exact_status"] == "measured" and row["chunked"]
    assert seen[0] == 2 * a.chunk  # resumed at the first chunk not yet timed
    assert row["exact_s"] > 0


def test_a_chunk_killed_twice_is_given_up(tmp_path, monkeypatch):
    timing.warmup()
    a = _args()

    def always_killed(g, ids=None, n_jobs=None):
        raise _Kill

    monkeypatch.setitem(timing.PHASES, "exact", (always_killed, True))
    for _ in range(2):
        with pytest.raises(_Kill):
            BM.run_size("ER", 300, 0, None, 1, str(tmp_path), [], a, 64.0)
    row = BM.run_size("ER", 300, 0, None, 1, str(tmp_path), [], a, 64.0)
    assert row["exact_status"].startswith("skipped (killed 2 times in chunk 0")
    assert row["sharpened_interval_status"] == "measured"


def test_budget_and_memory_guards_and_monotone_skips(tmp_path):
    timing.warmup()
    prev = [{"n": 100, "exact_s": 1.0, "exact_status": "measured"}, {"n": 200, "exact_s": 4.0, "exact_status": "measured"}]
    assert BM.predict(prev, "exact", 400) == pytest.approx(16.0)
    row = BM.run_size("ER", 300, 0, None, 1, str(tmp_path), prev, _args(chunk_above=10 ** 9, budget_hours=1e-4), 64.0)
    assert row["exact_status"].startswith("skipped (predicted") and row["exact_pred_s"] == pytest.approx(9.0)
    row2 = BM.run_size("ER", 350, 0, None, 1, str(tmp_path), prev + [row], _args(chunk_above=10 ** 9), 64.0)
    assert row2["exact_status"].startswith("skipped (not run from n=300")
    row3 = BM.run_size("ER", 320, 0, None, 64, str(tmp_path), [], _args(chunk_above=10 ** 9), 1e-9)
    assert row3["exact_status"].startswith("skipped (memory") and row3["bf_status"] == "measured"
