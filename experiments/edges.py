#!/usr/bin/env python3
"""Per-edge arrays for every graph of the registry (the single source of truth).

For each graph of :data:`suite.REGISTRY` this computes, on *every* edge of
its largest component: the 2-hop statistics, BF, every combinatorial bound,
the exact curvatures c_OR, c_OR-0, c_LLY, and the TLW dual certificates, and
stores them in

    <out>/edges/<file_tag>.<varpi>.npz    statistics + bounds + exact values
    <out>/edges/<file_tag>.exact.npz      exact values only (shared by both varpi)

Every table (``compare_bounds.py``), the screening fractions
(``screening.py``) and every figure (``--csv``, read by
``make_paper_figures.py``) is computed from these files, so the same graph
yields the same numbers everywhere.

Checkpointing. The exact curvatures are computed in chunks of ``--chunk``
edges; each chunk is saved (atomically) as soon as it is done, under
``<out>/edges/partial/<file_tag>.<fingerprint>/``. A killed job resumes at
the first missing chunk. The fingerprint hashes the edge set and
``EDGE_VERSION``, so chunks of a different graph are never reused. A graph
whose final file exists with the right fingerprint is skipped.

Usage
  python edges.py --out ../results                 # every graph
  python edges.py --out ../results --quick         # the --quick subset
  python edges.py --out ../results --csv           # + per-edge CSVs and manifest for the figures
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from curvature_transfer import CurvatureEngine, Graph  # noqa: E402
from curvature_transfer import _io  # noqa: E402
from curvature_transfer import _kernels as K  # noqa: E402
from curvature_transfer.engine import _threads  # noqa: E402
import suite  # noqa: E402

# Bump when a formula in curvature_transfer changes, to invalidate stored files.
EDGE_VERSION = 2
EXACT_KEYS = ("c_OR", "c_OR0", "c_LLY", "tlw_dual_lazy", "tlw_dual_nonlazy")  # columns of K.exact_ids


def edges_dir(out):
    return os.path.join(os.path.abspath(out), "edges")


def graph_of(spec, seed):
    n, e = suite.load(spec, seed)
    return Graph.from_edges(e, n)


def graph_fingerprint(g: Graph) -> str:
    return _io.fingerprint(int(g.num_nodes), g.edges, EDGE_VERSION)


def _log(spec, msg):
    print(f"[edges] {spec.name}: {msg}", flush=True)


def exact_arrays(spec, g: Graph, out, n_jobs=None, chunk=10000):
    """Exact curvatures + TLW duals on every edge, chunked and resumable."""
    d = edges_dir(out)
    fp = graph_fingerprint(g)
    final = os.path.join(d, f"{spec.file_tag}.exact.npz")
    z = _io.read_npz(final)
    if z is not None and str(z.get("fingerprint")) == fp:
        return {k: z[k] for k in EXACT_KEYS}
    import numba

    store = _io.ChunkStore(os.path.join(d, "partial", f"{spec.file_tag}.{fp}"), chunk)

    def f(ids):
        with _threads(n_jobs):
            return K.exact_ids(g.indptr, g.indices, g.edges, np.ascontiguousarray(ids, dtype=np.int64),
                               True, True, True, 8 * numba.get_num_threads())

    t0 = time.time()
    raw = store.run(np.arange(g.num_edges, dtype=np.int64), f, log=lambda m: _log(spec, m))
    raw = raw.reshape(g.num_edges, len(EXACT_KEYS))
    if np.isnan(raw[:, :3]).any():
        raise RuntimeError(f"{spec.name}: NaN in exact curvatures")
    arrays = {k: raw[:, c] for c, k in enumerate(EXACT_KEYS)}
    _io.atomic_npz(final, fingerprint=np.array(fp), seconds=np.array(time.time() - t0), **arrays)
    store.clear()
    return arrays


def compute(spec, out, seed=0, n_jobs=None, varpi="paper", chunk=10000, force=False):
    """All per-edge arrays of one graph (loaded if already stored)."""
    d = edges_dir(out)
    os.makedirs(d, exist_ok=True)
    g = graph_of(spec, seed)
    fp = graph_fingerprint(g)
    final = os.path.join(d, f"{spec.file_tag}.{varpi}.npz")
    if not force:
        z = _io.read_npz(final)
        if z is not None and str(z.get("fingerprint")) == fp:
            return z
    t0 = time.time()
    eng = CurvatureEngine(g, n_jobs=n_jobs, varpi=varpi)
    st = eng.stats()
    b = eng.bounds()
    ex = exact_arrays(spec, g, out, n_jobs, chunk)
    arrays = {"edges": g.edges, "num_nodes": np.array(g.num_nodes)}
    arrays.update({k: np.asarray(v) for k, v in st.items()})
    arrays.update({k: np.asarray(v) for k, v in b.items()})
    arrays.update(ex)
    arrays["prior_upper"] = np.minimum(ex["tlw_dual_lazy"], b["zero_cost_upper"])
    rev = eng.bounds_from_OR(ex["c_OR"])
    arrays["c_BF_lower_from_c_OR"] = rev["c_BF_lower_from_c_OR"]
    arrays["c_BF_upper_from_c_OR"] = rev["c_BF_upper_from_c_OR"]
    arrays["fingerprint"] = np.array(fp)
    _io.atomic_npz(final, **arrays)
    _log(spec, f"n={g.num_nodes} m={g.num_edges} stored ({time.time() - t0:.1f}s)")
    return _io.read_npz(final)


def write_csv(spec, arrays, csv_dir, seed=0, n_jobs=None, varpi="paper"):
    """Per-edge CSV in the format of :meth:`CurvatureEngine.edge_table`
    (read by make_paper_figures.py), from the stored arrays."""
    import pandas as pd  # noqa: F401

    g = Graph.from_edges(arrays["edges"], int(arrays["num_nodes"]))
    eng = CurvatureEngine(g, n_jobs=n_jobs, varpi=varpi)
    ex = {k: arrays[k] for k in EXACT_KEYS}
    df = eng.edge_table(exact=True, exact_values=ex)
    if not np.array_equal(df["c_BF"].to_numpy(), arrays["c_BF"]):
        raise RuntimeError(f"{spec.name}: BF recomputed for the CSV differs from the stored BF")
    path = os.path.join(csv_dir, f"{spec.file_tag}_edges.csv")
    tmp = _io.tmp_path(path)
    df.to_csv(tmp, index=False, float_format="%.12g")
    os.replace(tmp, path)
    return path, len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--only", default=None, help="semicolon-separated names or tags of suite graphs")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--varpi", choices=["paper", "legacy"], default="paper")
    ap.add_argument("--chunk", type=int, default=10000, help="edges per checkpointed chunk of exact evaluation")
    ap.add_argument("--csv", action="store_true", help="also write per-edge CSVs + manifest.json for the figures")
    a = ap.parse_args()
    specs = suite.specs(a.only.split(";") if a.only else None, quick=a.quick)
    csv_dir = os.path.join(edges_dir(a.out), "csv")
    if a.csv:
        os.makedirs(csv_dir, exist_ok=True)
    manifest_path = os.path.join(csv_dir, "manifest.json")
    manifest = _io.read_json(manifest_path, {"runs": []}) if a.csv else None
    for spec in specs:
        arr = compute(spec, a.out, a.seed, a.jobs, a.varpi, a.chunk)
        if a.csv:
            path = os.path.join(csv_dir, f"{spec.file_tag}_edges.csv")
            entry = next((r for r in manifest["runs"] if r["tag"] == spec.tag), None)
            if not (entry and entry.get("fingerprint") == str(arr["fingerprint"]) and os.path.exists(path)):
                _, m = write_csv(spec, arr, csv_dir, a.seed, a.jobs, a.varpi)
                manifest["runs"] = [r for r in manifest["runs"] if r["tag"] != spec.tag]
                manifest["runs"].append({"tag": spec.tag, "name": spec.name, "group": spec.group,
                                         "n": int(arr["num_nodes"]), "m": int(m),
                                         "fingerprint": str(arr["fingerprint"])})
                _io.atomic_json(manifest, manifest_path)
                _log(spec, f"CSV written ({m} edges)")
    partial = os.path.join(edges_dir(a.out), "partial")
    if os.path.isdir(partial) and not os.listdir(partial):
        shutil.rmtree(partial, ignore_errors=True)
    print(f"[edges] done: {len(specs)} graphs in {edges_dir(a.out)}", flush=True)


if __name__ == "__main__":
    main()
