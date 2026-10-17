#!/usr/bin/env python3
"""Merge the per-(dataset, arch) result files of gnn_experiments.py.

gnn_experiments.py writes one file per (dataset, architecture),
``<out>/parts/<dataset>_<arch>.json``, so that parallel jobs never overwrite
each other's results. ``merge(out)`` combines them into
``<out>/gnn_results.json`` (atomically), the file read by stats.py and by the
paper step. Running it again at any time is safe.

Usage: python merge_results.py <out_dir>
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from curvature_transfer import _io  # noqa: E402


def merge(out_dir):
    merged = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "parts", "*.json"))):
        data = _io.read_json(path)
        if isinstance(data, dict):
            merged[os.path.basename(path)[:-5]] = data
    _io.atomic_json(merged, os.path.join(out_dir, "gnn_results.json"), indent=1)
    return merged


if __name__ == "__main__":
    m = merge(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out"))
    print(f"merged {len(m)} (dataset, arch) result files: {sorted(m)}")
