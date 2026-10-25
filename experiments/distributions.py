#!/usr/bin/env python3
"""Figures 2-4 (Section 6.2) from the shared per-edge arrays.

Writes, for every graph of the registry, the per-edge CSV and manifest that
``make_paper_figures.py`` reads (``<results>/edges/csv/``), computed from the
same stored arrays as the tables (``edges.py``), and then renders the figures
into ``<results>/edges/csv/figures_paper/``. The figure file names use the
registry tags, e.g. ``fig_cOR_hist__er_n800_p0_0100125.png``.

Usage: python distributions.py --results ../results [--quick] [--jobs N]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--bins", type=int, default=60)
    a = ap.parse_args()
    cmd = [sys.executable, os.path.join(HERE, "edges.py"), "--out", a.results, "--csv",
           "--seed", str(a.seed), "--chunk", str(a.chunk)]
    cmd += ["--quick"] if a.quick else []
    cmd += ["--only", a.only] if a.only else []
    cmd += ["--jobs", str(a.jobs)] if a.jobs is not None else []
    print("[distributions]", " ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        raise SystemExit("edges.py failed")
    sys.path.insert(0, HERE)
    from make_paper_figures import generate_paper_figures
    root = os.path.join(os.path.abspath(a.results), "edges")
    generate_paper_figures(out_root=root, run_name="csv", bins=a.bins)
    print(f"[distributions] figures in {os.path.join(root, 'csv', 'figures_paper')}", flush=True)


if __name__ == "__main__":
    main()
