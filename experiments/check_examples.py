#!/usr/bin/env python3
"""Check every closed-form value stated in the paper against the code.

Remark 3.5 (Q_4), Remark 4.5, the paragraph after Theorem 5.3 (K_n, Q_n),
Remark 5.2 (BF deficits on Q_n), Remark 5.5 (strict gain on K_n, Q_n), and the
exactness claims of Theorem 4.8 (trees, hypercubes, grid). Exits non-zero if
any check fails, and writes the checked values to --out.
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from curvature_transfer import CurvatureEngine  # noqa: E402


def eng(G):
    return CurvatureEngine(nx.convert_node_labels_to_integers(G)).compute_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines, ok = [], True

    def check(name, got, want, tol=1e-12):
        nonlocal ok
        good = bool(np.all(np.abs(np.asarray(got, float) - want) <= tol))
        ok &= good
        lines.append(f"[{'ok' if good else 'FAIL'}] {name}: computed {np.unique(np.round(got, 12))}, stated {want}")

    r = eng(nx.hypercube_graph(4))
    check("Remark 3.5: Q_4 c_BF", r["c_BF"], 0.5); check("Remark 3.5: Q_4 c_LLY", r["c_LLY"], 0.5)
    check("Remark 3.5: Q_4 c_OR", r["c_OR"], 0.4); check("Remark 3.5: Q_4 c_OR-0", r["c_OR0"], 0.0)
    for n in (3, 4, 5, 6):
        r = eng(nx.hypercube_graph(n))
        check(f"Q_{n}: c_BF = c_LLY = 2/n", r["c_BF"], 2 / n); check(f"Q_{n}: c_LLY = 2/n", r["c_LLY"], 2 / n)
        check(f"Q_{n}: lower modulus exact", r["bf_lower"] - r["c_OR"], 0.0)
        check(f"Remark 5.2: Q_{n} deficits A = B = -1/n", r["bf_deficit_A"], -1 / n)
        check(f"Remark 5.2: Q_{n} c_BF - c_OR-0 = 2/n", r["c_BF"] - r["c_OR0"], 2 / n)
        check(f"Thm 4.8 exact on Q_{n}", r["matching_lower"] - r["c_OR0"], 0.0)
    for n in (4, 6, 8, 12):
        r = eng(nx.complete_graph(n))
        check(f"K_{n}: c_BF = n/(n-1)", r["c_BF"], n / (n - 1)); check(f"K_{n}: c_LLY = n/(n-1)", r["c_LLY"], n / (n - 1))
        check(f"K_{n}: lower modulus exact", r["bf_lower"] - r["c_OR"], 0.0)
        check(f"Remark 5.5: K_{n} has B^BF < 0", (r["bf_deficit_B"] < 0).astype(float), 1.0)
    for G, name in ((nx.balanced_tree(3, 3), "tree"), (nx.grid_2d_graph(8, 8), "grid")):
        r = eng(G)
        check(f"Thm 4.8 exact on the {name}", r["matching_lower"] - r["c_OR0"], 0.0, 1e-9)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
