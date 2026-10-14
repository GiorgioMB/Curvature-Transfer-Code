# Curvature Transfer

Code for *What Balanced Forman Curvature Certifies About Ollivier–Ricci Curvature* (Micaletto & Nigrelli, [arXiv:2603.13535](https://arxiv.org/abs/2603.13535)).

The package computes, for every edge of an undirected simple graph:

* **exact curvatures**: lazy Ollivier–Ricci (idleness 1/(deg+1)), non-lazy OR, and Lin–Lu–Yau, by optimal transport;
* **Balanced Forman curvature** and the 2-hop statistics behind it;
* **every certified bound in the paper**: Jost–Liu (JL) and Tian–Lubberts–Weber (TLW), the matching-augmented bound (Thm 4.8), the lazy transport envelope (Prop 4.6), the BF interval (Thms 5.3–5.4, Cor 5.6), the BF deficits (Prop 5.1), the sharpened interval (Prop 5.7), and the reverse direction (Cors 6.1–6.2);
* **certify-then-solve** screening, which answers threshold and bottom-k queries exactly while solving transport problems only on undecided edges.

```python
from curvature_transfer import CurvatureEngine
eng = CurvatureEngine.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a PyG Data / networkx graph
eng.bf()                          # Balanced Forman curvature, aligned with eng.edges (u < v)
eng.exact()["c_OR"]               # exact lazy OR curvature (also "c_OR0", "c_LLY")
iv = eng.intervals()              # BF, matching, and sharpened intervals for c_OR
eng.edge_table()                  # everything as a pandas DataFrame
```

## Installation

```bash
pip install -e .                    # core
pip install -e ".[experiments,dev]" # experiments + tests
pip install -e ".[gnn]"             # rewiring / GNN case study (install torch for your platform first)
pytest -q
```

## Layout

| Path                                                                                       | Contents                                                                        |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `curvature_transfer/graph.py`                                                            | CSR graph, canonical undirected edges, PyG/networkx/edge-list constructors      |
| `curvature_transfer/_kernels.py`                                                         | Numba kernels: 2-hop statistics, Hopcroft–Karp, local distances, exact W1      |
| `curvature_transfer/bounds.py`                                                           | all bounds, vectorized over edges, with equation/result references              |
| `curvature_transfer/engine.py`                                                           | `CurvatureEngine`: caching, threading, intervals, edge tables                 |
| `curvature_transfer/screening.py`                                                        | certify-then-solve                                                              |
| `curvature_transfer/rewiring.py`                                                         | SDRF / BORF / FoSR on edge arrays (no torch), with pluggable oracles            |
| `curvature_transfer/pyg.py`                                                              | PyG`BaseTransform` wrappers used by the GNN experiments                       |
| `experiments/compare_bounds.py`                                                          | Tables 3, 4, 5, the ranking table, and the pending quantities of the manuscript |
| `experiments/screening.py`                                                               | certify-then-solve experiment (Section 7.3)                                     |
| `experiments/benchmark.py`                                                               | runtime scaling (Section 7.4)                                                   |
| `experiments/run_experiments.py`                                                         | orchestrator: per-edge tables, manifest, figures, and the phases above          |
| `experiments/stats.py`                                                                   | paired non-inferiority tests, bootstrap CIs, Holm correction (Appendix A)       |
| `experiments/gnn_experiments.py`, `tree_neighbors_match.py`, `make_paper_figures.py` | experiments (and required files) for Appendix A                                     |
| `experiments/generators.py`                                                              | graph generators, unchanged so that seeded graphs reproduce                     |

## Reproducing the paper

```bash
pip install -e ".[experiments,dev]"
python main.py                       # every result except the GNN case study, then compiles the paper
python main.py --gnn --delta 0.01    # + Appendix A (needs the [gnn] extra; the margin must be fixed in advance)
python main.py --quick               # smoke run of the whole pipeline (~2 minutes; not the paper's numbers)
```

**Cost.** The full run is dominated by Figure 2, which matches the configuration of the paper: 100 ER graphs up to
n = 15000, with exact OR on every edge (about 7×10^5 edges at the largest size). It also computes exact curvatures on
every edge of every graph family, including Arxiv (4.2×10^5 edges). Expect a few hours on a workstation; use
`--jobs -1` on a many-core machine.

## Exact Ollivier–Ricci curvature

For an edge (i, j), every measure lives on $S=N(i)\cup N(j)$. All distances on $S$ lie in {0, 1, 2, 3} and are read off
the adjacency structure (edges, then shared neighbours), so no BFS is needed. The Kantorovich dual has an integral
optimal potential with range at most 3. Writing $\phi = \sum_k [\phi\ge k]$ turns it into a maximum-weight closure problem,
solved by one s–t min cut (Dinic) in exact integer arithmetic. Lipschitz constraints implied by transitivity through
i or j are dropped, so the cut graph stays sparse. Twin vertices, such as the leaves of a hub, are merged exactly.

The solver is verified against linear programming in the tests. With warm JIT on one thread, it is about 34× faster
than the preliminary transportation simplex on ER graphs with n = 100 and about 170× faster with n = 1200, and the
gap keeps widening with size.

## Notes on the experiments

* In `screening.py`, "sign" means the query $\{\mathfrak{c}_{\rm OR}<0\}$ , so edges with lower $\ge0$ count as decided, including
  zero-curvature edges. `compare_bounds.py` reports the paper's strict sign certification (lower > 0 or upper < 0).
* The TLW dual bound needs the local distance matrix, so it is computed inside the exact kernel. The matching and
  sharpened intervals in `compare_bounds.py` include it, as in the paper. `screening.py` uses only certificates that
  need no distances, so its fractions are slightly lower than the certified-sign column.
* Certify-then-solve saves wall-clock time mainly on large or dense graphs. On small graphs, exact evaluation is now
  cheap enough that computing the statistics costs about as much as it saves.
