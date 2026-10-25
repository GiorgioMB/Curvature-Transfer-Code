# Curvature Transfer

Code for *What Balanced Forman Curvature Certifies About Ollivier–Ricci Curvature* (Micaletto & Nigrelli, [arXiv:2603.13535](https://arxiv.org/abs/2603.13535)). 
The package computes, for every edge of an undirected simple graph:

* **exact curvatures**: lazy Ollivier–Ricci (idleness 1/(deg+1)), non-lazy OR, and Lin–Lu–Yau, by optimal transport;
* **Balanced Forman curvature** and the 2-hop statistics behind it;
* **every certified bound in the paper**: Jost–Liu (JL) and Tian–Lubberts–Weber (TLW), the matching-augmented bound (Thm 4.8), the lazy transport envelope (Prop 4.6), the BF interval (Thms 5.3–5.4, Cor 5.6), the BF deficits (Prop 5.1), the sharpened interval (Prop 5.7), and the reverse direction (Appendix C, Cors C.2–C.3);
* **certify-then-solve** screening, which answers threshold and bottom-k queries exactly while solving transport problems only on undecided edges: a cascade of 2-hop bounds on every edge, then the TLW certificate and the transport problem (sharing one distance matrix) only where needed, with an adaptive schedule for bottom-k queries (`curvature_transfer.screen`).

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

| Path                                                                                                | Contents                                                                                 |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `curvature_transfer/graph.py`                                                                     | CSR graph, canonical undirected edges, PyG/networkx/edge-list constructors               |
| `curvature_transfer/_kernels.py`                                                                  | Numba kernels: 2-hop statistics, Hopcroft–Karp, local distances, exact W1               |
| `curvature_transfer/bounds.py`                                                                    | all bounds, vectorized over edges, with equation/result references                       |
| `curvature_transfer/intervals.py`                                                                 | definition of the BF, matching, and sharpened intervals (used by every experiment)       |
| `curvature_transfer/engine.py`                                                                    | `CurvatureEngine`: caching, threading, intervals, edge tables                          |
| `curvature_transfer/screening.py`                                                                 | certify-then-solve                                                                       |
| `curvature_transfer/rewiring.py`                                                                  | SDRF / BORF / FoSR on edge arrays (no torch), resumable per iteration                    |
| `curvature_transfer/pyg.py`                                                                       | PyG `BaseTransform` wrappers used by the GNN experiments                               |
| `curvature_transfer/_io.py`                                                                       | atomic writes and chunked, resumable computations                                        |
| `experiments/suite.py`                                                                            | graph registry: every graph of every table and figure                                    |
| `experiments/edges.py`                                                                            | per-edge statistics, bounds, exact curvatures of every registry graph (the single store) |
| `experiments/compare_bounds.py`                                                                   | Tables 2–5 from the per-edge store                                                      |
| `experiments/screening.py` | Section 6.3: avoided fractions on the registry graphs (from the per-edge store) and Table 6, timed on large real networks |
| `experiments/large_graphs.py` | the large real networks of Table 6 (10^5 to 10^6 edges): sources, download cache, parsing |
| `experiments/distributions.py`                                                                    | Figures 2–4 from the per-edge store (`make_paper_figures.py` draws them)              |
| `experiments/timing.py`                                                                           | timing code shared by the screening table and the runtime figure                         |
| `experiments/benchmark.py`                                                                        | runtime scaling, Figure 5 (Section 6.4)                                                  |
| `experiments/gnn_experiments.py`, `merge_results.py`, `stats.py`, `tree_neighbors_match.py` | Appendix A                                                                               |
| `experiments/generators.py`                                                                       | graph generators, unchanged so that seeded graphs reproduce                              |

## Reproducing the paper

```bash
pip install -e ".[experiments,dev]"
python main.py                       # every result except the GNN case study, then compiles the paper
python main.py --gnn --delta 0.01    # + Appendix A (needs the [gnn] extra; the margin must be fixed in advance)
python main.py --quick               # smoke run of the whole pipeline (minutes; not the paper's numbers)
python main.py --list                # the steps and what each one produces
```

**One source for every number.** All graphs come from `experiments/suite.py` (restricted to their largest
connected component, never subsampled); all per-edge quantities are computed once by `edges.py` and stored in
`results/edges/`; all intervals come from `curvature_transfer/intervals.py`; the screening table and the runtime
figure use the same timing code (`experiments/timing.py`). Tables, text numbers, and figures are derived from
these, so the same graph gives the same numbers everywhere.

Aggregate statements in the text (ranges, medians, "on k of the N graphs") are over the random graphs and the
real networks (`suite.AGGREGATE_GROUPS`). The canonical graphs (cycle, grid, tori, tree, complete graph) are
listed in the tables but excluded from aggregates, because several bounds are exact on them by construction.

**Checkpointing.** Everything is saved as soon as it is computed, with atomic writes (temporary file, then
rename), so a killed job loses at most the unit of work in progress:

Every checkpoint name contains a fingerprint of its inputs, so a stale partial result from a different
graph or configuration is never reused. Finished steps are skipped through stamps in `results/.stamps`;
`--force` ignores the stamps but still reuses the per-unit checkpoints. To recompute from scratch, delete
`results/`.

**Runtime benchmark.** Four families (ER, BA, RGG, WS; `--bench-families`) with the scaling law of the ER
example, average degree ~ 2.1 n^0.4, one figure per family plus a 2×2 panel (`runtime_scaling_<family>.pdf`,
`runtime_scaling_all.pdf`; `runtime_scaling.pdf` is ER, as in the manuscript). `--bench-max-n` sets the largest
graph (default 15000). Above 2×10^6 edges a phase is timed as the sum of chunk times (marked in the figure).
A phase that is predicted to exceed `--bench-budget-hours` (default 12, per size), or the memory of the
allocation, or whose chunk was killed twice, is not run from that size on; its extrapolated time is drawn
as "predicted, not run" and the reason is in the CSV. Exact OR (and the TLW certificate) build dense
|S|×|S| arrays per edge, |S| <= deg(i) + deg(j), so on BA hubs they stop long before the BF interval does.
Families write to separate folders and can run as parallel jobs, e.g.
`python experiments/benchmark.py --family BA --max-n 200000 --jobs 32 --out results/benchmark`.

**Cost.** The full run is dominated by exact curvature on every edge of every graph (Arxiv: 4.2×10^5 edges;
K_120 is small but dense) and by the runtime benchmark. Expect a few hours on a workstation at the default
sizes; use `--jobs N` with the cores you have.

## Exact Ollivier–Ricci curvature

For an edge (i, j), every measure lives on $S=N(i)\cup N(j)$. All distances on $S$ lie in {0, 1, 2, 3} and are read off
the adjacency structure (edges, then shared neighbours), so no BFS is needed. The Kantorovich dual has an integral
optimal potential with range at most 3. Writing $\phi = \sum_k [\phi\ge k]$ turns it into a maximum-weight closure problem,
solved by one s–t min cut (Dinic) in exact integer arithmetic. Lipschitz constraints implied by transitivity through
$i$ or $j$ are dropped, so the cut graph stays sparse. Twin vertices, such as the leaves of a hub, are merged exactly.

The solver is verified against linear programming in the tests. With warm JIT on one thread, it is about 34× faster
than the preliminary transportation simplex on ER graphs with n = 100 and about 170× faster with n = 1200, and the
gap keeps widening with size.

## Notes on the experiments

* Certified decisions (the "sign certified" columns and certify-then-solve) use every interval widened by
  `intervals.ROUNDING_TOL` = 1e-9: the bounds are floating point, the exact curvatures exact rationals, and on
  graphs where lower and upper bound coincide (e.g. K_n) rounding can otherwise put lower > upper.
* "Sign certified" means lower > 0 or upper < 0; the screening query {c_OR < 0} also counts lower >= 0 as
  decided, so its avoided fraction is never below the certified-sign fraction.
* The matching and sharpened intervals of Section 6.1 include the TLW dual certificate, which needs the local
  distance matrix (but no transport solve). Screening and Figure 5 time exactly these intervals, computed from
  scratch.
* Certify-then-solve (Section 6.3) is timed on the large real networks of `experiments/large_graphs.py`, which are
  downloaded on first use (SNAP and GitHub; set `--data-dir` to reuse a cache) into `results/data/large/`. The
  baseline is exact evaluation of every edge with the same min-cut solver, without the TLW dual
  (`timing.exact_all`); both the baseline and the cascade process the edges in one fixed random order, so that
  the expensive edges at hubs are spread over the threads. Every answer is checked against the exact curvatures.
  The `paper` step fails its consistency check if some large graph shows no speedup, since the text says otherwise.
  On the small registry graphs exact evaluation takes milliseconds, so only their avoided fractions are reported.
* GNNs: GCN uses `GCNConv`, GIN `GINEConv`, GAT `GATConv`; every topology carries one edge feature (1 = added
  by rewiring), and TreeNeighborsMatch is read out at the root node. Every CV fold is seeded from
  (seed, fold), so restored and recomputed folds give the same results.
