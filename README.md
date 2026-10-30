# Curvature Transfer

Code for the experiments of 

> Giorgio Micaletto and Tebe Nigrelli. *What Balanced Forman Curvature Certifies About Ollivier–Ricci Curvature.* 2026.


The package computes, for every edge of an undirected simple graph:

* **exact curvatures**: lazy Ollivier–Ricci (idleness 1/(deg+1)), non-lazy OR, and Lin–Lu–Yau, from the solver
  package [`orcut`](https://github.com/GiorgioMB/orcut), pinned to 1.0.0 (the solver behind every exact value of the paper);
* **Balanced Forman curvature** and the 2-hop statistics behind it;
* **every certified bound in the paper**: Jost–Liu (JL) and Tian–Lubberts–Weber (TLW), the mixing lift (Prop 4.1, Cor 4.2), the quadrangle-augmented JL bound (Thm 4.3), the lazy quadrangle bound (Thm 4.7), the envelope for arbitrary idleness (Prop 4.8), the lazy transport envelope (Prop 4.10), the BF deficits (Prop 5.1), the BF interval (Thms 5.4 and 5.7, Cor 5.8), the 2-hop interval (Prop 5.9), and the reverse direction (Appendix C, Cors C.1–C.2);
* **certify-then-solve** screening, which answers threshold queries exactly while solving transport problems only on undecided edges: a cascade of 2-hop bounds on every edge, then the TLW certificate (one more 2-hop pass) and the transport problem only where needed (`curvature_transfer.screen`).

```python
from curvature_transfer import CurvatureEngine
eng = CurvatureEngine.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph
eng.bf()                          # Balanced Forman curvature, aligned with eng.edges (u < v)
eng.exact()["c_OR"]               # exact lazy OR curvature (also "c_OR0", "c_LLY")
iv = eng.intervals()              # BF, matching, and sharpened intervals for c_OR
eng.edge_table()                  # everything as a pandas DataFrame
```


The Lean 4 formalization of the paper is in a separate repository, [BF-OR-Lean](https://github.com/GiorgioMB/BF-OR-Lean/).

## Installation

```bash
pip install -e .                      # core: numpy, scipy, numba, orcut
pip install -e ".[experiments,dev]"   # experiments + tests
pytest -q
```

The exact solver, [`orcut`](https://github.com/GiorgioMB/orcut), is not on PyPI: pip installs it from its GitHub tag
`v1.0.0` (a direct reference in `pyproject.toml`), so git must be available.

## Layout

| Path                                                                                                | Contents                                                                                                                  |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `curvature_transfer/graph.py`                                                                     | CSR graph, canonical undirected edges, networkx/edge-list constructors                                                    |
| `curvature_transfer/_kernels.py`                                                                  | Numba kernels: 2-hop statistics and TLW certificate (one stamp-based pass), Hopcroft–Karp; re-exports the exact drivers of `orcut._kernels` |
| `curvature_transfer/bounds.py`                                                                    | all bounds, vectorized over edges                                                        |
| `curvature_transfer/intervals.py`                                                                 | definition of the BF, matching, and sharpened intervals                                        |
| `curvature_transfer/engine.py`                                                                    | `CurvatureEngine`: caching, threading, intervals, edge tables                                                           |
| `curvature_transfer/screening.py`                                                                 | certify-then-solve                                                                                                        |
| `curvature_transfer/_io.py`                                                                       | atomic writes and chunked, resumable computations                                                                         |
| `experiments/suite.py`                                                                            | graph registry: every graph of every table and figure                                                                     |
| `experiments/edges.py`                                                                            | per-edge statistics, bounds, exact curvatures of every registry graph                                  |
| `experiments/compare_bounds.py`                                                                   | Tables 2–5 from the per-edge store                                                                                       |
| `experiments/screening.py`                                                                        | Section 6.3: avoided fractions on the registry graphs (from the per-edge store) and Table 6, timed on large real networks |
| `experiments/large_graphs.py`                                                                     | the large real networks of Table 6: sources, download cache, parsing                                 |
| `experiments/distributions.py`                                                                    | Figures 2–4 from the per-edge store (`make_paper_figures.py` draws them)                                               |
| `experiments/timing.py`                                                                           | timing code shared by the screening table and the runtime figure                                                          |
| `experiments/benchmark.py`                                                                        | runtime scaling, Figure 5 (Section 6.4)                                                                                   |
| `experiments/generators.py`                                                                       | graph generators, unchanged so that seeded graphs reproduce                                                               |

## Reproducing the paper

```bash
pip install -e ".[experiments,dev]"
python main.py                       # every result
python main.py --quick               # smoke run of the whole pipeline (not the paper's numbers)
python main.py --list                # the steps and what each one produces
```

All graphs come from `experiments/suite.py` (restricted to their largest connected component); all per-edge 
quantities are computed by `edges.py` and stored in `results/edges/`; all intervals come from 
`curvature_transfer/intervals.py`; the screening table and the runtime figure use the same timing code 
(`experiments/timing.py`). 

Aggregate statements in the text (ranges, medians, "on k of the N graphs") are over the random graphs and the
real networks (`suite.AGGREGATE_GROUPS`). The canonical graphs (cycle, grid, tori, tree, complete graph) are
listed in the tables but excluded from aggregates, as several bounds are exact on them by construction.

**Checkpointing.** Everything is saved as soon as it is computed, with atomic writes, so a killed job loses 
at most the unit of work in progress:

Every checkpoint name contains a fingerprint of its inputs, so a stale partial result from a different
graph or configuration is never reused. Finished steps are skipped through stamps in `results/.stamps`;
`--force` ignores the stamps but still reuses checkpoints. To recompute from scratch, delete `results/`.

**Runtime benchmark.** Four families (ER, BA, RGG, WS; `--bench-families`) with scaling law of the average 
degree ~ 2.1 n^0.4. `--bench-max-n` sets the largest graph (default 200000). Above 2×10^6 edges a phase is 
timed as the sum of chunk times (marked in the figure). A phase that is predicted to exceed 
`--bench-budget-hours`, or the memory of the allocation, or whose chunk was killed twice, is not run from 
that size on: its extrapolated time is drawn as "predicted, not run" and the reason is in the CSV.
Exact OR keeps per-edge bitsets whose size grows like deg(i)·deg(j)/64 words, so edges between two hubs remain
its most expensive case; the intervals, TLW certificate included, need only O(n) vertex stamps per thread and
are not memory-guarded. Families write to separate folders and can run as parallel jobs, e.g.
`python experiments/benchmark.py --family BA --max-n 200000 --jobs 32 --out results/benchmark`.

**Cost.** The full run is dominated by the runtime benchmark. Expect a few days on a workstation at the default
sizes.

## Exact Ollivier–Ricci curvature

Exact curvatures (`CurvatureEngine.exact`, the per-edge store of `edges.py`, the exact baseline of the screening
table and of the runtime figure, and the last stage of certify-then-solve) come from
[`orcut`](https://github.com/GiorgioMB/orcut) 1.0.0, which reduces the edgewise transport problem to one minimum
cut in exact integer arithmetic, without a distance matrix. The solver, its tests (against an independent oracle
and against linear programming), and its comparison with POT and GraphRicciCurvature are in the orcut
repository. `curvature_transfer._kernels` re-exports its drivers (`exact_ids`, `exact_or_ids`, `screen_ids`,
`tlw_lazy_ids_local`).

## Notes on the experiments

* Certified decisions (the "sign certified" columns and certify-then-solve) use every interval widened by
  `intervals.ROUNDING_TOL` = 1e-9: the bounds are floating point and on graphs where lower and upper bound coincide
  (e.g. K_n) rounding can otherwise put lower > upper.
* "Sign certified" means lower > 0 or upper < 0; the screening query {c_OR < 0} also counts lower >= 0 as
  decided, so its avoided fraction is never below the certified-sign fraction.
* Parallel kernels deal blocks of consecutive edges to the threads cyclically (one chunk per thread), so that
  edges at hubs, which are contiguous in the edge order, are spread over the threads under Numba's static schedule.
  The exact baseline and the bounds use the same scheduling.
* Certify-then-solve (Section 6.3) is timed on the large real networks of `experiments/large_graphs.py`, which are
  downloaded on first use into `results/data/large/`. They include sparse networks, where exact evaluation of every
  edge takes seconds, and denser ones with hubs (Amazon-Photo, Twitch-DE, Crocodile, Squirrel, Amazon-Computers,
  GitHub), where it takes minutes of CPU time. The baseline is exact evaluation of every edge with the same
  solver (`timing.exact_all`); both the baseline and the cascade process the edges in one fixed random order, so that
  the expensive edges at hubs are spread over the threads. Every answer is checked against the exact curvatures.
  On the small registry graphs exact evaluation takes milliseconds, so only their avoided fractions are reported.
