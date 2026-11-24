# Curvature Transfer

Code for the experiments of

> Giorgio Micaletto and Tebe Nigrelli. *From Proxy to Certificate: Two-Sided Ollivier--Ricci Bounds from Balanced Forman Curvature.* 2026. [arXiv:2603.13535](https://arxiv.org/abs/2603.13535)

An earlier version, [*Scalable Graph Curvature*](https://neurips.cc/virtual/2026/loc/paris/159361), was accepted
(non-archival) at the NeurIPS 2026 workshop *Bridging Optimal Transport, Learning and Structured Data*. The Lean 4
formalization is in [BF-OR-Lean](https://github.com/GiorgioMB/BF-OR-Lean/).

For every edge of an undirected simple graph, the package computes the exact lazy, non-lazy and Lin–Lu–Yau
curvatures (with [`orcut`](https://github.com/GiorgioMB/orcut) 1.0.0, the solver of every exact value of the
paper), Balanced Forman curvature and its 2-hop statistics, every certified bound of the paper, and
certify-then-solve screening, which answers threshold queries exactly and solves transport problems only on
the edges its bounds leave undecided.

```python
from curvature_transfer import CurvatureEngine
eng = CurvatureEngine.from_edges([(0, 1), (1, 2), (2, 0), (2, 3)])  # or a networkx graph
eng.bf()                          # Balanced Forman curvature, aligned with eng.edges (u < v)
eng.exact()["c_OR"]               # exact lazy OR curvature (also "c_OR0", "c_LLY")
iv = eng.intervals()              # BF, matching, and sharpened intervals for c_OR
eng.edge_table()                  # everything as a pandas DataFrame
```

Names in the code are not always the paper's symbols (`c_BF`, `c_OR`, `c_OR0`, `c_LLY`, `matching`, `varpi`,
`C4` are κ_BF, κ_OR, κ_0, κ_LLY, M_ij, γ_max, □_ij); the full list is in `curvature_transfer/bounds.py`.

## Reproducing the paper

```bash
./run_experiments.sh                 # every result (uv, with the locked environment of uv.lock)
./run_experiments.sh --quick         # smoke run of the whole pipeline (minutes; not the paper's numbers)
./run_experiments.sh --list          # the steps and what each one produces
./run_experiments.sh --steps paper --paper ~/overleaf --manuscript tex/manuscript-journal.tex
```

Without uv: `pip install -e ".[experiments,dev]"`, then `python main.py` with the same options; `pytest -q`
runs the tests. The `paper` step (`experiments/paper_artifacts.py`) writes what the manuscript reads into
`--paper` (default `results/paper`): `generated/numbers.tex` (every number, `\res{Key}`, and every table body,
`\resrows{label}`), `generated/INDEX.md` (the source file of each number, and the manuscript's keys that are
still missing) and `figures_paper/`. Every step saves its work as soon as it is computed, so a killed run
resumes where it stopped when the same command is run again. The full run takes a few days on a workstation,
most of it in the runtime benchmark.

## Layout

| Path | Contents |
| --- | --- |
| `curvature_transfer/` | the package: CSR graphs, the 2-hop kernels (Numba), all bounds, the intervals, certify-then-solve |
| `experiments/suite.py` | the registry: every graph of every table and figure |
| `experiments/edges.py` | per-edge statistics, bounds and exact curvatures of every registry graph (`results/edges/`) |
| `experiments/compare_bounds.py` | the bound tables of Section 6.1, from the per-edge store |
| `experiments/screening.py`, `screening_map.py` | Section 6.3: avoided transport problems, Table 7 (large networks), the edge map |
| `experiments/distributions.py`, `make_paper_figures.py` | the figures of Sections 6.1–6.2 |
| `experiments/benchmark.py`, `timing.py` | Section 6.4: runtime scaling (the timing code is shared with the screening table) |
| `experiments/paper_artifacts.py` | numbers, table bodies and consistency checks for the manuscript |
| `experiments/generators.py` | graph generators, unchanged so that seeded graphs reproduce |
