# The triangle–quadrangle bound: code of the experiments

Code of the experiments (Section 6) of

> Giorgio Micaletto and Tebe Nigrelli. *From Proxy to Certificate: Ollivier–Ricci Bounds from the
> Building Blocks of Balanced Forman Curvature.* 2026.

The library `tqbound` computes the triangle–quadrangle (TQ) bound of Theorem 5.3 for one-step measures, the
full range of Theorem 5.2, screen-then-solve (Algorithm 1), BF, and, as baselines, the bounds of
Bhattacharya and Mukherjee (BM) and of Tian, Lubberts and Weber (TLW); `main.py` runs every experiment and
writes the numbers, tables and figures that the manuscript reads. Exact curvatures, and the baseline of every
timing, come from [orcut](https://github.com/GiorgioMB/orcut) 1.3.1 (`faster-or-curvature`) with integer
fractions; the Lean formalization is in [BF-OR-Lean](https://github.com/GiorgioMB/BF-OR-Lean).

## Install

```bash
uv lock && uv sync --extra experiments --extra dev      # or: pip install -e ".[experiments,dev]"
uv run python -m pytest -q tests/test_tq.py              # 13 fast checks against orcut
```

## Run

On a Slurm cluster (on Triton, from a login node):

```bash
uv sync --extra experiments --extra dev && uv run python -m pytest -q tests/test_tq.py
slurm/submit.sh --quick          # smoke run: a few dozen tasks
slurm/submit.sh                  # the experiments of the paper
```

`slurm/submit.sh` plans the tasks (`tasks/compute.txt`, `tasks/timing.txt`), compiles the kernels and fetches
the large networks on the login node, then submits two job arrays and a gather that runs once both have
finished. Compute tasks (4 cores, 32 GB each) are the tightness of every graph and the full range and the
other measures on chunks of 500 edges, so the integer programs at hubs spread over many tasks; timing tasks
(the kappa* stage, screen-then-solve, the runtime benchmark) run one per exclusive node with a fixed number of
threads. Rerunning `slurm/submit.sh` resumes: tasks whose output exists finish at once. Without Slurm,
`python main.py all [--quick]` runs every task in one process.

| Task | What it computes | Manuscript |
|---|---|---|
| `tightness` (graph) | non-lazy measure: exact curvature, TQ, BM and TLW, all decided exactly | Table 2 (BM, TLW, TQ); `NumSynthetic`, `NumReal`, `NumGraphs`, `TQSign*`, `BMSign*`, `TLWSign*` |
| `range` (graph, chunk) | non-lazy measure: the full range of Theorem 5.2, lower end by orcut on the adversarial local graphs, upper end by the integer program over admissible pairs (`--milp-time-limit` per edge, `--range-budget` per task; a graph with unsolved edges shows "--") | Table 2 (full range); `GapLoMean`, `GapHiMean`, `SignGapMax` |
| `measures` (graph, measure, chunk) | the same for the lazy measure, idleness 1/2 (certified at w_j and lifted by Corollary 5.4) and a degree-biased walk | `MeasGapLo`, `MeasGapHi`, `MeasSignGap` |
| `kstar` (graph) | lazy measure: the full range as a middle stage between TQ and orcut | `KStarDec*`, `KStarSlow*` |
| `screening` (network) | lazy measure: screen-then-solve against full evaluation on the networks of `experiments/large_graphs.py`, PubMed and Arxiv | Table 3; `MaxEdges`, `TQFrac*`, `Dec*`, `CTSSpeed*` |
| `benchmark` (family, size) | lazy measure: full evaluation, TQ and BF on four random families | Figure 3 |
| `gather` | merges the chunks; writes `generated/numbers.tex` (every `\res` and `\resrows` key), `generated/INDEX.md` and `figures_paper/` | all |

Every task records its provenance in `results/_tasks/` (wall and CPU time, allocated cores, peak memory,
node, processor, cluster, Slurm ids, software versions); the gather turns it into the keys of the code
availability statement: `NumTasks`, `CoreHours`, `CPUHours`, `MaxTaskHours`, `PeakMemGB`, `ClusterName`,
`RunMonth`, `CPUModels`, `TimingCPU`, `Threads`, `Reps`, and the versions of Python, NumPy, SciPy, Numba and
orcut.

The gather refuses to write if any check failed: a bound violated on some edge, the chain of
Theorem 5.2 broken, or screen-then-solve disagreeing with full evaluation, or tasks run with different software. To build the manuscript, copy
`generated/` and `figures_paper/` next to the `.tex` file.

## Layout

| Path | Contents |
|---|---|
| `tqbound/tq_lazy.py` | TQ for the lazy measure in exact integers (parallel), and BF |
| `tqbound/tq_general.py` | TQ for one-step measures (lazy, uniform idleness, idleness w_j, degree-biased), float64 |
| `tqbound/full_range.py` | the full range of Theorem 5.2 |
| `tqbound/screen.py` | screen-then-solve (Algorithm 1), exact for every rational threshold |
| `tqbound/baselines.py` | the bounds of BM and TLW (non-lazy measure) |
| `experiments/tq_experiments.py`, `experiments/tasks.py`, `experiments/tq_artifacts.py` | the tasks, their planning and provenance, and the paper's files |
| `slurm/` | the arrays, the gather, and `submit.sh` |
| `experiments/suite.py`, `generators.py`, `large_graphs.py`, `data/` | the graphs |
| `tests/test_tq.py` | checks against orcut |
