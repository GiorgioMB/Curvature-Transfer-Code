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

```bash
python main.py --quick                 # smoke run of every step (minutes, one repetition)
python main.py                         # every step, as in the paper (resumes from results/)
python main.py --steps tightness,range --only "BA(800, 2),Cora"
python main.py --steps paper           # only rebuild generated/ and figures_paper/ from results/
```

Every step writes one file per graph under `results/<step>/` and skips graphs already done, so a run can
be interrupted and resumed; `--force` recomputes. `--threads` sets the threads of numba, orcut and the
integer programs (default: all). On a cluster, `slurm/suite.sbatch` is a starting point.

| Step | What it computes | Manuscript |
|---|---|---|
| `tightness` | non-lazy measure, every registry graph: exact curvature, TQ, BM and TLW, all decided exactly | Table 2 (BM, TLW, TQ); `NumSynthetic`, `NumReal`, `NumGraphs`, `TQSign*`, `BMSign*`, `TLWSign*` |
| `range` | non-lazy measure: the full range of Theorem 5.2, lower end by orcut on the adversarial local graphs, upper end by the integer program over admissible pairs (per edge `--milp-time-limit`, per graph `--range-budget`; a graph with unsolved edges shows "--") | Table 2 (full range); `GapLoMean`, `GapHiMean`, `SignGapMax` |
| `measures` | the same for the lazy measure, idleness 1/2 (certified at w_j and lifted by Corollary 5.4) and a degree-biased walk, on graphs up to `--measures-max-edges` | `MeasGapLo`, `MeasGapHi`, `MeasSignGap` |
| `kstar` | lazy measure: the full range as a middle stage between TQ and orcut, on the edges TQ leaves undecided | `KStarDec*`, `KStarSlow*` |
| `screening` | lazy measure: screen-then-solve against full evaluation on the large networks of `experiments/large_graphs.py` (downloaded from SNAP and GitHub, cached in `experiments/data/large/`), PubMed and Arxiv | Table 3; `MaxEdges`, `TQFrac*`, `Dec*`, `CTSSpeed*` |
| `benchmark` | lazy measure: full evaluation, TQ and BF on four random families of growing size (`--sizes`) | Figure 3 |
| `paper` | `generated/numbers.tex` (every `\res` and `\resrows` key), `generated/INDEX.md` (what each number is), `figures_paper/runtime_scaling_*.pdf` | all |

The `paper` step refuses to write if any check failed: a bound violated on some edge, the chain of
Theorem 5.2 broken, or screen-then-solve disagreeing with full evaluation. To build the manuscript, copy
`generated/` and `figures_paper/` next to the `.tex` file.

## Layout

| Path | Contents |
|---|---|
| `tqbound/tq_lazy.py` | TQ for the lazy measure in exact integers (parallel), and BF |
| `tqbound/tq_general.py` | TQ for one-step measures (lazy, uniform idleness, idleness w_j, degree-biased), float64 |
| `tqbound/full_range.py` | the full range of Theorem 5.2 |
| `tqbound/screen.py` | screen-then-solve (Algorithm 1), exact for every rational threshold |
| `tqbound/baselines.py` | the bounds of BM and TLW (non-lazy measure) |
| `experiments/tq_experiments.py`, `experiments/tq_artifacts.py` | the steps and the paper artifacts |
| `experiments/suite.py`, `generators.py`, `large_graphs.py`, `data/` | the graphs |
| `tests/test_tq.py` | checks against orcut |
