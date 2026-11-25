# Rewrite of the experiment suite for the triangle–quadrangle version of the paper

Branch `tq-suite`. Ground truth and timing baseline: orcut 1.3.1 (faster-or-curvature, native backend).

## Status
- [x] `curvature_transfer/tq_general.py`: Theorem 5.3 for one-step measures (lazy, uniform idleness p,
      per-edge idleness w_j for Corollary 5.4, degree-biased walks), float64, parallel over edge chunks,
      components by breadth-first search, exact bound on the mass at {i, j}. Validated against orcut on
      19,528 edges per measure (5 measures), 0 violations.
- [x] `curvature_transfer/tq_lazy.py`: lazy measure in exact integers, parallel, components by breadth-first
      search, routing order of Theorem 5.3 (iv), plus a lean BF kernel for timing. 0 exact violations on
      33,665 edges; both ends exact on K_8, Q_5, Q_7; lower end exact on trees and long cycles; matches the
      float general kernel to 2e-16.
- [x] `curvature_transfer/full_range.py`: lower end by orcut on adversarial local graphs (batched), upper end by
      the integer program over admissible F (any one-step measure), slot-free relaxation. 0 chain violations
      on 3,351 edges x 4 measures; matches the earlier lazy integer program to 3e-14.
- [ ] `curvature_transfer/prior.py`: the preliminary version's sharpened interval (matching bounds and the
      dual bound of TLW), from the existing 2-hop kernel.
- [ ] `curvature_transfer/screening.py`: certify-then-solve with TQ, then orcut on the undecided edges.
- [ ] experiments (registry, generators and downloads kept; Planetoid added):
      tightness, optimality, measures, kstar, screening (large networks), benchmark (runtime scaling).
- [ ] `experiments/paper_artifacts.py`: generated/numbers.tex, INDEX.md, figures; tests; --quick smoke run.

## Manuscript keys and their producers
| Key(s) | Producer |
|---|---|
| NumGraphs, NumSynthetic, NumReal, OrcutVersion, Reps, Threads | paper_artifacts (registry, environment) |
| TightnessRows, TQSignMin/Max, PriorSignMin/Max | tightness (TQ vs prior, every registry graph) |
| GapLoMean, GapHiMean, SignGapMax (+ "Exact range" columns) | optimality (exact range, graphs with tractable MILP) |
| MeasGapLo, MeasGapHi, MeasSignGap | measures (non-lazy, idleness 1/2 via Corollary 5.4, degree-biased) |
| KStarDecMin/Max, KStarSlowMin/Max | kstar (three-stage pipeline) |
| ScreeningRows, MaxEdges, TQFracMin/Max, DecMin/Max, CTSSpeedMin/Max/Med | screening (large networks) |
| figures_paper/runtime_scaling_{er,ba,rgg,ws}.pdf | benchmark |
