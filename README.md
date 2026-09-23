# Curvature Transfer: Comparing Edgewise Graph Curvatures and Analytic Envelopes

This repository contains the official implementation of the experiments described in our paper, [Curvature Transfer](https://arxiv.org/abs/2603.13535). The codebase provides a pipeline for evaluating empirical runtime bounds and benchmarking curvature-driven graph neural network (GNN) rewiring.

## Architecture and Core Modules

* **Curvature Engine (`pyg_curvature.py`):** Computes exact Lazy and Non-lazy Ollivier-Ricci curvatures, Balanced Forman curvature, analytic envelopes, and transfer bounds between the two.

* **Topology Rewiring (`curvature_rewiring.py`):** Implements dynamic edge addition and deletion layers for PyTorch Geometric adapted to our `CurvatureEngine`. Features curvature-guided Stochastic Discrete Ricci Flow (SDRF) and Batched Ollivier-Ricci Flow (BORF). Includes also First-Order Spectral Rewiring (FoSR), which iteratively maximizes algebraic connectivity via the Fiedler vector and serves as the primary non-curvature editing baseline for comparative ablation against the curvature-driven methods.


* **GNN Evaluation Pipeline (`gnn_experiments.py`):** Standardized framework executing GCN, GIN, and GAT architectures on datasets including ZINC, Peptides-func/struct, and minesweeper. Uses Optuna for hyperparameter optimization and $K$-Fold cross-validation.



## Execution Presets

The `run_experiments.py` orchestrator supports predefined scaling suites tailored to different execution hardware and computational constraints. Use `--preset <name>` to load these configurations.

* **`tiny`:** A rapid diagnostic suite designed to verify compilation and execution pathways without runtime errors.


* **`small`:** Intended for local workstation evaluation.


* **`medium`:** Evaluates moderate-to-large topologies; omits per-run exploratory plotting and halves the parameters in the GNN pipeline.


* **`paper`:** The complete empirical reproduction suite yielding all artifacts for publication. Evaluates large-scale models, real-world networks, and automatically renders all comparative metric figures. GNNs undergo 100 Optuna optimization trials, 500 training epochs, and 50 independent empirical repetitions to ensure statistical significance.



## Custom Evaluation and Command-Line Interface

Individual structural families, scale parameters, and compute parallelization parameters can be specified explicitly, overriding the preset baselines.

```bash
# Execute specific topologies with maximum multiprocessing
python experiments/run_experiments.py \
  --seed 42 \
  --jobs -1 \
  --er 800 0.01 \
  --hrg 800 5.0 1.0 0.5 \
  --sbm2 1000 0.012 0.004 \
  --cycle 400 \
  --include-real \
  --auto-figures

```

Outputs, including CSV data matrices, JSON metric summaries, and PDF/PNG artifact plots, are directed to `experiments/out/<run_name>/`. The rendering engine (`make_paper_figures.py`) applies unified styling conventions to map the mathematical relationships between $\mathfrak{c}_{\mathrm{BF}}$, $\mathfrak{c}_{\mathrm{OR}}$, and derived theoretical limits across these subdirectories.

## Checkpointing and Soft Restarts

The evaluation pipelines feature fault-tolerant checkpoints.

1. **Topological Properties:** Using the `--soft-restart` flag in `run_experiments.py` bypasses edge curvature calculations for any configuration whose `$TAG_edges.csv` already exists in the target output directory.


2. **GNN Optimization:** GNN performance scores, standard deviations, and optimal architecture parameters are serialized to `gnn_results.json` upon completion of a dataset/architecture pair. The pipeline natively skips re-evaluation of logged configuration pairs. **Note:** If executing codebase modifications (e.g., altering GNN depth, modifying FoSR or SDRF topological limits, or shifting Optuna search domains), you must manually delete or rename `gnn_results.json` to force a complete computational re-evaluation.



## Additional Diagnostics

* **Runtime Scaling Benchmark (`--benchmark`):** Overrides standard execution to generate hardware scaling datasets and log-log visual plots isolating the asymptotic computation cost of Exact Optimal Transport against proposed combinatorial metric bounds across a geometric sequence of graph sizes.


* **SDRF Ablation (`--ablation-only`):** Evaluates the computational scaling of Stochastic Discrete Ricci Flow under variable constraints on `max_candidates` and `max_iters`, projecting heatmap and conditional scaling matrices.

## Testing Suite

Lastly, the repository includes a comprehensive `pytest` suite; to execute it run:

```bash
pytest -q
```
