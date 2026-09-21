# Curvature Transfer: Comparing Edgewise Graph Curvatures and Analytic Envelopes

This repository contains the official implementation of the experiments described in our paper, [Curvature Transfer](https://arxiv.org/abs/2603.13535), and is designed to generate the materials used in it.

## Computed Metrics & Artifacts

For each graph instance, the engine evaluates the following local structural and geometric properties per edge $(i,j)$:

* **Curvatures:** Lazy Ollivier-Ricci, Balanced Forman, and Non-lazy OR
* **Structural Counts:** Triangle count $\triangle(i,j)$ and 4-cycle coverage $\Xi_{ij}$ (alongside its theoretical structural limits).
* **Analytic Envelopes:** Linear envelope parameters ($\Theta_{\mathrm{Const}}$, $\Theta_{\mathrm{Slope}}$) and the exact lazy transport envelope.
* **Transfer Bounds:** Rigorous lower and upper bounds translating between metrics ($\varphi_{\mathrm{BF}\to\mathrm{OR}}$, $\psi_{\mathrm{BF}\to\mathrm{OR}}$, $\varphi_{\mathrm{OR}\to\mathrm{BF}}$, $\psi_{\mathrm{OR}\to\mathrm{BF}}$).

The execution pipeline automatically generates:

* Comparative histograms of $\mathfrak{c}$<sub>OR</sub> and $\mathfrak{c}$<sub>BF</sub>.
* Distributions of the envelope slacks: `Theta_alpha(triangle) - c_OR` and `cOR_upper - c_OR`.
* Comprehensive JSON manifests and CSV logs detailing execution metrics, standard deviations, quantiles, and the strict fraction of edges captured within the analytic transfer bands.
* Runtime scaling benchmarks (CSV logs and log-log PDF plots) isolating the asymptotic execution time of Exact Optimal Transport against the proposed Combinatorial Bounds.

## Supported Graph Families

### Random Graph Models

* **Erdős–Rényi** $(n, p)$
* **Watts–Strogatz** small-world model $(n, k, \beta)$
* **Barabási–Albert** preferential attachment $(n, m)$
* **Random Geometric** (unit square with radius $r$)
* **Random $d$-Regular** $(n, d)$
* **Hyperbolic Random Graphs** $(n, R, \alpha, T)$
* **2-Block Stochastic Block Model** (equal communities with $p_{\mathrm{in}}, p_{\mathrm{out}}$)

### Canonical Combinatorial Families

* **Cycles** $C_n$
* **2D Grids** $m \times n$
* **Toroidal Grids** $C_m \times C_n$ (wraparound)
* **$d$-ary Trees** (finite, height $h$)
* **Complete Graphs** $K_n$

### Real-World Networks

* **Zachary's Karate Club** (`karate.csv`)
* **Jazz Musicians** (`jazz.csv`)
* **US Power Grid** (`power_grid.csv`)
* **Yeast Transcription** (`yeast.csv`)
* **arXiv hep-ph Citations** (`arxiv.csv`)

## Installation & Usage

The codebase requires **Python 3.11**.

```bash
# Install dependencies (CPU)
source installation

# Install dependencies (GPU)
source installation-gpu

# Run the standard paper reproduction suite
python experiments/run_experiments.py --preset paper

```

**Custom Execution Example:**
You can manually choose combinations of graph families, scales, and parallelization using the CLI flags.

```bash
python experiments/run_experiments.py \
  --seed 42 \
  --er 300 0.02 \
  --ws 300 6 0.15 \
  --ba 300 2 \
  --rg 300 0.09 \
  --rreg 300 8 \
  --sbm2 300 0.012 0.004 \
  --cycle 200 \
  --grid 20 20 \
  --torus 20 20 \
  --tree 3 6 \
  --complete 60 \
  --include-real

```

Outputs (CSVs, JSON, and PNGs) will be generated under `experiments/out/<run_name>/`.

## Validation & Testing Suite

The repository includes a comprehensive `pytest` suite; to execute it run:

```bash
pytest -q

```

*(Note: PyTorch is required. If `torch` is not installed, the suite is safely skipped.)*
