# Curvature Transfer: Comparing Edgewise Graph Curvatures and Analytic Envelopes
In this repo we implement the experiments described in the paper `Transport Envelopes and Edgewise Transfer Between Balanced Forman and Ollivier-Ricci Curvatures`. 
The aim is to produce *histograms of edgewise curvature* and compare them against the analytic envelopes and transfer inequalities derived in the paper.

Concretely, for each graph instance we compute per-edge:
- **Lazy OR curvature** $\mathfrak c_{\rm OR}$
- **Balanced Forman curvature** $\mathfrak c_{\rm BF}$
- **Non-lazy OR** $\mathfrak c_{\rm OR-0}$ (neighbors only; used for sign sharpening)
- **Triangle count** $\triangle(i,j)$
- **4-cycle coverage** $\Xi_{ij}$ and its structural limits
- **Envelope terms**: $\Theta_{\rm Const}$, $\Theta_{\rm Slope}$, and the **lazy transport envelope**
- **Transfer bounds**: $\varphi_{\rm BF\to OR}$, $\psi_{\rm BF\to OR}$, $\varphi_{\rm OR\to BF}$, $\psi_{\rm OR\to BF}$

We then produce distributional artifacts:
- Histograms of $\mathfrak c_{\rm OR}$ and $\mathfrak c_{\rm BF}$
- Histogram of the *slack* to the monotone coverage envelope: `Theta_alpha(triangle) - c_OR`
- Histogram of the *slack* to the lazy transport envelope: `cOR_upper - c_OR`
- Summary CSVs and a JSON manifest with metrics including means, std, quantiles, and fraction of edges within analytic bands

## Graph families (aligned with the paragraph)

### (a) Random graph models

- **Erdős–Rényi** (n,p)
- **Watts–Strogatz** small-world model (n, k, beta)
- **Barabási–Albert** preferential attachment (n, m)
- **Random geometric** in the unit square with radius `r`
- **Random d-regular** graphs (n, d)
- **Hyperbolic random graphs** (n, R, alpha, T)
- **2-block SBM (equal communities)** (n, p_in, p_out)


### (b) Canonical combinatorial families

- Cycles `C_n`
- 2D Grids `Grid(m, n)`
- Toroidal grids `C_m x C_n` (wraparound)
- `d`-ary trees of height `h` (finite)
- Complete graphs `K_n`

### (c) Real networks often used in the curvature literature (TODO FOR NOW)
Place edge list CSVs under `experiments/data/` to include these in a run:

- Karate Club (`karate.csv`, from Zachary 1977)
- Jazz collaboration network (`jazz.csv`, from Glaiser--Danon 2003)
- Western US power grid (`power_grid.csv` from Watts--Strogatz 1998)
- Yeast transcription network (`yeast.csv` from Milo et al. 2002)
- arXiv hep-ph citation network (`arxiv.csv`)

**Format:** CSV with two integer columns `u,v` (0-indexed) and no header.
If a file is missing, it is simply skipped.


## Usage
Install requirements and run a preset suite (options: paper, small, tiny).
**Important**: PyTorch 2.1 wheels are compiled against NumPy 1.x; you must pin `numpy==1.26.4` to avoid ABI errors with NumPy 2.x.

```bash
# install with the correct wheel indexes
python -m pip install -U -r requirements.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# then run a preset suite
python experiments/run_experiments.py --preset paper
```

Custom run example:

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

Outputs (CSVs, JSON, PNGs) are written under `experiments/out/<run_name>/`.

## Notes and Tests
- The test suite verifies, per edge of a graph:
  - **Two-sided transfer inequalities** between Balanced Forman (BF) and Ollivier–Ricci (OR) curvature
  - **Lazy-to-non-lazy comparison** and the **lazy transport envelope** upper bound for $c_{\rm OR}$
  - **Property-based tests** assert shape invariants, universal bounds ($c_{\rm OR}, c_{\rm OR-0} \le 1$), degree-1 BF rule, $\Theta$-envelope consistency, and that the bounds contain the exact curvatures

  - **Monotone coverage envelope**: the affine bound $\Theta_\alpha(\triangle)=Const_\alpha+Slope_\alpha\triangle$ with strictly positive slope and $\mathfrak c_{\rm OR}(i,j)\le \Theta_\alpha(\triangle(i,j))$
  - **Structural 4-cycle coverage bounds** used by the BF curvature: $\Xi_{ij}\le \varrho_i+\varrho_j-2-2\,\triangle(i,j)$
  and $\Xi_{ij}\le sho_{\max}(i,j)=\varpi_{\max}(i,j)\,\max\{\varrho_i,\varrho_j\}$
  - A small **coverage monotonicity sanity check** contrasting an interior edge of a path vs. an edge of a 4-cycle
  (same degrees and triangle count but different coverage), showing the lazy transport envelope is larger when coverage is larger.
How to run the tests:
```bash
python -m pip install -U pytest
pytest -q
```
If `torch` is not installed, the entire suite will be skipped with a clear message.
