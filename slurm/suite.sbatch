#!/bin/bash
# Every step of the experiments on one node. Adjust the partition, time and memory to your cluster;
# the run resumes from results/, so it can be resubmitted after a timeout.
#SBATCH --job-name=tq-suite
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/suite-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg
uv run --extra experiments python main.py --threads "$SLURM_CPUS_PER_TASK" "$@"
