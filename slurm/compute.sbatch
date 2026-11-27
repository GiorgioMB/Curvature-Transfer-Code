#!/bin/bash
# One task of tasks/compute.txt per array element (submitted by slurm/submit.sh).
#SBATCH --job-name=tq-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/compute-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python main.py task --list tasks/compute.txt --threads "$SLURM_CPUS_PER_TASK" "$@"
