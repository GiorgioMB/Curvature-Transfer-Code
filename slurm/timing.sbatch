#!/bin/bash
# One task of tasks/timing.txt per array element, each on a whole node so that the reported times are clean
# (submitted by slurm/submit.sh). Give every timing task the same processor: uncomment the constraint and
# pick a feature of your cluster (on Triton, `slurm features` lists them).
#SBATCH --job-name=tq-timing
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/timing-%A_%a.out
##SBATCH --constraint=CHANGE_ME
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
THREADS=${TIMING_THREADS:-32}
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg NUMBA_NUM_THREADS=$THREADS
.venv/bin/python main.py task --list tasks/timing.txt --threads "$THREADS" "$@"
