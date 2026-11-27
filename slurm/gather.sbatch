#!/bin/bash
# Merge the chunks and write generated/ and figures_paper/ (submitted by slurm/submit.sh, after both arrays).
#SBATCH --job-name=tq-gather
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gather-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
MPLBACKEND=Agg .venv/bin/python main.py gather
