#!/bin/bash
# Plan the tasks, compile the kernels and fetch the networks on this (login) node, then submit the compute
# array, the timing array, and the gather, which runs once both arrays have finished.
#
#   slurm/submit.sh                 the experiments of the paper
#   slurm/submit.sh --quick         a smoke run
#   slurm/submit.sh --range-chunk-edges 200          plan options pass through (python main.py plan -h)
#
# MAX_COMPUTE (default 500) and MAX_TIMING (default 4) cap the array elements running at once;
# TIMING_THREADS (default 32) sets the threads of the timing tasks. Rerunning this script resumes: elements
# whose output exists finish at once.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || { echo "first run: uv sync --extra experiments"; exit 1; }
mkdir -p logs
quick=""
for a in "$@"; do [ "$a" = "--quick" ] && quick="--quick"; done
$PY main.py plan "$@"
$PY main.py warmup
$PY main.py download $quick
NC=$(wc -l < tasks/compute.txt); NT=$(wc -l < tasks/timing.txt)
deps=""
if [ "$NC" -gt 0 ]; then
  JC=$(sbatch --parsable --array=0-$((NC - 1))%"${MAX_COMPUTE:-500}" slurm/compute.sbatch)
  deps="$deps:$JC"; echo "compute array: job $JC, $NC tasks"
fi
if [ "$NT" -gt 0 ]; then
  JT=$(sbatch --parsable --array=0-$((NT - 1))%"${MAX_TIMING:-4}" slurm/timing.sbatch)
  deps="$deps:$JT"; echo "timing array: job $JT, $NT tasks"
fi
JG=$(sbatch --parsable --dependency=afterany"$deps" slurm/gather.sbatch)
echo "gather: job $JG, after both arrays; then copy generated/ and figures_paper/ next to the manuscript"
