#!/usr/bin/env bash
set -euo pipefail

uv run python experiments/compare_bounds.py
uv run python experiments/run_experiments.py --preset paper "$@"
