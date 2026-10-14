#!/usr/bin/env bash
set -euo pipefail

uv run python main.py --gnn --delta 0.02 "$@"
