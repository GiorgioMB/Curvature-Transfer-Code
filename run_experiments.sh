#!/usr/bin/env bash
set -euo pipefail

uv run --extra experiments --extra gnn python main.py --gnn --delta 0.02 "$@"
