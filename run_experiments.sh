#!/usr/bin/env bash
set -euo pipefail

uv run --extra experiments python main.py "$@"
