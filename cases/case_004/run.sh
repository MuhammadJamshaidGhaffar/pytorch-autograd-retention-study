#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$CASE_DIR/collect_metrics.py" "$@"
