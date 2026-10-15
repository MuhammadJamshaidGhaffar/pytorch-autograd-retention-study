#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$CASE_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" "$REPO_ROOT/scripts/phase_2_runtime/measure.py" --case case_001 "$@"
