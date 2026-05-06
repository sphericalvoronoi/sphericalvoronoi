#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/scratch/nerf/dataset/tandt
OUT_ROOT="$REPO/output"

for scene in train truck; do
    python3 "$REPO/eval.py" \
        -m "$OUT_ROOT/$scene" \
        -s "$DATA_ROOT/$scene"
done
