#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/scratch/nerf/dataset/nerf_synthetic
OUT_ROOT="$REPO/output"

for scene in lego mic ship chair ficus hotdog materials drums; do
    python3 "$REPO/eval.py" \
        -m "$OUT_ROOT/$scene" \
        -s "$DATA_ROOT/$scene" \
        --white_background
done
