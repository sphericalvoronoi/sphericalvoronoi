#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/scratch/nerf/dataset/db
OUT_ROOT="$REPO/output"

for scene in drjohnson playroom; do
    python3 "$REPO/train.py" \
        --eval \
        --color_mode voronoi \
        -s "$DATA_ROOT/$scene" \
        -m "$OUT_ROOT/$scene" \
        --config "$REPO/configs/$scene.json"
done
