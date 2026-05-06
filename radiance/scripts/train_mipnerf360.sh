#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/scratch/nerf/dataset/nerf_real_360
OUT_ROOT="$REPO/output"

# indoor — downsampled 2×
for scene in bonsai kitchen room counter; do
    python3 "$REPO/train.py" \
        --eval \
        --color_mode voronoi \
        --images images_2 \
        -s "$DATA_ROOT/$scene" \
        -m "$OUT_ROOT/$scene" \
        --config "$REPO/configs/$scene.json"
done

# outdoor — downsampled 4×
for scene in bicycle garden flowers stump treehill; do
    python3 "$REPO/train.py" \
        --eval \
        --color_mode voronoi \
        --images images_4 \
        -s "$DATA_ROOT/$scene" \
        -m "$OUT_ROOT/$scene" \
        --config "$REPO/configs/$scene.json"
done
