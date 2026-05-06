#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/scratch/nerf/dataset/nerf_real_360
OUT_ROOT="$REPO/output"

# indoor
for scene in bonsai kitchen room counter; do
    python3 "$REPO/eval.py" \
        -m "$OUT_ROOT/$scene" \
        -s "$DATA_ROOT/$scene" \
        
done

# outdoor
for scene in bicycle garden flowers stump treehill; do
    python3 "$REPO/eval.py" \
        -m "$OUT_ROOT/$scene" \
        -s "$DATA_ROOT/$scene" \
        --images images_4
done
