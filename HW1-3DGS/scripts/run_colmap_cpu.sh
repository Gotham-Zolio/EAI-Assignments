#!/usr/bin/env bash
set -euo pipefail
WORK=${1:-"$(pwd)/datasets/fruit"}
IMAGES="$WORK/images"
DB="$WORK/database.db"
SPARSE="$WORK/sparse"
mkdir -p "$SPARSE" "$WORK/dense" "$(pwd)/assets"

colmap feature_extractor --database_path "$DB" --image_path "$IMAGES" --SiftExtraction.use_gpu 0
colmap exhaustive_matcher --database_path "$DB" --SiftMatching.use_gpu 0
colmap mapper --database_path "$DB" --image_path "$IMAGES" --output_path "$SPARSE" --Mapper.num_threads 8

for d in "$SPARSE"/*; do [ -d "$d" ] && MODEL_DIR="$d" && break; done
[ -n "${MODEL_DIR:-}" ] || { echo "no sparse model" >&2; exit 1; }

colmap model_converter --input_path "$MODEL_DIR" --output_path "$MODEL_DIR/model.ply" --output_type PLY
cp -f "$MODEL_DIR/model.ply" "$(pwd)/assets/points3D.ply" 2>/dev/null || true
echo "wrote: $MODEL_DIR/model.ply"
