#!/usr/bin/env bash
set -euo pipefail

# Script to run COLMAP CLI pipeline using CPU (no OpenGL/X required)
# Usage: ./scripts/run_colmap_cpu.sh [WORKSPACE_PATH]
# Default WORKSPACE_PATH: ./datasets/fruit

WORK=${1:-"$(pwd)/datasets/fruit"}
IMAGES="$WORK/images/images"
DB="$WORK/database.db"
SPARSE="$WORK/sparse"
DENSE="$WORK/dense"

mkdir -p "$SPARSE" "$DENSE"

echo "Workspace: $WORK"
echo "Images: $IMAGES"

echo "1) Feature extraction (CPU)"
colmap feature_extractor --database_path "$DB" --image_path "$IMAGES" --SiftExtraction.use_gpu 0

echo "2) Exhaustive matching (CPU)"
colmap exhaustive_matcher --database_path "$DB" --SiftMatching.use_gpu 0

echo "3) Sparse reconstruction (mapper)"
colmap mapper --database_path "$DB" --image_path "$IMAGES" --output_path "$SPARSE" --Mapper.num_threads 8

# Find first model folder (usually SPARSE/0)
MODEL_DIR="$(ls -dv $SPARSE/* 2>/dev/null | head -n 1)"
if [ -z "$MODEL_DIR" ]; then
  echo "No sparse model folder found in $SPARSE"
  exit 1
fi

echo "4) Convert model to PLY"
colmap model_converter --input_path "$MODEL_DIR" --output_path "$MODEL_DIR/model.ply" --output_type PLY

# Optionally copy to project assets
ASSETS_DIR="$(pwd)/assets"
mkdir -p "$ASSETS_DIR"
cp "$MODEL_DIR/model.ply" "$ASSETS_DIR/points3D.ply" || true

echo "Done. PLY saved to: $MODEL_DIR/model.ply and copied to $ASSETS_DIR/points3D.ply (if writable)."
