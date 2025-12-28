#!/bin/bash
set -e

echo "======================================"
echo " Running MODEL TRAINING step"
echo "======================================"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional: activate virtualenv
# source venv/bin/activate

python ml_pipeline/training/train.py

echo "✅ Training completed successfully"
