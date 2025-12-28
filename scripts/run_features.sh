#!/bin/bash
set -e

echo "======================================"
echo " Running FEATURE ENGINEERING step"
echo "======================================"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional: activate virtualenv
# source venv/bin/activate

python ml_pipeline/features/make_features.py

echo "✅ Feature engineering completed successfully"
