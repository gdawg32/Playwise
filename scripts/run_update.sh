#!/bin/bash
set -e

echo "======================================"
echo " Running UPDATE step (raw match data)"
echo "======================================"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

python ml_pipeline/update/update.py

# 🔥 FIX: reattach terminal
exec </dev/tty
exec >/dev/tty
exec 2>/dev/tty

python -u manage.py import_matches

echo "✅ Update step completed successfully"
