#!/bin/bash
set -e  # exit immediately if any command fails

echo "======================================"
echo " 🚀 Updating Player Stats Pipeline"
echo "======================================"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

SEASON_UPDATE_SCRIPT="$PROJECT_ROOT/ml_pipeline/update/update_player_season_stats.py"
MATCH_UPDATE_SCRIPT="$PROJECT_ROOT/ml_pipeline/update/update_player_match_summary_2526.py"

cd "$PROJECT_ROOT"

echo "▶ Updating player SEASON stats CSV..."
python3 "$SEASON_UPDATE_SCRIPT"
echo "✅ Player season stats CSV updated"

echo "▶ Importing player SEASON stats into DB..."
python -u manage.py import_player_season_stats
echo "✅ Player season stats imported"

echo "▶ Updating player MATCH summary CSV..."
python3 "$MATCH_UPDATE_SCRIPT"
echo "✅ Player match summary CSV updated"

echo "▶ Importing player MATCH stats into DB..."
python -u manage.py import_player_match_stats
echo "✅ Player match stats imported"

echo "🎉 Player stats pipeline completed successfully"
