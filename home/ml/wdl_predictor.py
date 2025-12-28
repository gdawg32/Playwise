# home/ml/wdl_predictor.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_PATH = BASE_DIR / "artifacts" / "model_artifacts.joblib"
FEATURES_CSV = BASE_DIR.parent / "ml_pipeline" / "data" / "processed" / "features_played_enhanced.csv"

_label_map = {0: "H", 1: "D", 2: "A"}

_artifacts = joblib.load(ARTIFACT_PATH)
_df = pd.read_csv(FEATURES_CSV, parse_dates=["date_parsed"])


def predict_match(home_team: str, away_team: str) -> dict:
    home_team = home_team.strip()
    away_team = away_team.strip()

    features = _artifacts["features"]
    home_map = _artifacts["team_maps"]["home_map"]
    away_map = _artifacts["team_maps"]["away_map"]

    def last_row(team):
        m = (_df.home_team == team) | (_df.away_team == team)
        if not m.any():
            raise ValueError(f"Team '{team}' not found")
        return _df.loc[m].sort_values("date_parsed").iloc[-1]

    h = last_row(home_team)
    a = last_row(away_team)

    def pick(r, t, hf, af):
        return r[hf] if r["home_team"] == t else r[af]

    row = {
        "home_elo": pick(h, home_team, "home_elo", "away_elo"),
        "away_elo": pick(a, away_team, "home_elo", "away_elo"),
    }

    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    row["elo_diff_abs"] = abs(row["elo_diff"])

    row["home_last5_points"] = pick(h, home_team, "home_last5_points", "away_last5_points")
    row["away_last5_points"] = pick(a, away_team, "home_last5_points", "away_last5_points")
    row["last5_points_diff"] = row["home_last5_points"] - row["away_last5_points"]

    row["home_last5_goal_diff"] = pick(h, home_team, "home_last5_goal_diff", "away_last5_goal_diff")
    row["away_last5_goal_diff"] = pick(a, away_team, "home_last5_goal_diff", "away_last5_goal_diff")
    row["last5_goal_diff"] = row["home_last5_goal_diff"] - row["away_last5_goal_diff"]

    row["home_last5_goals_scored"] = pick(h, home_team, "home_last5_goals_scored", "away_last5_goals_scored")
    row["away_last5_goals_scored"] = pick(a, away_team, "home_last5_goals_scored", "away_last5_goals_scored")
    row["goals_last5_diff"] = row["home_last5_goals_scored"] - row["away_last5_goals_scored"]

    row["home_rest_days"] = pick(h, home_team, "home_rest_days", "away_rest_days")
    row["away_rest_days"] = pick(a, away_team, "home_rest_days", "away_rest_days")
    row["rest_days_diff"] = row["home_rest_days"] - row["away_rest_days"]

    row["home_team_te"] = home_map.get(home_team, np.mean(list(home_map.values())))
    row["away_team_te"] = away_map.get(away_team, np.mean(list(away_map.values())))

    row["home_favorite"] = int(row["home_elo"] > row["away_elo"])
    row["elo_x_form"] = row["elo_diff"] * row["last5_points_diff"]

    row["matchday"] = h["matchday"]
    row["days_from_season_start"] = h["days_from_season_start"]
    row["is_weekend"] = h["is_weekend"]

    X = pd.DataFrame([row])
    for f in features:
        if f not in X:
            X[f] = 0.0
    X = X[features].astype("float32")

    scaler = _artifacts["scaler"]
    lgbm = _artifacts["lgb_model"]
    logm = _artifacts["log_model"]
    alpha = _artifacts["ensemble_alpha"]

    probs = alpha * lgbm.predict_proba(X) + (1 - alpha) * logm.predict_proba(scaler.transform(X))
    pred = int(np.argmax(probs))

    return {
        "home": home_team,
        "away": away_team,
        "probs": {
            "H": float(probs[0][0]),
            "D": float(probs[0][1]),
            "A": float(probs[0][2]),
        },
        "predicted": _label_map[pred],
    }
