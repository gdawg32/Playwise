import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import skellam, poisson
import pathlib

# ---------- robust models folder resolution ----------
# Candidate locations to look for 'models' directory (in order).
# Adjust or add paths as needed for your project layout.
HERE = pathlib.Path(__file__).resolve().parent      # .../home/utils
APP_DIR = HERE.parent                                # .../home
PROJECT_DIR = APP_DIR.parent                         # project root (where manage.py usually is)
CWD = pathlib.Path.cwd()

candidate_dirs = [
    APP_DIR / "models",          # e.g. home/models
    PROJECT_DIR / "models",      # e.g. project_root/models
    CWD / "models",              # current working dir/models
    HERE / "models",             # utils/models (unlikely)
]

def find_models_dir():
    for p in candidate_dirs:
        if p.exists() and p.is_dir():
            return str(p)
    # fallback: try environment variable
    env = os.environ.get("MODELS_DIR")
    if env and os.path.isdir(env):
        return env
    # not found, raise informative error listing tried locations
    tried = "\n".join([str(p) for p in candidate_dirs] + [f"env MODELS_DIR={env}" if env else "env MODELS_DIR not set"])
    raise FileNotFoundError(
        "Could not locate models/ directory. Tried the following locations:\n" + tried
    )

MODELS_DIR = find_models_dir()

ART_CLF = os.path.join(MODELS_DIR, "clf.joblib")
ART_CAL = os.path.join(MODELS_DIR, "calibrator.joblib")
ART_REG_HOME = os.path.join(MODELS_DIR, "reg_home.joblib")
ART_REG_AWAY = os.path.join(MODELS_DIR, "reg_away.joblib")
ART_META = os.path.join(MODELS_DIR, "meta.joblib")

# ---------- Load once (with helpful errors) ----------
def safe_load(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing artifact for {name}: expected file at {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {name} from {path}: {e}") from e

clf = safe_load(ART_CLF, "clf")
calibrator = safe_load(ART_CAL, "calibrator")
reg_home = safe_load(ART_REG_HOME, "reg_home")
reg_away = safe_load(ART_REG_AWAY, "reg_away")
meta = safe_load(ART_META, "meta")

# ---------- Helpers ----------
def skellam_diff_probs(home_exp, away_exp, max_diff=6):
    diffs = range(-max_diff, max_diff+1)
    probs = {d: float(skellam.pmf(d, home_exp, away_exp)) for d in diffs}
    s = sum(probs.values())
    return {k: v/s for k, v in probs.items()}

def exact_score_probs(home_exp, away_exp, max_goals=5):
    probs = {}
    for i in range(max_goals+1):
        p_i = poisson.pmf(i, home_exp)
        for j in range(max_goals+1):
            p_j = poisson.pmf(j, away_exp)
            probs[f"{i}-{j}"] = float(p_i * p_j)
    s = sum(probs.values())
    return {k: v/s for k, v in probs.items()}

def get_latest_team_prevalue(team, col_home, col_away, df_reference, medians):
    mask_h = (df_reference["hometeam"] == team)
    mask_a = (df_reference["awayteam"] == team)
    candidates = []
    if mask_h.any():
        r = df_reference.loc[mask_h].iloc[-1]
        if col_home in r:
            candidates.append((r["date"], r[col_home]))
    if mask_a.any():
        r = df_reference.loc[mask_a].iloc[-1]
        if col_away in r:
            candidates.append((r["date"], r[col_away]))
    if not candidates:
        if col_home in medians.index:
            return float(medians[col_home])
        if col_away in medians.index:
            return float(medians[col_away])
        return 0.0
    best = sorted(candidates, key=lambda x: x[0])[-1]
    return float(best[1]) if not pd.isna(best[1]) else (float(medians[col_home]) if col_home in medians.index else 0.0)

def build_feature_row(home_team, away_team, meta):
    medians = meta["medians"]
    feature_cols = meta["feature_cols"]
    df_reference = meta.get("df", meta.get("df_proc", None))
    if df_reference is None:
        raise RuntimeError("meta does not contain reference dataframe for pre-match lookups.")
    team_to_id = meta.get("team_to_id", {})
    fallback_team_id = meta.get("fallback_team_id", 0)

    feat = medians.copy().to_frame().T
    feat = feat.reindex(columns=feature_cols).astype(np.float32)

    def encode_team_runtime(team):
        if pd.isna(team):
            return fallback_team_id
        return team_to_id.get(team, fallback_team_id)

    feat["home_encoded_all"] = encode_team_runtime(home_team)
    feat["away_encoded_all"] = encode_team_runtime(away_team)

    for col in feature_cols:
        if col.startswith("home_"):
            home_col = col
            away_col = "away_" + col.split("home_", 1)[1]
            val = get_latest_team_prevalue(home_team, home_col, away_col, df_reference, medians)
            feat.loc[0, col] = val
        elif col.startswith("away_"):
            away_col = col
            home_col = "home_" + col.split("away_", 1)[1]
            val = get_latest_team_prevalue(away_team, home_col, away_col, df_reference, medians)
            feat.loc[0, col] = val
        elif col in ("elo_home_pre", "elo_away_pre", "elo_diff_pre"):
            if col == "elo_home_pre":
                val = get_latest_team_prevalue(home_team, "elo_home_pre", "elo_away_pre", df_reference, medians)
                feat.loc[0, col] = val
            elif col == "elo_away_pre":
                val = get_latest_team_prevalue(away_team, "elo_home_pre", "elo_away_pre", df_reference, medians)
                feat.loc[0, col] = val
            else:
                eh = float(get_latest_team_prevalue(home_team, "elo_home_pre", "elo_away_pre", df_reference, medians))
                ea = float(get_latest_team_prevalue(away_team, "elo_home_pre", "elo_away_pre", df_reference, medians))
                feat.loc[0, col] = eh - ea

    diffs = {
        "form5_pts_diff": ("home_form5_pts_pre", "away_form5_pts_pre"),
        "gf5_diff": ("home_gf5_pre", "away_gf5_pre"),
        "ga5_diff": ("home_ga5_pre", "away_ga5_pre"),
        "xg5_diff": ("home_xg5_pre", "away_xg5_pre"),
        "team_avg_gf_diff": ("home_team_avg_gf_pre", "away_team_avg_gf_pre"),
        "team_avg_ga_diff": ("home_team_avg_ga_pre", "away_team_avg_ga_pre"),
        "rest_days_diff": ("home_rest_days_pre", "away_rest_days_pre"),
        "h2h_gf_diff": ("h2h_home_gf_pre", "h2h_away_gf_pre"),
    }
    def safe_get(c): return feat.get(c, pd.Series([0])).iloc[0]
    for newcol, (hcol, acol) in diffs.items():
        if hcol in feat.columns and acol in feat.columns:
            feat[newcol] = safe_get(hcol) - safe_get(acol)

    Xrow = feat[feature_cols].values.astype(np.float32)
    return Xrow, feat

# ---------- Public Function ----------
def predict_match(home_team, away_team):
    Xrow, _ = build_feature_row(home_team, away_team, meta)
    probs = calibrator.predict_proba(Xrow)[0]
    home_exp = float(reg_home.predict(Xrow)[0])
    away_exp = float(reg_away.predict(Xrow)[0])

    top_scores = sorted(
        exact_score_probs(home_exp, away_exp, max_goals=5).items(),
        key=lambda x: x[1], reverse=True
    )[:5]

    sk = {k: v for k, v in skellam_diff_probs(home_exp, away_exp, max_diff=6).items() if abs(k) <= 2}

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(probs[0] * 100, 2),
        "draw_prob": round(probs[1] * 100, 2),
        "away_win_prob": round(probs[2] * 100, 2),
        "expected_goals": [round(home_exp, 3), round(away_exp, 3)],
        "top_scores": [[k, v] for k, v in top_scores],
        "skellam_center": {str(k): v for k, v in sk.items()}
    }
