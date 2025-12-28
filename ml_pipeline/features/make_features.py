# File: feature.py
# Enhanced feature engineering for EPL W/D/L model (season-scoped, no leakage)
# - Reads: combined_matches_played.csv (must contain played matches, with home_goals/away_goals present)
# - Produces: features_played_enhanced.csv with additional improvements including rest days and time features
#
# Usage: put this script in the folder with combined_matches_played.csv and run.
# Output file: features_played_enhanced.csv

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

BASE_DIR = Path(__file__).resolve().parent.parent  # ml_pipeline/
DATA_DIR = BASE_DIR / "data"

INPUT = DATA_DIR / "raw" / "combined_matches_played.csv"
OUTPUT = DATA_DIR / "processed" / "features_played_enhanced.csv"
ROLL_N = 5

if not INPUT.exists():
    raise FileNotFoundError(f"Input '{INPUT}' not found.")

# --- helper functions ---
def elo_expected(home_elo, away_elo, home_adv=100):
    adj_home = home_elo + home_adv
    return 1.0 / (1.0 + 10 ** ((away_elo - adj_home) / 400.0))

def elo_update(home_elo, away_elo, home_result, k=20, home_adv=100):
    exp_home = elo_expected(home_elo, away_elo, home_adv)
    exp_away = 1.0 - exp_home
    new_home = home_elo + k * (home_result - exp_home)
    new_away = away_elo + k * ((1.0 - home_result) - exp_away)
    return new_home, new_away

def result_from_goals(hg, ag):
    if hg > ag:
        return "H", 1.0
    elif hg < ag:
        return "A", 0.0
    else:
        return "D", 0.5

def compute_league_positions(points_dict, goal_diff_dict, goals_scored_dict):
    teams = list(points_dict.keys())
    # ensure presence
    for t in teams:
        points_dict.setdefault(t, 0)
        goal_diff_dict.setdefault(t, 0)
        goals_scored_dict.setdefault(t, 0)
    # sort by points desc, goal diff desc, goals scored desc, team name asc
    standings = sorted(teams, key=lambda t: (-points_dict.get(t,0), -goal_diff_dict.get(t,0), -goals_scored_dict.get(t,0), t))
    pos = {team: rank+1 for rank, team in enumerate(standings)}
    return pos

# --- load and prepare data ---
df = pd.read_csv(INPUT, dtype=str)

# parse date
if "date_parsed" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
else:
    df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")

# ensure numeric goal columns
df["home_goals"] = pd.to_numeric(df.get("home_goals"), errors="coerce")
df["away_goals"] = pd.to_numeric(df.get("away_goals"), errors="coerce")

# sort chronologically
df = df.sort_values("date_parsed").reset_index(drop=True)

# validate essentials
for c in ("season","home_team","away_team","home_goals","away_goals"):
    if c not in df.columns:
        raise ValueError(f"Essential column '{c}' missing from input file.")

# --- season-scoped state containers ---
season_home_elo = defaultdict(lambda: defaultdict(lambda: 1500.0))  # season -> team -> home_elo
season_away_elo = defaultdict(lambda: defaultdict(lambda: 1500.0))  # season -> team -> away_elo

season_last_matches = defaultdict(lambda: defaultdict(lambda: deque(maxlen=ROLL_N)))  # season -> team -> deque of dicts {'points','gd','gs'}

season_points = defaultdict(lambda: defaultdict(int))     # season -> team -> points
season_goals = defaultdict(lambda: defaultdict(int))      # season -> team -> goals scored
season_goal_diff = defaultdict(lambda: defaultdict(int))  # season -> team -> gd
season_matches = defaultdict(lambda: defaultdict(int))    # season -> team -> matches played

out_rows = []

# iterate matches in chronological order and compute pre-match features per match
for idx, row in df.iterrows():
    season = row["season"]
    home = row["home_team"]
    away = row["away_team"]
    hg = int(row["home_goals"]) if pd.notna(row["home_goals"]) else None
    ag = int(row["away_goals"]) if pd.notna(row["away_goals"]) else None

    # --- get pre-match season-scoped Elo ratings ---
    home_elo = season_home_elo[season][home]
    away_elo = season_away_elo[season][away]
    elo_diff = home_elo - away_elo
    elo_diff_abs = abs(elo_diff)

    # --- compute last-N form stats (season-scoped) ---
    def lastn_stats(season_key, team):
        dq = season_last_matches[season_key][team]
        if not dq:
            return 0, 0, 0  # points, gd, goals_scored
        pts = sum(x["points"] for x in dq)
        gd = sum(x["gd"] for x in dq)
        gs = sum(x["gs"] for x in dq)
        return pts, gd, gs

    h_l5_pts, h_l5_gd, h_l5_gs = lastn_stats(season, home)
    a_l5_pts, a_l5_gd, a_l5_gs = lastn_stats(season, away)
    last5_points_diff = h_l5_pts - a_l5_pts
    last5_goal_diff = h_l5_gd - a_l5_gd
    goals_last5_diff = h_l5_gs - a_l5_gs

    # --- season standings positions BEFORE this match ---
    season_team_points = season_points[season]
    season_team_gd = season_goal_diff[season]
    season_team_goals = season_goals[season]

    season_team_points.setdefault(home, 0); season_team_points.setdefault(away, 0)
    season_team_gd.setdefault(home, 0); season_team_gd.setdefault(away, 0)
    season_team_goals.setdefault(home, 0); season_team_goals.setdefault(away, 0)

    positions = compute_league_positions(season_team_points, season_team_gd, season_team_goals)
    home_league_pos = positions.get(home, None)
    away_league_pos = positions.get(away, None)
    if (home_league_pos is not None) and (away_league_pos is not None):
        league_pos_diff = home_league_pos - away_league_pos
    else:
        league_pos_diff = None

    # --- compute target/result from goals (for updating state, not used as pre-match feature) ---
    if hg is None or ag is None:
        result, home_result_score = (None, None)
    else:
        result, home_result_score = result_from_goals(hg, ag)

    # --- build pre-match feature row (do NOT include post-match leaking fields) ---
    out_rows.append({
        "date_parsed": row.get("date_parsed"),
        "season": season,
        "home_team": home,
        "away_team": away,
        # Elo features (season-scoped, home/away-specific)
        "home_elo": round(home_elo, 3),
        "away_elo": round(away_elo, 3),
        "elo_diff": round(elo_diff, 3),
        "elo_diff_abs": round(elo_diff_abs, 3),
        # Rolling form (last N) - season-scoped
        "home_last5_points": h_l5_pts,
        "away_last5_points": a_l5_pts,
        "last5_points_diff": last5_points_diff,
        "home_last5_goal_diff": h_l5_gd,
        "away_last5_goal_diff": a_l5_gd,
        "last5_goal_diff": last5_goal_diff,
        "home_last5_goals_scored": h_l5_gs,
        "away_last5_goals_scored": a_l5_gs,
        "goals_last5_diff": goals_last5_diff,
        # League position features (pre-match)
        "home_league_pos": home_league_pos,
        "away_league_pos": away_league_pos,
        "league_pos_diff": league_pos_diff,
        # Season totals (points) - pre-match
        "home_season_points": season_team_points.get(home, 0),
        "away_season_points": season_team_points.get(away, 0),
        "season_points_diff": season_team_points.get(home, 0) - season_team_points.get(away, 0),
        # result for evaluation
        "result": result
    })

    # --- UPDATE season-scoped histories using this match outcome (after recorded pre-match features) ---
    if hg is not None and ag is not None:
        hg_i = int(hg); ag_i = int(ag)
        # compute points
        if hg_i > ag_i:
            h_pts, a_pts = 3, 0
        elif hg_i < ag_i:
            h_pts, a_pts = 0, 3
        else:
            h_pts, a_pts = 1, 1

        # update rolling last-N
        season_last_matches[season][home].append({"points": h_pts, "gd": hg_i - ag_i, "gs": hg_i})
        season_last_matches[season][away].append({"points": a_pts, "gd": ag_i - hg_i, "gs": ag_i})

        # update season totals
        season_points[season][home] += h_pts
        season_points[season][away] += a_pts
        season_goal_diff[season][home] += (hg_i - ag_i)
        season_goal_diff[season][away] += (ag_i - hg_i)
        season_goals[season][home] += hg_i
        season_goals[season][away] += ag_i
        season_matches[season][home] += 1
        season_matches[season][away] += 1

        # update season-scoped home/away Elo ratings
        h_home_elo = season_home_elo[season][home]
        a_away_elo = season_away_elo[season][away]
        new_h_home_elo, new_a_away_elo = elo_update(h_home_elo, a_away_elo, home_result_score, k=20, home_adv=100)
        season_home_elo[season][home] = new_h_home_elo
        season_away_elo[season][away] = new_a_away_elo

# --- build DataFrame from accumulated rows ---
features_df = pd.DataFrame(out_rows)

# ensure date_parsed is datetime
features_df['date_parsed'] = pd.to_datetime(features_df['date_parsed'])

# --- Add time-based features required by downstream pipeline ---
# matchday: match index within season
features_df = features_df.sort_values('date_parsed').reset_index(drop=True)
features_df['matchday'] = features_df.groupby('season').cumcount() + 1
features_df['dow'] = features_df['date_parsed'].dt.dayofweek
features_df['is_weekend'] = features_df['dow'].isin([5,6]).astype(int)
season_start = features_df.groupby('season')['date_parsed'].transform('min')
features_df['days_from_season_start'] = (features_df['date_parsed'] - season_start).dt.days

# --- Compute rest days for each team (days since previous match for that team) ---
# We'll create a long-form table of (match_idx, team, role, date_parsed), compute prev_date per team, then map back.
long_home = features_df[['date_parsed', 'home_team']].rename(columns={'home_team':'team'})
long_home['role'] = 'home'
long_home['match_idx'] = long_home.index
long_away = features_df[['date_parsed', 'away_team']].rename(columns={'away_team':'team'})
long_away['role'] = 'away'
long_away['match_idx'] = long_away.index
long = pd.concat([long_home, long_away], ignore_index=True).sort_values(['team','date_parsed','match_idx']).reset_index(drop=True)
long['prev_date'] = long.groupby('team')['date_parsed'].shift(1)
long['rest_days'] = (long['date_parsed'] - long['prev_date']).dt.days
# fill missing with a large number (999)
long['rest_days'] = long['rest_days'].fillna(999).astype(int)

# split back to home/away and map to features_df
home_rest = long[long['role']=='home'].set_index('match_idx')['rest_days']
away_rest = long[long['role']=='away'].set_index('match_idx')['rest_days']

# assign to features_df (match_idx corresponds to original features_df index)
features_df['home_rest_days'] = features_df.index.map(home_rest).fillna(999).astype(int)
features_df['away_rest_days'] = features_df.index.map(away_rest).fillna(999).astype(int)

# double-check any missing columns and fill with defaults
for col in ['home_rest_days', 'away_rest_days', 'matchday', 'is_weekend', 'days_from_season_start']:
    if col not in features_df.columns:
        if col in ['home_rest_days','away_rest_days']:
            features_df[col] = 999
        else:
            features_df[col] = 0

# reorder columns for readability and include newly added columns
cols_order = [
    "date_parsed","season","home_team","away_team",
    "home_elo","away_elo","elo_diff","elo_diff_abs",
    "home_last5_points","away_last5_points","last5_points_diff",
    "home_last5_goal_diff","away_last5_goal_diff","last5_goal_diff",
    "home_last5_goals_scored","away_last5_goals_scored","goals_last5_diff",
    "home_rest_days","away_rest_days","rest_days_diff",  # note: rest_days_diff not computed above but can be created if desired
    "matchday","dow","is_weekend","days_from_season_start",
    "home_season_points","away_season_points","season_points_diff",
    "home_league_pos","away_league_pos","league_pos_diff",
    "result"
]
# ensure columns present
cols_present = [c for c in cols_order if c in features_df.columns]
other_cols = [c for c in features_df.columns if c not in cols_present]
features_df = features_df[cols_present + other_cols]

# compute rest_days_diff if not present
if 'rest_days_diff' not in features_df.columns:
    features_df['rest_days_diff'] = features_df['home_rest_days'] - features_df['away_rest_days']

# save
features_df.to_csv(OUTPUT, index=False)
print(f"Saved enhanced features to {OUTPUT}. Rows: {len(features_df)} Columns: {len(features_df.columns)}")
