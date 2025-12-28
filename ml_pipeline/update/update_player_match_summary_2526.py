#!/usr/bin/env python3

import soccerdata as sd
import pandas as pd
import time
import unicodedata
import re
from pathlib import Path

# --------------------------------------------------
# PATH SETUP (CRITICAL FIX)
# --------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]          # project root (where manage.py is)
DATA_DIR = PROJECT_ROOT / "ml_pipeline" / "data" / "raw"

MATCHES_CSV = DATA_DIR / "combined_matches_played.csv"
SUMMARY_CSV = DATA_DIR / "player_match_summary_2526.csv"

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
LEAGUE = "ENG-Premier League"
FBREF_SEASON = 2025
CSV_SEASON = "25/26"
SLEEP_SECONDS = 1

# --------------------------------------------------
# TEAM NAME NORMALIZATION
# --------------------------------------------------
TEAM_ALIASES = {
    "Arsenal": ["Arsenal", "Arsenal FC"],
    "Aston Villa": ["Aston Villa", "Aston Villa FC"],
    "Bournemouth": ["Bournemouth", "AFC Bournemouth"],
    "Brentford": ["Brentford", "Brentford FC"],
    "Brighton": [
        "Brighton",
        "Brighton & Hove Albion",
        "Brighton and Hove Albion",
        "Brighton & Hove Albion FC",
    ],
    "Burnley": ["Burnley", "Burnley FC"],
    "Chelsea": ["Chelsea", "Chelsea FC"],
    "Crystal Palace": ["Crystal Palace", "Crystal Palace FC"],
    "Everton": ["Everton", "Everton FC"],
    "Fulham": ["Fulham", "Fulham FC"],
    "Leeds United": ["Leeds United", "Leeds Utd"],
    "Liverpool": ["Liverpool", "Liverpool FC"],
    "Manchester City": ["Manchester City", "Man City", "Manchester City FC"],
    "Manchester Utd": [
        "Manchester United",
        "Man United",
        "Man Utd",
        "Manchester Utd",
        "Manchester United FC",
    ],
    "Newcastle Utd": [
        "Newcastle United",
        "Newcastle Utd",
        "Newcastle United FC",
    ],
    "Nott'ham Forest": [
        "Nottingham Forest",
        "Nott'ham Forest",
        "Nottingham Forest FC",
    ],
    "Sunderland": ["Sunderland", "Sunderland AFC"],
    "Tottenham": [
        "Tottenham",
        "Tottenham Hotspur",
        "Tottenham Hotspur FC",
        "Spurs",
    ],
    "West Ham": ["West Ham", "West Ham United", "West Ham United FC"],
    "Wolves": [
        "Wolves",
        "Wolverhampton Wanderers",
        "Wolverhampton Wanderers FC",
    ],
}

TEAM_LOOKUP = {
    alias.lower(): canon
    for canon, aliases in TEAM_ALIASES.items()
    for alias in aliases
}


def normalize_team(team):
    if pd.isna(team):
        return team
    return TEAM_LOOKUP.get(team.strip().lower(), team)


def slugify(text):
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    s = s.replace(".", "")
    s = s.replace(" ", "_")
    return s


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def flatten_columns(columns):
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            g, s = col
            g = g or ""
            s = s or ""
            flat.append(f"{g}_{s}".strip("_").replace(" ", "_"))
        else:
            flat.append(col)
    return flat


def fetch_clean_summary(fbref, match_id):
    df = fbref.read_player_match_stats(
        stat_type="summary",
        match_id=match_id
    )

    df = df.reset_index()
    df.columns = flatten_columns(df.columns)

    df = df.rename(columns={
        "player": "player_name",
        "team": "team",
        "league": "league",
        "season": "fbref_season",
    })

    keep_cols = [
        "league",
        "fbref_season",
        "team",
        "player_name",
        "pos",
        "min",
        "Performance_Gls",
        "Performance_Ast",
        "Expected_xG",
        "Expected_xAG",
        "Performance_Touches",
        "Passes_PrgP",
        "Carries_PrgC",
        "Take-Ons_Att",
        "Take-Ons_Succ",
        "Performance_Tkl",
        "Performance_Int",
        "Performance_Blocks",
        "SCA_SCA",
        "SCA_GCA",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Normalize team + player_id
    df["team"] = df["team"].apply(normalize_team)
    df["player_id"] = (
        df["player_name"].apply(slugify)
        + "_"
        + df["team"].apply(slugify)
    )

    # Match context
    df["game_id"] = match_id
    df["season"] = CSV_SEASON

    non_numeric = {
        "league", "fbref_season", "team",
        "player_name", "pos",
        "season", "game_id", "player_id"
    }

    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# MAIN UPDATE LOGIC
# --------------------------------------------------
def main():
    if not MATCHES_CSV.exists():
        raise RuntimeError(f"Missing file: {MATCHES_CSV}")

    if not SUMMARY_CSV.exists():
        raise RuntimeError(
            f"{SUMMARY_CSV} not found. Run full build script first."
        )

    matches = pd.read_csv(MATCHES_CSV)

    matches_2526 = matches[
        (matches["season"] == CSV_SEASON) &
        (matches["game_id"].notna())
    ].reset_index(drop=True)

    summary_df = pd.read_csv(SUMMARY_CSV)

    last_game_id = summary_df["game_id"].iloc[-1]
    print(f"Last processed game_id: {last_game_id}")

    if last_game_id not in matches_2526["game_id"].values:
        raise RuntimeError(
            "Last game_id not found in combined CSV. Abort to stay safe."
        )

    start_idx = matches_2526.index[
        matches_2526["game_id"] == last_game_id
    ][0]

    new_matches = matches_2526.iloc[start_idx + 1 :]

    if new_matches.empty:
        print("✅ No new matches found. Nothing to update.")
        return

    print(f"New matches to process: {len(new_matches)}")

    fbref = sd.FBref(
        leagues=LEAGUE,
        seasons=FBREF_SEASON,
    )

    new_dfs = []

    for _, row in new_matches.iterrows():
        game_id = row["game_id"]
        print(f"Processing new match: {game_id}")

        try:
            df = fetch_clean_summary(fbref, game_id)
            new_dfs.append(df)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ Failed for {game_id}: {e}")

    if not new_dfs:
        print("⚠️ No new player data collected.")
        return

    new_data = pd.concat(new_dfs, ignore_index=True)
    final_df = pd.concat([summary_df, new_data], ignore_index=True)

    final_df.to_csv(SUMMARY_CSV, index=False)

    print("\n✅ UPDATE COMPLETE")
    print(f"Added rows: {len(new_data)}")
    print(f"Total rows: {len(final_df)}")


# --------------------------------------------------
if __name__ == "__main__":
    main()
