#!/usr/bin/env python3
"""
Fetch & normalize FBref player season stats (Standard + Defense + Passing)
Season: 2025 -> stored as 25/26
Output: ml_pipeline/data/raw/player_season_stats_2526.csv
"""

import pandas as pd
import soccerdata as sd
from pathlib import Path

# ---------------- CONFIG ----------------
SEASON_YEAR = 2025
SEASON_DB = "25/26"
LEAGUE = "ENG-Premier League"

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "data" / "raw" / "player_season_stats_2526.csv"

# ---------------- FETCH ----------------
print("📥 Fetching FBref player season stats...")

fbref = sd.FBref(leagues=LEAGUE, seasons=SEASON_YEAR, no_cache=True)

df_std = fbref.read_player_season_stats(stat_type="standard")
df_def = fbref.read_player_season_stats(stat_type="defense")
df_pass = fbref.read_player_season_stats(stat_type="passing")
df_pass_types = fbref.read_player_season_stats(stat_type="passing_types")

# ---------------- HELPERS ----------------
def flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = [
        f"{a}_{b}".strip("_") if isinstance((a, b), tuple) else a
        for a, b in df.columns
    ]
    df.columns = (
        pd.Index(df.columns)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("__", "_")
    )
    return df

# ---------------- FLATTEN ----------------
df_std = flatten(df_std)
df_def = flatten(df_def)
df_pass = flatten(df_pass)
df_pass_types = flatten(df_pass_types)

# ---------------- MERGE ----------------
merge_keys = ["player", "team"]

df = (
    df_std
    .merge(df_def, on=merge_keys, how="left", suffixes=("", "_def"))
    .merge(df_pass, on=merge_keys, how="left", suffixes=("", "_pass"))
    .merge(df_pass_types, on=merge_keys, how="left", suffixes=("", "_ptype"))
)

# ---------------- RENAME CORE ----------------
df = df.rename(columns={
    "player": "player_name",
    "team": "team",
    "league": "league",
    "pos": "pos",
})

df["season"] = SEASON_DB

# ---------------- player_id ----------------
df["player_id"] = (
    df["player_name"].str.lower()
        .str.replace(".", "", regex=False)
        .str.replace(" ", "_", regex=False)
    + "_"
    + df["team"].str.lower().str.replace(" ", "_", regex=False)
)

# ---------------- COLUMN SELECTION ----------------
FINAL_COLS = [
    # Identity
    "league","season","team","player_name","nation","pos","age","born",

    # Playing time
    "Playing_Time_MP","Playing_Time_Starts","Playing_Time_Min","Playing_Time_90s",

    # Performance
    "Performance_Gls","Performance_Ast","Performance_G+A","Performance_G-PK",
    "Performance_PK","Performance_PKatt","Performance_CrdY","Performance_CrdR",

    # Expected
    "Expected_xG","Expected_npxG","Expected_xAG","Expected_npxG+xAG",

    # Progression
    "Progression_PrgC","Progression_PrgP","Progression_PrgR",

    # Per 90
    "Per_90_Minutes_Gls","Per_90_Minutes_Ast","Per_90_Minutes_G+A",
    "Per_90_Minutes_G-PK","Per_90_Minutes_G+A-PK",
    "Per_90_Minutes_xG","Per_90_Minutes_xAG","Per_90_Minutes_xG+xAG",
    "Per_90_Minutes_npxG","Per_90_Minutes_npxG+xAG",

    # Defense
    "Tackles_Tkl","Tackles_TklW",
    "Tackles_Def_3rd","Tackles_Mid_3rd","Tackles_Att_3rd",
    "Int","Tkl+Int","Clr","Blocks_Blocks",
    "Challenges_Tkl%","Challenges_Lost","Err",

    # Passing
    "Total_Cmp","Total_Att","Total_Cmp%",
    "Total_PrgDist","KP","Expected_xA",
    "1/3","PPA","PrgP",
    "Short_Cmp%","Medium_Cmp%","Long_Cmp%",

    # Passing types
    "Pass_Types_TB","Pass_Types_Sw","Pass_Types_Crs",
    "Outcomes_Blocks","Outcomes_Off",

    # ID
    "player_id"
]

df = df[[c for c in FINAL_COLS if c in df.columns]]

# ---------------- SAVE ----------------
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print(f"✅ Saved → {OUT_PATH}")
print(f"📊 Rows: {len(df)} | Columns: {len(df.columns)}")
