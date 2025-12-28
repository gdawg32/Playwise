#!/usr/bin/env python3
# File: update.py
# Purpose: append new match rows (from a source CSV or FBref via soccerdata) to combined_matches_played.csv
# Usage:
#   python update.py                 # fetches season 2025 via soccerdata (requires soccerdata)
#   python update.py --season 2025   # fetch specific season year
#   python update.py --source new.csv  # use local CSV file with new matches instead of fetching
#
# Notes:
#  - No training or feature generation is run.
#  - A backup combined_matches_played.csv.backup.<ts>.csv is created before changes.

import argparse
from pathlib import Path
import pandas as pd
import re
import shutil
import time
import sys

# Default config
BASE_DIR = Path(__file__).resolve().parents[1]  # ml_pipeline/
COMBINED_CSV = BASE_DIR / "data" / "raw" / "combined_matches_played.csv"
BACKUP = True
DEFAULT_SEASONS = [2025]  # if fetching via soccerdata

# optional soccerdata import
try:
    import soccerdata as sd
except Exception:
    sd = None

# ---------------- utilities ----------------
def numbers_from_text(s):
    if s is None:
        return []
    return [int(x) for x in re.findall(r"\d+", str(s))]

def normalize_date_column(df):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    return df

def parse_score_first_two_ints(score):
    """
    Robust: take first two integer occurrences from the score string and return them as (home, away).
    Returns (None, None) if not found.
    """
    nums = numbers_from_text(score)
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None, None

def normalize_team_name(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def fallback_key_from_row(r):
    # date only (YYYY-MM-DD) + home_lower + away_lower
    d = None
    if "date_parsed" in r and pd.notna(r["date_parsed"]):
        try:
            d = pd.to_datetime(r["date_parsed"], errors="coerce").date().isoformat()
        except Exception:
            d = ""
    elif "date" in r and pd.notna(r["date"]):
        try:
            d = pd.to_datetime(r["date"], errors="coerce").date().isoformat()
        except Exception:
            d = ""
    else:
        d = ""
    home = normalize_team_name(r.get("home_team", "")).lower()
    away = normalize_team_name(r.get("away_team", "")).lower()
    return f"{d}|{home}|{away}"

def ensure_date_parsed_and_hour(df):
    # date_parsed
    if "date_parsed" not in df.columns:
        if "date" in df.columns:
            df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date_parsed"] = pd.NaT
    else:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    # match_hour (integer hour) from 'time' column
    if "match_hour" not in df.columns:
        if "time" in df.columns:
            def parse_hour(t):
                try:
                    dt = pd.to_datetime(str(t), errors="coerce")
                    if pd.isna(dt):
                        m = re.search(r"(\d{1,2}):(\d{2})", str(t))
                        return int(m.group(1)) if m else pd.NA
                    return int(dt.hour)
                except Exception:
                    return pd.NA
            df["match_hour"] = df["time"].apply(parse_hour)
        else:
            df["match_hour"] = pd.NA
    return df

# ---------------- loading functions ----------------
def load_combined(path: Path):
    if not path.exists():
        print(f"{path} not found. Will create new file when appending.")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    if "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    else:
        if "date" in df.columns:
            df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date_parsed"] = pd.NaT
    return df

def fetch_schedule_via_soccerdata(season_year):
    if sd is None:
        raise RuntimeError("soccerdata not installed. Install with: pip install soccerdata")
    fb = sd.FBref(leagues="ENG-Premier League", seasons=season_year, no_cache=True)
    sched = fb.read_schedule()
    df = pd.DataFrame(sched)
    # normalize season short form, e.g., 2025 -> "25/26"
    s = f"{str(season_year)[-2:]}/{str(season_year+1)[-2:]}"
    df["season"] = s
    # ensure date_parsed exists
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    elif "date_parsed" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date_parsed"], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT
    # normalize team strings
    for col in ("home_team","away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())
        else:
            df[col] = pd.NA
    # ensure score col
    if "score" not in df.columns:
        df["score"] = pd.NA
    # try to create a game_id column from common alternatives if missing
    if "game_id" not in df.columns:
        for alt in ("match_id","id","gameid"):
            if alt in df.columns:
                df["game_id"] = df[alt].astype(str)
                break
    return df

def load_source_csv(path: Path):
    df = pd.read_csv(path, dtype=str)
    # If there is no season column and filename contains year, infer simple season short form
    if "season" not in df.columns or df["season"].isnull().all():
        m = re.search(r"(\d{4})", path.stem)
        if m:
            y = int(m.group(1))
            df["season"] = f"{str(y)[-2:]}/{str(y+1)[-2:]}"
        else:
            df["season"] = pd.NA
    # preserve original columns; ensure date_parsed exists
    if "date_parsed" not in df.columns and "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# ---------------- core update logic ----------------
def find_and_append_new_rows(combined_df: pd.DataFrame, source_df: pd.DataFrame, combined_path: Path):
    # prepare combined keys
    existing_game_ids = set()
    if not combined_df.empty and "game_id" in combined_df.columns:
        existing_game_ids = set(combined_df["game_id"].dropna().astype(str).unique())

    existing_keys = set()
    if not combined_df.empty:
        combined_df["_match_key"] = combined_df.apply(fallback_key_from_row, axis=1)
        existing_keys = set(combined_df["_match_key"].dropna().unique())

    # ensure date_parsed/match_hour in source
    source_df = ensure_date_parsed_and_hour(source_df)

    # parse scores -> home_goals/away_goals if missing or not numeric
    if "home_goals" not in source_df.columns or "away_goals" not in source_df.columns:
        source_df["home_goals"] = pd.NA
        source_df["away_goals"] = pd.NA

    # Try robust parse for each row using first two ints in 'score'
    parsed_home = []
    parsed_away = []
    for _, row in source_df.iterrows():
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        # if already present and numeric, keep
        if pd.notna(hg) and str(hg).strip() != "":
            try:
                parsed_home.append(int(hg))
            except:
                parsed_home.append(pd.NA)
        else:
            ph, pa = parse_score_first_two_ints(row.get("score"))
            parsed_home.append(ph if ph is not None else pd.NA)
        if pd.notna(ag) and str(ag).strip() != "":
            try:
                parsed_away.append(int(ag))
            except:
                parsed_away.append(pd.NA)
        else:
            ph2, pa2 = parse_score_first_two_ints(row.get("score"))
            parsed_away.append(pa2 if pa2 is not None else pd.NA)

    source_df["home_goals"] = pd.Series(parsed_home)
    source_df["away_goals"] = pd.Series(parsed_away)

    # mark played rows (we treat as played if both goals present or score string contains at least 2 numbers)
    def row_played(r):
        if pd.notna(r.get("home_goals")) and pd.notna(r.get("away_goals")):
            return True
        s = r.get("score")
        if pd.notna(s) and len(numbers_from_text(s)) >= 2:
            return True
        return False

    source_df["_match_key"] = source_df.apply(fallback_key_from_row, axis=1)
    source_df["_is_played"] = source_df.apply(row_played, axis=1)

    # select rows to append: played and not present by game_id or key
    rows_to_append = []
    for _, r in source_df.iterrows():
        if not r["_is_played"]:
            continue
        gid = str(r.get("game_id")) if "game_id" in r else None
        key = r["_match_key"]
        append = False
        if gid and gid.lower() != "nan" and gid.strip() != "":
            if gid not in existing_game_ids:
                append = True
        else:
            if key not in existing_keys:
                append = True
        if append:
            rows_to_append.append(r)

    if not rows_to_append:
        print("No new played matches to append.")
        return 0

    new_df = pd.DataFrame(rows_to_append).reset_index(drop=True)

    # ensure season exists for new rows
    if "season" not in new_df.columns:
        # if source had season column in some rows, fill; otherwise set from first provided season or leave NA
        new_df["season"] = new_df.get("season", pd.NA)
        if new_df["season"].isnull().all():
            # attempt to use provided seasons from args in calling function; fallback to NA
            # (we set this outside)
            pass

    # ensure date_parsed/match_hour exist (already ensured for source_df earlier)
    new_df = ensure_date_parsed_and_hour(new_df)

    # coerce home_goals/away_goals to numeric where possible
    new_df["home_goals"] = pd.to_numeric(new_df["home_goals"], errors="coerce")
    new_df["away_goals"] = pd.to_numeric(new_df["away_goals"], errors="coerce")

    # backup combined
    if BACKUP and combined_path.exists():
        bak = combined_path.with_suffix(f".backup.{int(time.time())}.csv")
        shutil.copyfile(combined_path, bak)
        print(f"Backup written to {bak}")

    # append preserving existing columns order, and add any new columns at end
    if combined_df.empty:
        final_df = new_df.copy()
    else:
        existing = combined_df.copy()
        union_cols = list(existing.columns) + [c for c in new_df.columns if c not in existing.columns]
        existing_aligned = existing.reindex(columns=union_cols)
        new_aligned = new_df.reindex(columns=union_cols)
        final_df = pd.concat([existing_aligned, new_aligned], ignore_index=True, sort=False)

    # drop helper columns before saving
    for tmp in ("_match_key", "_is_played"):
        if tmp in final_df.columns:
            final_df = final_df.drop(columns=[tmp])

    # save
    final_df = normalize_date_column(final_df)
    final_df.to_csv(combined_path, index=False)
    print(f"Appended {len(new_df)} new rows. Saved {combined_path} (total rows: {len(final_df)})")
    return len(new_df)

# ---------------- main entry ----------------
def main():
    p = argparse.ArgumentParser(description="Append new played matches to combined_matches_played.csv")
    p.add_argument("--source", type=str, default=None,
                   help="Optional path to source CSV with new matches (if not provided, script will fetch via soccerdata for seasons)")
    p.add_argument("--season", type=int, nargs="*", default=DEFAULT_SEASONS,
                   help="Season years to fetch if --source omitted (e.g. --season 2025)")
    args = p.parse_args()

    # load combined
    combined_df = load_combined(COMBINED_CSV)

    # prepare source_df
    if args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Source file {source_path} not found.", file=sys.stderr)
            sys.exit(1)
        source_df = load_source_csv(source_path)
        # if season missing in source, try to fill from provided season argument if single
        if "season" not in source_df.columns or source_df["season"].isnull().all():
            if len(args.season) == 1:
                y = args.season[0]
                source_df["season"] = f"{str(y)[-2:]}/{str(y+1)[-2:]}"
    else:
        # fetch seasons and concat
        if sd is None:
            print("soccerdata not installed and --source not provided; cannot fetch. Install soccerdata or pass --source.", file=sys.stderr)
            sys.exit(1)
        frames = []
        for y in args.season:
            try:
                df = fetch_schedule_via_soccerdata(y)
                frames.append(df)
            except Exception as e:
                print(f"Failed to fetch season {y}: {e}", file=sys.stderr)
        if not frames:
            print("No source data available.", file=sys.stderr)
            sys.exit(1)
        source_df = pd.concat(frames, ignore_index=True, sort=False)

    # do append
    # If source_df has no season column but user specified a single season, fill it
    if "season" not in source_df.columns or source_df["season"].isnull().all():
        if len(args.season) == 1:
            y = args.season[0]
            source_df["season"] = f"{str(y)[-2:]}/{str(y+1)[-2:]}"
    appended_count = find_and_append_new_rows(combined_df, source_df, COMBINED_CSV)
    if appended_count > 0:
        print(f"Done. Appended {appended_count} rows.")
    else:
        print("Done. No rows appended.")

# small helper to load source CSV (kept here for completeness)
def load_source_csv(path: Path):
    df = pd.read_csv(path, dtype=str)
    # ensure a season if inferrable from filename
    if "season" not in df.columns or df["season"].isnull().all():
        m = re.search(r"(\d{4})", path.stem)
        if m:
            y = int(m.group(1))
            df["season"] = f"{str(y)[-2:]}/{str(y+1)[-2:]}"
    if "date_parsed" not in df.columns and "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df

if __name__ == "__main__":
    main()
