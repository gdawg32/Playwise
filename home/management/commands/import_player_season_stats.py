import csv
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from home.models import PlayerSeasonStat, Team, Competition


class Command(BaseCommand):
    help = "Import FBref player season stats into PlayerSeasonStat model (25/26)"

    CSV_PATH = Path("ml_pipeline/data/raw/player_season_stats_2526.csv")
    SEASON = "25/26"
    COMPETITION_NAME = "English Premier League"

    def handle(self, *args, **options):
        if not self.CSV_PATH.exists():
            self.stderr.write(self.style.ERROR(
                f"CSV not found at {self.CSV_PATH}"
            ))
            return

        try:
            competition = Competition.objects.get(name=self.COMPETITION_NAME)
        except Competition.DoesNotExist:
            self.stderr.write(self.style.ERROR(
                f"Competition '{self.COMPETITION_NAME}' not found in DB"
            ))
            return

        created = 0
        updated = 0
        skipped = 0

        with open(self.CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            with transaction.atomic():
                for row in reader:
                    player_id = row.get("player_id")
                    team_name = row.get("team")

                    if not player_id or not team_name:
                        skipped += 1
                        continue

                    # ---- Resolve team ----
                    try:
                        team = Team.objects.get(name=team_name.strip())
                    except Team.DoesNotExist:
                        self.stderr.write(self.style.WARNING(
                            f"Skipping {player_id}: team '{team_name}' not found"
                        ))
                        skipped += 1
                        continue

                    # ---- Helpers ----
                    def to_int(val):
                        try:
                            return int(float(val))
                        except Exception:
                            return None

                    def to_float(val):
                        try:
                            return float(val)
                        except Exception:
                            return None

                    # ---- Create / Update ----
                    obj, is_created = PlayerSeasonStat.objects.update_or_create(
                        player_id=player_id,
                        season=self.SEASON,
                        defaults={
                            # Identity
                            "player_name": row.get("player_name"),
                            "nation": row.get("nation"),
                            "position": row.get("pos"),
                            "age": row.get("age"),
                            "born": to_int(row.get("born")),

                            # Relations
                            "team": team,
                            "league": competition,

                            # Playing time
                            "playing_time_mp": to_int(row.get("Playing_Time_MP")),
                            "playing_time_starts": to_int(row.get("Playing_Time_Starts")),
                            "playing_time_min": to_int(row.get("Playing_Time_Min")),
                            "playing_time_90s": to_float(row.get("Playing_Time_90s")),

                            # Performance
                            "goals": to_int(row.get("Performance_Gls")),
                            "assists": to_int(row.get("Performance_Ast")),
                            "goals_assists": to_int(row.get("Performance_G+A")),
                            "goals_non_penalty": to_int(row.get("Performance_G-PK")),
                            "penalties_scored": to_int(row.get("Performance_PK")),
                            "penalties_attempted": to_int(row.get("Performance_PKatt")),
                            "yellow_cards": to_int(row.get("Performance_CrdY")),
                            "red_cards": to_int(row.get("Performance_CrdR")),

                            # Expected
                            "xg": to_float(row.get("Expected_xG")),
                            "npxg": to_float(row.get("Expected_npxG")),
                            "xag": to_float(row.get("Expected_xAG")),
                            "npxg_xag": to_float(row.get("Expected_npxG+xAG")),

                            # Progression
                            "progressive_carries": to_int(row.get("Progression_PrgC")),
                            "progressive_passes": to_int(row.get("Progression_PrgP")),
                            "progressive_runs": to_int(row.get("Progression_PrgR")),

                            # Defense
                            "tackles": to_int(row.get("Tackles_Tkl")),
                            "tackles_won": to_int(row.get("Tackles_TklW")),
                            "tackles_defensive_third": to_int(row.get("Tackles_Def_3rd")),
                            "tackles_middle_third": to_int(row.get("Tackles_Mid_3rd")),
                            "tackles_attacking_third": to_int(row.get("Tackles_Att_3rd")),
                            "interceptions": to_int(row.get("Int")),
                            "tackles_interceptions": to_int(row.get("Tkl+Int")),
                            "clearances": to_int(row.get("Clr")),
                            "blocks": to_int(row.get("Blocks_Blocks")),
                            "challenge_success_pct": to_float(row.get("Challenges_Tkl%")),
                            "challenges_lost": to_int(row.get("Challenges_Lost")),
                            "errors": to_int(row.get("Err")),

                            # Passing
                            "passes_completed": to_int(row.get("Total_Cmp")),
                            "passes_attempted": to_int(row.get("Total_Att")),
                            "pass_completion_pct": to_float(row.get("Total_Cmp%")),
                            "progressive_pass_distance": to_float(row.get("Total_PrgDist")),
                            "key_passes": to_int(row.get("KP")),
                            "expected_assists": to_float(row.get("Expected_xA")),
                            "passes_into_final_third": to_int(row.get("1/3")),
                            "passes_into_penalty_area": to_int(row.get("PPA")),

                            "short_pass_completion_pct": to_float(row.get("Short_Cmp%")),
                            "medium_pass_completion_pct": to_float(row.get("Medium_Cmp%")),
                            "long_pass_completion_pct": to_float(row.get("Long_Cmp%")),

                            # Passing types
                            "through_balls": to_int(row.get("Pass_Types_TB")),
                            "switches": to_int(row.get("Pass_Types_Sw")),
                            "crosses": to_int(row.get("Pass_Types_Crs")),
                            "passes_blocked": to_int(row.get("Outcomes_Blocks")),
                            "passes_offside": to_int(row.get("Outcomes_Off")),

                            # Per 90
                            "p90_goals": to_float(row.get("Per_90_Minutes_Gls")),
                            "p90_assists": to_float(row.get("Per_90_Minutes_Ast")),
                            "p90_goals_assists": to_float(row.get("Per_90_Minutes_G+A")),
                            "p90_goals_non_penalty": to_float(row.get("Per_90_Minutes_G-PK")),
                            "p90_goals_assists_non_penalty": to_float(row.get("Per_90_Minutes_G+A-PK")),
                            "p90_xg": to_float(row.get("Per_90_Minutes_xG")),
                            "p90_xag": to_float(row.get("Per_90_Minutes_xAG")),
                            "p90_xg_xag": to_float(row.get("Per_90_Minutes_xG+xAG")),
                            "p90_npxg": to_float(row.get("Per_90_Minutes_npxG")),
                            "p90_npxg_xag": to_float(row.get("Per_90_Minutes_npxG+xAG")),
                        }
                    )

                    if is_created:
                        created += 1
                    else:
                        updated += 1

        self.stdout.write(self.style.SUCCESS(
            f"\nImport complete:"
            f"\n  Created: {created}"
            f"\n  Updated: {updated}"
            f"\n  Skipped: {skipped}"
        ))
