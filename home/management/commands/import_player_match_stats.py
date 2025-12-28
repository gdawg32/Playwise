import csv
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from home.models import (
    Match,
    Team,
    Competition,
    PlayerSeasonStat,
    PlayerMatchStat,
)


class Command(BaseCommand):
    help = "Import player match-level stats from player_match_summary_2526.csv"

    def handle(self, *args, **options):
        csv_path = Path("ml_pipeline/data/raw/player_match_summary_2526.csv")

        if not csv_path.exists():
            self.stderr.write(self.style.ERROR(
                f"CSV not found at {csv_path}"
            ))
            return

        created = 0
        updated = 0
        skipped = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            with transaction.atomic():
                for row in reader:
                    game_id = row.get("game_id")
                    player_id = row.get("player_id")

                    if not game_id or not player_id:
                        skipped += 1
                        continue

                    # ---- Match ----
                    try:
                        match = Match.objects.get(game_id=game_id)
                    except Match.DoesNotExist:
                        self.stderr.write(
                            f"Skipping game_id={game_id}: Match not found"
                        )
                        skipped += 1
                        continue

                    # ---- Player ----
                    try:
                        player = PlayerSeasonStat.objects.get(
                            player_id=player_id,
                            season=row.get("season")
                        )
                    except PlayerSeasonStat.DoesNotExist:
                        self.stderr.write(
                            f"Skipping player_id={player_id}: PlayerSeasonStat not found"
                        )
                        skipped += 1
                        continue

                    # ---- Team ----
                    try:
                        team = Team.objects.get(name=row["team"].strip())
                    except Team.DoesNotExist:
                        self.stderr.write(
                            f"Skipping team={row.get('team')}: Team not found"
                        )
                        skipped += 1
                        continue

                    # ---- League ----
                    try:
                        league = Competition.objects.get(name="English Premier League")
                    except Competition.DoesNotExist:
                        self.stderr.write(
                            f"Skipping league={row.get('league')}: Competition not found"
                        )
                        skipped += 1
                        continue

                    # ---- Helpers ----
                    def to_int(val):
                        try:
                            return int(val)
                        except Exception:
                            return None

                    def to_float(val):
                        try:
                            return float(val)
                        except Exception:
                            return None

                    defaults = {
                        "team": team,
                        "league": league,
                        "season": row.get("season"),

                        "player_name": row.get("player_name"),
                        "position": row.get("pos"),

                        "minutes": to_int(row.get("min")),

                        "goals": to_int(row.get("Performance_Gls")),
                        "assists": to_int(row.get("Performance_Ast")),

                        "xg": to_float(row.get("Expected_xG")),
                        "xag": to_float(row.get("Expected_xAG")),

                        "touches": to_int(row.get("Performance_Touches")),
                        "progressive_passes": to_int(row.get("Passes_PrgP")),
                        "progressive_carries": to_int(row.get("Carries_PrgC")),

                        "take_ons_attempted": to_int(row.get("Take-Ons_Att")),
                        "take_ons_successful": to_int(row.get("Take-Ons_Succ")),

                        "tackles": to_int(row.get("Performance_Tkl")),
                        "interceptions": to_int(row.get("Performance_Int")),
                        "blocks": to_int(row.get("Performance_Blocks")),

                        "sca": to_int(row.get("SCA_SCA")),
                        "gca": to_int(row.get("SCA_GCA")),
                    }

                    obj, is_created = PlayerMatchStat.objects.update_or_create(
                        match=match,
                        player=player,
                        defaults=defaults
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
