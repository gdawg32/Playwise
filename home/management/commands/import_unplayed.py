import csv
from pathlib import Path
from datetime import datetime

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_time
from django.db import IntegrityError

from home.models import Match, Team


class Command(BaseCommand):
    help = "Import unplayed fixtures into Match table"

    def handle(self, *args, **options):
        csv_path = (
            Path(settings.BASE_DIR)
            / "unplayed_2526.csv"
        )

        if not csv_path.exists():
            self.stderr.write(self.style.ERROR(f"CSV not found at {csv_path}"))
            return

        created = 0
        skipped = 0
        updated = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # ---- Teams ----
                try:
                    home_team = Team.objects.get(name=row["home_team"].strip())
                    away_team = Team.objects.get(name=row["away_team"].strip())
                except Team.DoesNotExist:
                    skipped += 1
                    continue

                # ---- Date & time ----
                date = parse_date(row["date"])
                time = parse_time(row["time"]) if row.get("time") else None

                if not date:
                    skipped += 1
                    continue

                # ---- date_parsed (timezone aware) ----
                dt = datetime.combine(date, time or datetime.min.time())
                date_parsed = timezone.make_aware(dt)

                # ---- Match lookup via NATURAL KEY ----
                try:
                    match, is_created = Match.objects.get_or_create(
                        season="25/26",
                        date=date,
                        home_team=home_team,
                        away_team=away_team,
                        defaults={
                            "week": int(row["week"]) if row.get("week") else None,
                            "day": row.get("day"),
                            "time": time,
                            "date_parsed": date_parsed,
                            "match_hour": date_parsed.hour if time else None,
                            "venue": row.get("venue"),
                        },
                    )
                except IntegrityError:
                    skipped += 1
                    continue

                # ---- If match already exists AND is played, skip ----
                if (
                    not is_created
                    and match.home_goals is not None
                    and match.away_goals is not None
                ):
                    skipped += 1
                    continue

                # ---- Safe metadata updates ----
                match.week = match.week or (int(row["week"]) if row.get("week") else None)
                match.day = match.day or row.get("day")
                match.time = match.time or time
                match.date_parsed = match.date_parsed or date_parsed
                match.match_hour = match.match_hour or (date_parsed.hour if time else None)

                match.venue = match.venue or row.get("venue")
                match.referee = match.referee or row.get("referee")
                match.match_report = match.match_report or row.get("match_report")
                match.notes = match.notes or row.get("notes")

                # ---- Attach game_id if present ----
                if row.get("game_id") and not match.game_id:
                    match.game_id = row["game_id"]

                match.save()

                if is_created:
                    created += 1
                else:
                    updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"\nUnplayed fixture import complete:"
                f"\n  Created: {created}"
                f"\n  Updated: {updated}"
                f"\n  Skipped: {skipped}"
            )
        )
