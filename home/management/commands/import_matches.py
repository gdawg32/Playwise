import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.dateparse import parse_date, parse_time, parse_datetime
from django.utils import timezone
from django.db import IntegrityError

from home.models import Match, Team


class Command(BaseCommand):
    help = "Import played matches and update existing unplayed fixtures (idempotent)"

    def handle(self, *args, **options):
        csv_path = (
            Path(settings.BASE_DIR)
            / "ml_pipeline"
            / "data"
            / "raw"
            / "combined_matches_played.csv"
        )

        if not csv_path.exists():
            self.stderr.write(self.style.ERROR(f"CSV not found at {csv_path}"))
            return

        created = 0
        updated = 0
        skipped = 0
        self.stdout.write("import_matches command started")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                # ---------- Teams ----------
                try:
                    home_team = Team.objects.get(name=row["home_team"].strip())
                    away_team = Team.objects.get(name=row["away_team"].strip())
                except Team.DoesNotExist:
                    skipped += 1
                    continue

                # ---------- Date / time ----------
                date = parse_date(row.get("date"))
                time = parse_time(row.get("time")) if row.get("time") else None

                if not date:
                    skipped += 1
                    continue

                # ---------- date_parsed ----------
                date_parsed = parse_datetime(row.get("date_parsed"))
                if not date_parsed:
                    skipped += 1
                    continue

                if timezone.is_naive(date_parsed):
                    date_parsed = timezone.make_aware(date_parsed)

                # ---------- Natural key lookup ----------
                try:
                    match, is_created = Match.objects.get_or_create(
                        season=row["season"],
                        date=date,
                        home_team=home_team,
                        away_team=away_team,
                        defaults={
                            "game_id": row.get("game_id"),
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

                # ---------- Track changes ----------
                changed_fields = []

                def change(field, value):
                    if value is None:
                        return
                    if getattr(match, field) != value:
                        setattr(match, field, value)
                        changed_fields.append(field)

                # ---------- Played state ----------
                already_played = (
                    match.home_goals is not None
                    and match.away_goals is not None
                )

                row_is_played = row.get("home_goals") and row.get("away_goals")

                # ---------- Attach game_id if missing ----------
                if row.get("game_id") and not match.game_id:
                    match.game_id = row["game_id"]
                    changed_fields.append("game_id")

                # ---------- Update played data ----------
                if row_is_played and not already_played:
                    change("home_goals", int(float(row["home_goals"])))
                    change("away_goals", int(float(row["away_goals"])))

                    if row.get("home_xg"):
                        change("home_xg", float(row["home_xg"]))
                    if row.get("away_xg"):
                        change("away_xg", float(row["away_xg"]))
                    if row.get("score"):
                        change("score", row["score"])

                # ---------- Freeze metadata once played ----------
                if not already_played:
                    change("week", int(row["week"]) if row.get("week") else None)
                    change("day", row.get("day"))
                    change("time", time)
                    change("date_parsed", date_parsed)
                    change("match_hour", date_parsed.hour if time else None)
                    change("venue", row.get("venue"))
                    change("referee", row.get("referee"))
                    change("match_report", row.get("match_report"))
                    change("notes", row.get("notes"))
                    change(
                        "attendance",
                        int(row["attendance"]) if row.get("attendance") else None,
                    )

                # ---------- Save only if needed ----------
                if changed_fields:
                    match.save(update_fields=changed_fields)
                    if is_created:
                        created += 1
                    else:
                        updated += 1
                else:
                    skipped += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"\nImport complete:"
                f"\n  Created: {created}"
                f"\n  Updated: {updated}"
                f"\n  Skipped: {skipped}"
            )
        )
