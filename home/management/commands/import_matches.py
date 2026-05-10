import csv
import re
from datetime import datetime, time as dtime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_date, parse_time, parse_datetime
from django.utils import timezone

from home.models import Match, Team


TEAM_ALIASES = {
    # canonicalize across sources
    "Manchester United": "Manchester Utd",
    "Manchester Utd": "Manchester Utd",
    "Man United": "Manchester Utd",
    "Man Utd": "Manchester Utd",

    "Newcastle United": "Newcastle Utd",
    "Newcastle Utd": "Newcastle Utd",

    "Nottingham Forest": "Nott'ham Forest",
    "Nott'ham Forest": "Nott'ham Forest",

    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",

    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",

    "West Ham United": "West Ham",
    "West Ham": "West Ham",

    "Leeds": "Leeds United",
    "Leeds United": "Leeds United",

    "Brighton & Hove Albion": "Brighton",
    "Brighton": "Brighton",

    "Crystal Palace": "Crystal Palace",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Fulham": "Fulham",
    "Everton": "Everton",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Aston Villa": "Aston Villa",
    "Sunderland": "Sunderland",
    "Manchester City": "Manchester City",
}


class Command(BaseCommand):
    help = "Import played matches and backfill partial match rows safely"

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

        def clean_str(val):
            if val is None:
                return None
            val = str(val).strip()
            return val if val else None

        def to_int(val):
            val = clean_str(val)
            if val is None:
                return None
            try:
                return int(float(val))
            except Exception:
                return None

        def to_float(val):
            val = clean_str(val)
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        def resolve_team(raw_name):
            raw_name = clean_str(raw_name)
            if not raw_name:
                return None

            canonical = TEAM_ALIASES.get(raw_name, raw_name)

            team = Team.objects.filter(name=canonical).first()
            if team:
                return team

            team = Team.objects.filter(name__iexact=canonical).first()
            if team:
                return team

            team = Team.objects.filter(name__iexact=raw_name).first()
            if team:
                return team

            return None

        def parse_match_datetime(date_val, time_val, dt_val):
            dt = parse_datetime(clean_str(dt_val) or "")
            if dt:
                if timezone.is_naive(dt):
                    dt = timezone.make_aware(dt, timezone.get_current_timezone())
                return dt

            d = parse_date(clean_str(date_val) or "")
            t = parse_time(clean_str(time_val) or "")

            if not d:
                return None

            if not t:
                t = dtime(20, 0)

            naive = datetime.combine(d, t)
            return timezone.make_aware(naive, timezone.get_current_timezone())

        def parse_score(score):
            score = clean_str(score)
            if not score:
                return None, None
            m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", score)
            if not m:
                return None, None
            return int(m.group(1)), int(m.group(2))

        created = 0
        updated = 0
        skipped = 0

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            with transaction.atomic():
                for row in reader:
                    season = clean_str(row.get("season"))
                    date = parse_date(clean_str(row.get("date")) or "")
                    if not season or not date:
                        skipped += 1
                        continue

                    home_team = resolve_team(row.get("home_team"))
                    away_team = resolve_team(row.get("away_team"))

                    if not home_team or not away_team:
                        skipped += 1
                        continue

                    match_dt = parse_match_datetime(
                        row.get("date"),
                        row.get("time"),
                        row.get("date_parsed"),
                    )
                    if not match_dt:
                        skipped += 1
                        continue

                    time_val = parse_time(clean_str(row.get("time")) or "")
                    if not time_val and match_dt:
                        time_val = match_dt.timetz().replace(tzinfo=None)

                    home_goals = to_int(row.get("home_goals"))
                    away_goals = to_int(row.get("away_goals"))
                    home_xg = to_float(row.get("home_xg"))
                    away_xg = to_float(row.get("away_xg"))
                    score = clean_str(row.get("score"))

                    # Fill from score if goals are missing
                    if (home_goals is None or away_goals is None) and score:
                        s_home, s_away = parse_score(score)
                        if home_goals is None:
                            home_goals = s_home
                        if away_goals is None:
                            away_goals = s_away

                    # Derive score from goals if score is missing
                    if not score and home_goals is not None and away_goals is not None:
                        score = f"{home_goals}-{away_goals}"

                    # Use natural key for the match
                    match, is_created = Match.objects.get_or_create(
                        season=season,
                        date=date,
                        home_team=home_team,
                        away_team=away_team,
                        defaults={
                            "game_id": clean_str(row.get("game_id")),
                            "week": to_int(row.get("week")),
                            "day": clean_str(row.get("day")),
                            "time": time_val,
                            "date_parsed": match_dt,
                            "match_hour": match_dt.hour if match_dt else None,
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "home_xg": home_xg,
                            "away_xg": away_xg,
                            "score": score,
                            "attendance": to_int(row.get("attendance")),
                            "venue": clean_str(row.get("venue")),
                            "referee": clean_str(row.get("referee")),
                            "match_report": clean_str(row.get("match_report")),
                            "notes": clean_str(row.get("notes")),
                        },
                    )

                    changed_fields = []

                    def set_if_present(field, value):
                        if value is None:
                            return
                        if getattr(match, field) != value:
                            setattr(match, field, value)
                            changed_fields.append(field)

                    # Always backfill anything available
                    set_if_present("game_id", clean_str(row.get("game_id")))
                    set_if_present("week", to_int(row.get("week")))
                    set_if_present("day", clean_str(row.get("day")))
                    set_if_present("time", time_val)
                    set_if_present("date_parsed", match_dt)
                    set_if_present("match_hour", match_dt.hour if match_dt else None)

                    set_if_present("home_goals", home_goals)
                    set_if_present("away_goals", away_goals)
                    set_if_present("home_xg", home_xg)
                    set_if_present("away_xg", away_xg)
                    set_if_present("score", score)

                    set_if_present("attendance", to_int(row.get("attendance")))
                    set_if_present("venue", clean_str(row.get("venue")))
                    set_if_present("referee", clean_str(row.get("referee")))
                    set_if_present("match_report", clean_str(row.get("match_report")))
                    set_if_present("notes", clean_str(row.get("notes")))

                    if is_created:
                        created += 1
                        if changed_fields:
                            match.save(update_fields=changed_fields)
                    else:
                        if changed_fields:
                            match.save(update_fields=changed_fields)
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