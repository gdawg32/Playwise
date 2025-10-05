import csv
from django.core.management.base import BaseCommand
from home.models import Match, Team

class Command(BaseCommand):
    help = "Update match stats (xG, goals, etc.) for 2526 season matches from CSV"

    # ✅ specify the path or URL to your CSV here
    CSV_FILE_PATH = "/home/gauresh/Documents/Programming/PlayWise/2526data_fixed.csv"
    # example alternative:
    # CSV_FILE_PATH = "https://example.com/2526data_fixed.csv"

    def handle(self, *args, **options):
        csv_file = self.CSV_FILE_PATH
        updated, skipped = 0, 0

        self.stdout.write(f"Reading CSV from: {csv_file}")

        with open(csv_file, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    season = row["season"]
                    home_team_name = row["hometeam"].strip()
                    away_team_name = row["awayteam"].strip()
                    date = row["date"]

                    home_team = Team.objects.filter(name__iexact=home_team_name).first()
                    away_team = Team.objects.filter(name__iexact=away_team_name).first()

                    if not home_team or not away_team:
                        self.stdout.write(f"⚠️  Team not found: {home_team_name} or {away_team_name}")
                        skipped += 1
                        continue

                    match = Match.objects.filter(
                        season=season,
                        home_team=home_team,
                        away_team=away_team,
                        date=date
                    ).first()

                    if not match:
                        self.stdout.write(f"⚠️  Match not found: {home_team_name} vs {away_team_name} ({date})")
                        skipped += 1
                        continue

                    # safe float conversion helper
                    def safe_float(val):
                        try:
                            return float(val) if val not in ("", None) else None
                        except ValueError:
                            return None

                    # ✅ update stats
                    match.home_gf = safe_float(row.get("homegf"))
                    match.home_ga = safe_float(row.get("homega"))
                    match.away_gf = safe_float(row.get("awaygf"))
                    match.away_ga = safe_float(row.get("awayga"))
                    match.home_xg = safe_float(row.get("homexg"))
                    match.away_xg = safe_float(row.get("awayxg"))
                    match.y_home_goals = safe_float(row.get("y_home_goals"))
                    match.y_away_goals = safe_float(row.get("y_away_goals"))

                    match.save()
                    updated += 1

                except Exception as e:
                    self.stdout.write(f"❌ Error updating {row.get('hometeam')} vs {row.get('awayteam')}: {e}")
                    skipped += 1

        self.stdout.write(self.style.SUCCESS(f"✅ Updated {updated} matches, skipped {skipped}."))
