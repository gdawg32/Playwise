import pandas as pd
from django.core.management.base import BaseCommand
from home.models import Match, Team  # import your models
from datetime import datetime

class Command(BaseCommand):
    help = "Push unplayed matches from CSV into the Match table"

    def handle(self, *args, **options):
        csv_file = "unplayed_2526.csv"  # path relative to manage.py
        df = pd.read_csv(csv_file)

        inserted = 0
        skipped = 0

        for _, row in df.iterrows():
            # Check for duplicates
            if Match.objects.filter(
                season=row['season'],
                date=row['date'],
                home_team__name=row['hometeam'],
                away_team__name=row['awayteam']
            ).exists():
                skipped += 1
                continue

            # Get or create teams
            home_team, _ = Team.objects.get_or_create(name=row['hometeam'])
            away_team, _ = Team.objects.get_or_create(name=row['awayteam'])

            # Create match
            Match.objects.create(
                season=row['season'],
                date=datetime.strptime(row['date'], "%Y-%m-%d"),
                matchweek=row['matchweek'],
                home_team=home_team,
                away_team=away_team,
                played=False,
                # Leave other stats null for now
            )
            inserted += 1

        self.stdout.write(self.style.SUCCESS(
            f"Inserted {inserted} matches, skipped {skipped} duplicates."
        ))
