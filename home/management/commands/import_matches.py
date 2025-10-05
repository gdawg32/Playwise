import csv
from django.core.management.base import BaseCommand
from home.models import Match, Team

class Command(BaseCommand):
    help = 'Import matches from matches.csv into the Match model'

    def handle(self, *args, **kwargs):
        file_path = '/home/gauresh/Documents/Programming/PlayWise/matches.csv'  # Replace with the actual path
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Get home and away teams
                try:
                    home_team = Team.objects.get(name=row['hometeam'])
                    away_team = Team.objects.get(name=row['awayteam'])
                except Team.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f"Team not found: {row['hometeam']} or {row['awayteam']}"))
                    continue

                # Create match entry
                match, created = Match.objects.get_or_create(
                    season=row['season'],
                    date=row['date'],
                    matchweek=int(row['matchweek']),
                    home_team=home_team,
                    away_team=away_team,
                    defaults={
                        'result': row['result'],
                        'y_home_goals': int(row['y_home_goals']),
                        'y_away_goals': int(row['y_away_goals']),
                        'home_gf': float(row['homegf']) if row['homegf'] else None,
                        'home_ga': float(row['homega']) if row['homega'] else None,
                        'home_xg': float(row['homega']) if row['homega'] else None,  # Assuming xG field is missing, adjust if needed
                        'away_gf': float(row['awaygf']) if 'awaygf' in row and row['awaygf'] else None,
                        'away_ga': float(row['awayga']) if 'awayga' in row and row['awayga'] else None,
                        'away_xg': float(row['awayga']) if 'awayga' in row and row['awayga'] else None,
                        'table_points_gap': float(row['table_points_gap']) if row['table_points_gap'] else None,
                        'home_rank_in_season': int(row['home_rank_in_season']) if row['home_rank_in_season'] else None,
                        'away_rank_in_season': int(row['away_rank_in_season']) if row['away_rank_in_season'] else None,
                    }
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f"Created match: {home_team} vs {away_team} on {row['date']}"))
                else:
                    self.stdout.write(self.style.WARNING(f"Match already exists: {home_team} vs {away_team} on {row['date']}"))
