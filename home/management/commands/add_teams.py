from django.core.management.base import BaseCommand
from django.core.files import File
from pathlib import Path

from home.models import Team, Competition


class Command(BaseCommand):
    help = "Add Premier League teams with FBref names and logos"

    def handle(self, *args, **options):
        competition_name = "English Premier League"

        try:
            competition = Competition.objects.get(name=competition_name)
        except Competition.DoesNotExist:
            self.stderr.write(self.style.ERROR(
                f"Competition '{competition_name}' not found. Create it first."
            ))
            return

        # Base directory where logos are stored
        logos_dir = Path("media/team_logos")

        # filename -> FBref official name
        team_map = {
            "arsenal.png": "Arsenal",
            "villa.png": "Aston Villa",
            "bou.png": "Bournemouth",
            "brentford.png": "Brentford",
            "brighton.png": "Brighton",
            "burnley.png": "Burnley",
            "chelsea.png": "Chelsea",
            "crystalpalace.png": "Crystal Palace",
            "everton.png": "Everton",
            "fulham.png": "Fulham",
            "ipswich.png": "Ipswich Town",
            "leeds.png": "Leeds United",
            "leicester.png": "Leicester City",
            "liverpool.png": "Liverpool",
            "luton.png": "Luton Town",
            "mancity.png": "Manchester City",
            "manutd.png": "Manchester Utd",
            "newcastle.png": "Newcastle Utd",
            "nottm.png": "Nott'ham Forest",
            "sheffutd.png": "Sheffield Utd",
            "southampton.png": "Southampton",
            "sunderland.png": "Sunderland",
            "spurs.png": "Tottenham",
            "watford.png": "Watford",
            "westbrom.png": "West Brom",
            "westham.png": "West Ham",
            "wolves.png": "Wolves",
        }

        created = 0
        skipped = 0

        for filename, team_name in team_map.items():
            logo_path = logos_dir / filename

            if not logo_path.exists():
                self.stderr.write(self.style.WARNING(
                    f"Logo not found: {logo_path}"
                ))
                continue

            team, is_created = Team.objects.get_or_create(
                name=team_name,
                defaults={
                    "competition": competition,
                }
            )

            if not is_created:
                skipped += 1
                self.stdout.write(f"Skipped (exists): {team_name}")
                continue

            # attach logo
            with open(logo_path, "rb") as f:
                team.logo.save(filename, File(f), save=True)

            created += 1
            self.stdout.write(self.style.SUCCESS(f"Created: {team_name}"))

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. Created: {created}, Skipped: {skipped}"
            )
        )
