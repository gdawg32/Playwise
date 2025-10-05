from django.db import models
from django.contrib.auth.models import User

class Competition(models.Model):
    name = models.CharField(max_length=100, unique=True)
    country = models.CharField(max_length=50, blank=True, null=True)
    logo = models.ImageField(upload_to='competition_logos/', blank=True, null=True)

    def __str__(self):
        return self.name


class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)
    short_name = models.CharField(max_length=20, blank=True, null=True)
    logo = models.ImageField(upload_to='team_logos/', blank=True, null=True)
    competition = models.ForeignKey(Competition, on_delete=models.CASCADE, related_name='teams')
    home_stadium = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return self.name

class Match(models.Model):
    # Basic match info
    season = models.CharField(max_length=10)  # e.g., "1920"
    date = models.DateField()
    matchweek = models.IntegerField()
    home_team = models.ForeignKey('Team', on_delete=models.CASCADE, related_name='home_matches')
    away_team = models.ForeignKey('Team', on_delete=models.CASCADE, related_name='away_matches')
    
    # Result
    result = models.CharField(max_length=1)  # H/D/A
    y_home_goals = models.IntegerField(blank=True, null=True)
    y_away_goals = models.IntegerField(blank=True, null=True)
    
    # Match stats
    home_gf = models.FloatField(blank=True, null=True)  # Goals for home
    home_ga = models.FloatField(blank=True, null=True)  # Goals against home
    home_xg = models.FloatField(blank=True, null=True)  # Expected goals home
    away_gf = models.FloatField(blank=True, null=True)
    away_ga = models.FloatField(blank=True, null=True)
    away_xg = models.FloatField(blank=True, null=True)
    
    # Ranking & gap
    table_points_gap = models.FloatField(blank=True, null=True)
    home_rank_in_season = models.IntegerField(blank=True, null=True)
    away_rank_in_season = models.IntegerField(blank=True, null=True)

    # Played or not
    played = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.home_team} vs {self.away_team} ({self.season})"

class Coach(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="coach_profile")
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="coach")
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    joined_date = models.DateField(auto_now_add=True)
    image = models.ImageField(upload_to='coach_images/', blank=True, null=True)

    def __str__(self):
        return f"{self.user.username} - {self.team.name}"
