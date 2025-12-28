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
    game_id = models.CharField(
        max_length=50,
        unique=True,
        null=True,
        blank=True
    )

    season = models.CharField(max_length=10)     
    week = models.PositiveIntegerField(blank=True, null=True)
    day = models.CharField(max_length=10, blank=True, null=True)

    date = models.DateField()
    time = models.TimeField(blank=True, null=True)
    date_parsed = models.DateTimeField()
    match_hour = models.PositiveIntegerField(blank=True, null=True)

    home_team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="home_matches"
    )
    away_team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="away_matches"
    )

    home_goals = models.PositiveIntegerField(blank=True, null=True)
    away_goals = models.PositiveIntegerField(blank=True, null=True)

    home_xg = models.FloatField(blank=True, null=True)
    away_xg = models.FloatField(blank=True, null=True)

    score = models.CharField(max_length=10, blank=True, null=True)

    attendance = models.PositiveIntegerField(blank=True, null=True)
    venue = models.CharField(max_length=100, blank=True, null=True)
    referee = models.CharField(max_length=100, blank=True, null=True)

    match_report = models.URLField(blank=True, null=True)
    notes = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date_parsed"]
        indexes = [
            models.Index(fields=["season"]),
            models.Index(fields=["date_parsed"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["season", "date", "home_team", "away_team"],
                name="unique_match_per_day_per_season"
            )
        ]

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

class PlayerSeasonStat(models.Model):
    # =====================================================
    # Identity
    # =====================================================
    id = models.BigAutoField(primary_key=True)
    player_id = models.CharField(max_length=100)
    player_name = models.CharField(max_length=100)

    nation = models.CharField(max_length=10, blank=True, null=True)
    position = models.CharField(max_length=10, blank=True, null=True)

    age = models.CharField(max_length=10, blank=True, null=True)   # "28-075"
    born = models.PositiveIntegerField(blank=True, null=True)

    # =====================================================
    # Relations
    # =====================================================
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="player_season_stats"
    )

    league = models.ForeignKey(
        Competition,
        on_delete=models.CASCADE,
        related_name="player_season_stats"
    )

    season = models.CharField(max_length=10)  # "25/26"

    # =====================================================
    # Playing Time
    # =====================================================
    playing_time_mp = models.PositiveIntegerField(blank=True, null=True)
    playing_time_starts = models.PositiveIntegerField(blank=True, null=True)
    playing_time_min = models.PositiveIntegerField(blank=True, null=True)
    playing_time_90s = models.FloatField(blank=True, null=True)

    # =====================================================
    # Performance (Raw)
    # =====================================================
    goals = models.PositiveIntegerField(blank=True, null=True)
    assists = models.PositiveIntegerField(blank=True, null=True)
    goals_assists = models.PositiveIntegerField(blank=True, null=True)

    goals_non_penalty = models.PositiveIntegerField(blank=True, null=True)
    penalties_scored = models.PositiveIntegerField(blank=True, null=True)
    penalties_attempted = models.PositiveIntegerField(blank=True, null=True)

    yellow_cards = models.PositiveIntegerField(blank=True, null=True)
    red_cards = models.PositiveIntegerField(blank=True, null=True)

    # =====================================================
    # Expected Metrics
    # =====================================================
    xg = models.FloatField(blank=True, null=True)
    npxg = models.FloatField(blank=True, null=True)
    xag = models.FloatField(blank=True, null=True)
    npxg_xag = models.FloatField(blank=True, null=True)

    # =====================================================
    # Progression
    # =====================================================
    progressive_carries = models.PositiveIntegerField(blank=True, null=True)
    progressive_passes = models.PositiveIntegerField(blank=True, null=True)
    progressive_runs = models.PositiveIntegerField(blank=True, null=True)

    # =====================================================
    # Defensive Actions
    # =====================================================
    tackles = models.PositiveIntegerField(blank=True, null=True)
    tackles_won = models.PositiveIntegerField(blank=True, null=True)

    tackles_defensive_third = models.PositiveIntegerField(blank=True, null=True)
    tackles_middle_third = models.PositiveIntegerField(blank=True, null=True)
    tackles_attacking_third = models.PositiveIntegerField(blank=True, null=True)

    interceptions = models.PositiveIntegerField(blank=True, null=True)
    tackles_interceptions = models.PositiveIntegerField(blank=True, null=True)

    clearances = models.PositiveIntegerField(blank=True, null=True)
    blocks = models.PositiveIntegerField(blank=True, null=True)

    challenge_success_pct = models.FloatField(blank=True, null=True)
    challenges_lost = models.PositiveIntegerField(blank=True, null=True)

    errors = models.PositiveIntegerField(blank=True, null=True)

    # =====================================================
    # Passing Volume & Quality
    # =====================================================
    passes_completed = models.PositiveIntegerField(blank=True, null=True)
    passes_attempted = models.PositiveIntegerField(blank=True, null=True)
    pass_completion_pct = models.FloatField(blank=True, null=True)

    progressive_pass_distance = models.FloatField(blank=True, null=True)

    # =====================================================
    # Chance Creation & Final Third
    # =====================================================
    key_passes = models.PositiveIntegerField(blank=True, null=True)
    expected_assists = models.FloatField(blank=True, null=True)

    passes_into_final_third = models.PositiveIntegerField(blank=True, null=True)
    passes_into_penalty_area = models.PositiveIntegerField(blank=True, null=True)

    # =====================================================
    # Passing Profile (Percentages)
    # =====================================================
    short_pass_completion_pct = models.FloatField(blank=True, null=True)
    medium_pass_completion_pct = models.FloatField(blank=True, null=True)
    long_pass_completion_pct = models.FloatField(blank=True, null=True)

    # =====================================================
    # Passing Types
    # =====================================================
    through_balls = models.PositiveIntegerField(blank=True, null=True)
    switches = models.PositiveIntegerField(blank=True, null=True)
    crosses = models.PositiveIntegerField(blank=True, null=True)

    passes_blocked = models.PositiveIntegerField(blank=True, null=True)
    passes_offside = models.PositiveIntegerField(blank=True, null=True)

    # =====================================================
    # Per 90 Metrics
    # =====================================================
    p90_goals = models.FloatField(blank=True, null=True)
    p90_assists = models.FloatField(blank=True, null=True)
    p90_goals_assists = models.FloatField(blank=True, null=True)

    p90_goals_non_penalty = models.FloatField(blank=True, null=True)
    p90_goals_assists_non_penalty = models.FloatField(blank=True, null=True)

    p90_xg = models.FloatField(blank=True, null=True)
    p90_xag = models.FloatField(blank=True, null=True)
    p90_xg_xag = models.FloatField(blank=True, null=True)

    p90_npxg = models.FloatField(blank=True, null=True)
    p90_npxg_xag = models.FloatField(blank=True, null=True)

    # =====================================================
    # Meta
    # =====================================================
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("player_id", "season")
        ordering = ["-playing_time_min"]
        indexes = [
            models.Index(fields=["season"]),
            models.Index(fields=["team"]),
            models.Index(fields=["league"]),
        ]

    def __str__(self):
        return f"{self.player_name} ({self.team.name}, {self.season})"

class PlayerMatchStat(models.Model):
    # ---- Relations ----
    match = models.ForeignKey(
        Match,
        on_delete=models.CASCADE,
        related_name="player_stats"
    )

    player = models.ForeignKey(
        PlayerSeasonStat,
        on_delete=models.CASCADE,
        related_name="match_stats"
    )

    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="player_match_stats"
    )

    league = models.ForeignKey(
        Competition,
        on_delete=models.CASCADE,
        related_name="player_match_stats"
    )

    season = models.CharField(max_length=10)  # "25/26"

    # ---- Identity (snapshot, denormalized for safety) ----
    player_name = models.CharField(max_length=100)
    position = models.CharField(max_length=10, blank=True, null=True)

    # ---- Playing Time ----
    minutes = models.PositiveIntegerField(blank=True, null=True)

    # ---- Attacking ----
    goals = models.PositiveIntegerField(blank=True, null=True)
    assists = models.PositiveIntegerField(blank=True, null=True)

    xg = models.FloatField(blank=True, null=True)
    xag = models.FloatField(blank=True, null=True)

    # ---- Possession & Progression ----
    touches = models.PositiveIntegerField(blank=True, null=True)
    progressive_passes = models.PositiveIntegerField(blank=True, null=True)
    progressive_carries = models.PositiveIntegerField(blank=True, null=True)

    # ---- Take-ons ----
    take_ons_attempted = models.PositiveIntegerField(blank=True, null=True)
    take_ons_successful = models.PositiveIntegerField(blank=True, null=True)

    # ---- Defensive ----
    tackles = models.PositiveIntegerField(blank=True, null=True)
    interceptions = models.PositiveIntegerField(blank=True, null=True)
    blocks = models.PositiveIntegerField(blank=True, null=True)

    # ---- Chance Creation ----
    sca = models.PositiveIntegerField(blank=True, null=True)  # Shot-Creating Actions
    gca = models.PositiveIntegerField(blank=True, null=True)  # Goal-Creating Actions

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("match", "player")
        indexes = [
            models.Index(fields=["season"]),
            models.Index(fields=["player"]),
            models.Index(fields=["match"]),
        ]
        ordering = ["-match__date_parsed"]

    def __str__(self):
        return f"{self.player_name} — {self.match}"
