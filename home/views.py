from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
import json
from .models import *
from django.conf import settings
from .utils.xi_pitch import draw_xi
import soccerdata as sd
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import (
    Avg, Sum, Count, F, FloatField, ExpressionWrapper, Q
)
import io
from io import BytesIO
import statistics
from django.utils import timezone
import threading
import time
from typing import Dict
import re
import numpy as np
from mplsoccer import PyPizza, FontManager
import matplotlib.pyplot as plt
from mplsoccer import Radar, grid
from django.db.models.functions import Coalesce
from decimal import Decimal
from datetime import date
from .ml.wdl_predictor import predict_match
from django.views.decorators.csrf import csrf_exempt
import subprocess
from pathlib import Path
from collections import defaultdict
from scipy.stats import percentileofscore


BASE_DIR = Path(__file__).resolve().parent.parent

ML_DIR = BASE_DIR / "ml_pipeline"
UPDATE_SCRIPT = ML_DIR / "update" / "update.py"
FEATURE_SCRIPT = ML_DIR / "features" / "make_features.py"
TRAIN_SCRIPT = ML_DIR / "training" / "train.py"

# Global state for pipeline progress
pipeline_status = {
    "running": False,
    "current_step": None,
    "steps": [],
    "error": None,
    "logs": []
}

player_stats_status = {
    "running": False,
    "current_step": None,
    "steps": [],
    "error": None,
    "logs": []
}

SEASON = "25/26"
MIN_PLAYER_MINS = 90

def superuser_required(user):
    return user.is_superuser

@login_required
@user_passes_test(superuser_required)
def admin_dashboard(request):
    return render(request, "admin_dashboard.html")

def run_pipeline_async():
    """Run the full pipeline in background"""
    global pipeline_status
    
    pipeline_status["running"] = True
    pipeline_status["error"] = None
    pipeline_status["logs"] = []
    
    steps = [
        {
            "name": "Update Matches",
            "tasks": [
                ("Updating CSV data", ["python", str(UPDATE_SCRIPT)], str(ML_DIR)),
                ("Importing to database", ["python", "manage.py", "import_matches"], str(settings.BASE_DIR))
            ]
        },
        {
            "name": "Generate Features",
            "tasks": [
                ("Building feature dataset", ["python", str(FEATURE_SCRIPT)], str(ML_DIR / "features"))
            ]
        },
        {
            "name": "Train Model",
            "tasks": [
                ("Training prediction model", ["python", str(TRAIN_SCRIPT)], str(ML_DIR / "training"))
            ]
        }
    ]
    
    pipeline_status["steps"] = [{"name": step["name"], "status": "pending"} for step in steps]
    
    try:
        for i, step in enumerate(steps):
            pipeline_status["current_step"] = i
            pipeline_status["steps"][i]["status"] = "running"
            
            for task_name, cmd, cwd in step["tasks"]:
                pipeline_status["logs"].append(f"{task_name}...")
                
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                pipeline_status["logs"].append(f"{task_name} completed")
            
            pipeline_status["steps"][i]["status"] = "completed"
            time.sleep(0.5)  # Brief pause between steps
        
        pipeline_status["current_step"] = None
        pipeline_status["logs"].append("✓ Pipeline completed successfully!")
        
    except subprocess.CalledProcessError as e:
        pipeline_status["error"] = f"Error in {steps[i]['name']}: {e.stderr}"
        pipeline_status["steps"][i]["status"] = "error"
        pipeline_status["logs"].append(f"✗ Pipeline failed: {e.stderr}")
    
    finally:
        pipeline_status["running"] = False


TEAM_NAME_MAP = {
    "Arsenal": [
        "Arsenal", "Arsenal FC"
    ],
    "Aston Villa": [
        "Aston Villa", "Aston Villa FC"
    ],
    "Bournemouth": [
        "Bournemouth", "AFC Bournemouth"
    ],
    "Brentford": [
        "Brentford", "Brentford FC"
    ],
    "Brighton": [
        "Brighton",
        "Brighton & Hove Albion",
        "Brighton and Hove Albion",
        "Brighton & Hove Albion FC"
    ],
    "Burnley": [
        "Burnley", "Burnley FC"
    ],
    "Chelsea": [
        "Chelsea", "Chelsea FC"
    ],
    "Crystal Palace": [
        "Crystal Palace", "Crystal Palace FC"
    ],
    "Everton": [
        "Everton", "Everton FC"
    ],
    "Fulham": [
        "Fulham", "Fulham FC"
    ],
    "Leeds United": [
        "Leeds United", "Leeds Utd"
    ],
    "Liverpool": [
        "Liverpool", "Liverpool FC"
    ],
    "Manchester City": [
        "Manchester City", "Man City", "Manchester City FC"
    ],
    "Manchester Utd": [
        "Manchester United",
        "Man United",
        "Man Utd",
        "Manchester Utd",
        "Manchester United FC"
    ],
    "Newcastle Utd": [
        "Newcastle United",
        "Newcastle Utd",
        "Newcastle United FC"
    ],
    "Nott'ham Forest": [
        "Nottingham Forest",
        "Nott'ham Forest",
        "Nottingham Forest FC"
    ],
    "Sunderland": [
        "Sunderland", "Sunderland AFC"
    ],
    "Tottenham": [
        "Tottenham",
        "Tottenham Hotspur",
        "Tottenham Hotspur FC",
        "Spurs"
    ],
    "West Ham": [
        "West Ham",
        "West Ham United",
        "West Ham United FC"
    ],
    "Wolves": [
        "Wolves",
        "Wolverhampton Wanderers",
        "Wolverhampton Wanderers FC"
    ],
}

STAT_REGISTRY = {
    # ---- Performance (Totals) ----
    "goals": {
        "label": "Goals",
        "order": "-goals",
        "less_is_better": False
    },
    "assists": {
        "label": "Assists",
        "order": "-assists",
        "less_is_better": False
    },
    "goals_assists": {
        "label": "Goals + Assists",
        "order": "-goals_assists",
        "less_is_better": False
    },
    "goals_non_penalty": {
        "label": "Non-Penalty Goals",
        "order": "-goals_non_penalty",
        "less_is_better": False
    },
    "penalties_scored": {
        "label": "Penalties Scored",
        "order": "-penalties_scored",
        "less_is_better": False
    },
    "penalties_attempted": {
        "label": "Penalties Attempted",
        "order": "-penalties_attempted",
        "less_is_better": False
    },

    # ---- Discipline (lower is better) ----
    "yellow_cards": {
        "label": "Yellow Cards",
        "order": "yellow_cards",
        "less_is_better": True
    },
    "red_cards": {
        "label": "Red Cards",
        "order": "red_cards",
        "less_is_better": True
    },

    # ---- Expected (Totals) ----
    "xg": {
        "label": "Expected Goals (xG)",
        "order": "-xg",
        "less_is_better": False
    },
    "npxg": {
        "label": "Non-Penalty xG",
        "order": "-npxg",
        "less_is_better": False
    },
    "xag": {
        "label": "Expected Assists (xA)",
        "order": "-xag",
        "less_is_better": False
    },
    "npxg_xag": {
        "label": "Non-Penalty xG + xA",
        "order": "-npxg_xag",
        "less_is_better": False
    },

    # ---- Progression (Totals) ----
    "progressive_passes": {
        "label": "Progressive Passes",
        "order": "-progressive_passes",
        "less_is_better": False
    },
    "progressive_carries": {
        "label": "Progressive Carries",
        "order": "-progressive_carries",
        "less_is_better": False
    },
    "progressive_runs": {
        "label": "Progressive Runs",
        "order": "-progressive_runs",
        "less_is_better": False
    },

    # ---- Per 90 (Preferred) ----
    "p90_goals": {
        "label": "Goals / 90",
        "order": "-p90_goals",
        "less_is_better": False
    },
    "p90_assists": {
        "label": "Assists / 90",
        "order": "-p90_assists",
        "less_is_better": False
    },
    "p90_goals_assists": {
        "label": "Goals + Assists / 90",
        "order": "-p90_goals_assists",
        "less_is_better": False
    },
    "p90_goals_non_penalty": {
        "label": "Non-Penalty Goals / 90",
        "order": "-p90_goals_non_penalty",
        "less_is_better": False
    },
    "p90_goals_assists_non_penalty": {
        "label": "Non-Penalty G+A / 90",
        "order": "-p90_goals_assists_non_penalty",
        "less_is_better": False
    },
    "p90_xg": {
        "label": "xG / 90",
        "order": "-p90_xg",
        "less_is_better": False
    },
    "p90_xag": {
        "label": "xA / 90",
        "order": "-p90_xag",
        "less_is_better": False
    },
    "p90_xg_xag": {
        "label": "xG + xA / 90",
        "order": "-p90_xg_xag",
        "less_is_better": False
    },
    "p90_npxg": {
        "label": "Non-Penalty xG / 90",
        "order": "-p90_npxg",
        "less_is_better": False
    },
    "p90_npxg_xag": {
        "label": "Non-Penalty xG + xA / 90",
        "order": "-p90_npxg_xag",
        "less_is_better": False
    },
}

ALIAS_TO_FIELD: Dict[str, str] = {
    # per-90 / direct per-90 metrics
    "goals/90": "p90_goals",
    "assists/90": "p90_assists",
    "xG/90": "p90_xg",
    "xA/90": "p90_xag",
    "xG+xA/90": "p90_xg_xag",
    "npg/90": "p90_npxg",
    "npg": "goals_non_penalty",
    "p90_goals_assists": "p90_goals_assists",
    "p90_goals_assists_non_penalty": "p90_goals_assists_non_penalty",
    "p90_goals_non_penalty": "p90_goals_non_penalty",
    "p90_npxg_xag": "p90_npxg_xag",

    # totals / raw
    "goals": "goals",
    "assists": "assists",
    "goals_assists": "goals_assists",
    "goals_non_penalty": "goals_non_penalty",
    "penalties_attempted": "penalties_attempted",
    "penalties_scored": "penalties_scored",

    # xG / xA totals
    "xg": "xg",
    "npxg": "npxg",
    "xag": "xag",
    "npxg_xag": "npxg_xag",

    # progression / volume
    "progressive_passes": "progressive_passes",
    "progPasses": "progressive_passes",
    "progressive_carries": "progressive_carries",
    "progCarries": "progressive_carries",
    "progressive_runs": "progressive_runs",
    "progressive_pass_distance": "progressive_pass_distance",

    # defensive actions
    "tackles": "tackles",
    "tackles_won": "tackles_won",
    "tackles_defensive_third": "tackles_defensive_third",
    "tackles_middle_third": "tackles_middle_third",
    "tackles_attacking_third": "tackles_attacking_third",
    "interceptions": "interceptions",
    "tackles_interceptions": "tackles_interceptions",
    "clearances": "clearances",
    "blocks": "blocks",
    "challenge_success_pct": "challenge_success_pct",
    "challenges_lost": "challenges_lost",
    "errors": "errors",

    # passing volume & quality
    "passes_completed": "passes_completed",
    "passes_attempted": "passes_attempted",
    "pass%": "pass_completion_pct",
    "pass_completion_pct": "pass_completion_pct",
    "progressive_pass_distance": "progressive_pass_distance",

    # chance creation / final third
    "key_passes": "key_passes",
    "Expected_xA": "expected_assists",
    "expected_assists": "expected_assists",
    "passes_into_final_third": "passes_into_final_third",
    "passes_into_penalty_area": "passes_into_penalty_area",
    "PPA": "passes_into_penalty_area",

    # passing profile (percentages)
    "short_pass_completion_pct": "short_pass_completion_pct",
    "medium_pass_completion_pct": "medium_pass_completion_pct",
    "long_pass_completion_pct": "long_pass_completion_pct",

    # passing types
    "through_balls": "through_balls",
    "switches": "switches",
    "crosses": "crosses",
    "passes_blocked": "passes_blocked",
    "passes_offside": "passes_offside",

    # playing time / availability
    "playing_time_90s": "playing_time_90s",
    "playing_time_mp": "playing_time_mp",
    "playing_time_starts": "playing_time_starts",

    # cards / risk
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
}

# ---------------------------
# Raw role weights (human-friendly). Keep it compact & relevant.
# Use keys that are either in ALIAS_TO_FIELD or reasonable aliases mapped above.
# ---------------------------
POSITION_WEIGHTS_RAW: Dict[str, Dict[str, float]] = {
    # Strikers / forwards
    "ST - Pure Finisher": {
        "goals/90": 2.0,
        "npg/90": 1.6,
        "xG/90": 1.4,
        "goals_non_penalty": 1.2
    },
    "ST - Creative Forward": {
        "goals/90": 1.2,
        "assists/90": 1.4,
        "xA/90": 1.6,
        "p90_goals_assists": 1.8
    },

    # Attacking midfield / creators
    "AM - Primary Creator": {
        "assists/90": 1.9,
        "xA/90": 2.0,
        "expected_assists": 1.3,
        "key_passes": 1.2
    },
    "AM - Secondary Scorer": {
        "goals/90": 1.6,
        "xG/90": 1.5,
        "p90_goals_assists": 1.2
    },

    # Wingers / wide players
    "Winger - Classic": {
        "crosses": 1.5,
        "assists/90": 1.4,
        "xA/90": 1.4,
        "progressive_carries": 1.2
    },
    "Winger - Ball Progressor": {
        "progressive_carries": 1.8,
        "progressive_passes": 1.5,
        "progressive_runs": 1.2
    },

    # Central midfield
    "CM - Water Carrier": {
        "progressive_carries": 1.2,
        "progressive_passes": 1.2,
        "pass%": 1.05,
        "challenge_success_pct": 0.9
    },
    "CM - Progresser": {
        "progressive_passes": 1.8,
        "progressive_pass_distance": 1.4,
        "progressive_carries": 1.0
    },
    "CM - Box-to-Box": {
        "p90_goals_assists": 1.4,
        "progressive_passes": 1.3,
        "progressive_carries": 1.3,
        "tackles": 1.1
    },

    # Defensive roles
    "CB - Ball-Playing": {
        "progressive_passes": 1.6,
        "pass%": 1.25,
        "interceptions": 1.2,
        "clearances": 1.0
    },
    "CB - Stopper": {
        "tackles": 1.8,
        "tackles_interceptions": 1.6,
        "clearances": 1.4,
        "blocks": 1.0
    },
    "DM - Ball Winner": {
        "tackles": 1.9,
        "interceptions": 1.7,
        "tackles_interceptions": 1.6,
        "challenge_success_pct": 1.2
    },

    # Fullback / wingback
    "Fullback - Overlapping": {
        "progressive_carries": 1.4,
        "progressive_passes": 1.3,
        "crosses": 1.2,
        "tackles": 1.0
    },

    # Meta / squad value
    "Durability Asset": {
        "playing_time_mp": 1.4,
        "playing_time_starts": 1.6,
        "playing_time_90s": 1.2
    },
    "Risk-Aware Player": {
        "yellow_cards": -1.5,
        "red_cards": -2.0,
        "playing_time_90s": 1.0
    },

    "Overall": {
    # Attacking output
    "p90_goals": 1.0,
    "p90_assists": 1.0,
    "p90_goals_assists": 1.2,

    # Chance quality
    "p90_xg": 1.0,
    "p90_xag": 1.0,
    "p90_xg_xag": 1.2,

    # Ball progression
    "progressive_passes": 1.0,
    "progressive_carries": 1.0,

    # Possession & circulation
    "passes_completed": 0.8,
    "pass_completion_pct": 0.8,

    # Defensive contribution (light, role-agnostic)
    "tackles_interceptions": 0.8,
    "clearances": 0.6,

    # Availability / trust
    "playing_time_90s": 0.6,
    },

}



def build_cleaned_weights(raw_map):
    cleaned = {}
    for role, wmap in raw_map.items():
        cw = {}
        for key, weight in wmap.items():
            mapped = ALIAS_TO_FIELD.get(key)
            if mapped:
                cw[mapped] = float(weight)
        if cw:
            cleaned[role] = cw
    return cleaned

POSITION_WEIGHTS = build_cleaned_weights(POSITION_WEIGHTS_RAW)


# ---------------------------
# Position-specific adjustment functions (post score transform)
# ---------------------------
POSITION_ADJUSTMENTS = {
    # Centre-backs: defensive impact compressed by percentiles
    "CB - Ball-Playing": lambda x: min(x * 1.04, 100),
    "CB - Stopper":      lambda x: min(x * 1.06, 100),

    # Defensive midfielders: influence spread across many stats
    "DM - Ball Winner":  lambda x: min(x * 1.06, 100),
    "DM - Deep-Lying Playmaker": lambda x: min(x * 1.05, 100),

    # Structural gap (until aerial stats exist)
    "ST - Target Man":   lambda x: min(x * 1.05, 100),
}

# Keep only adjustments for roles that exist in POSITION_WEIGHTS
POSITION_ADJUSTMENTS = {k: v for k, v in POSITION_ADJUSTMENTS.items() if k in POSITION_WEIGHTS}


# ---------------------------
# Stats where lower value is better (so percentiles are flipped)
# ---------------------------
LESS_IS_BETTER_KEYS = {
    "yellow_cards",
    "red_cards",
    "penalties_attempted",
    "errors",
    "challenges_lost",
}

def percentiles_vs_pool(values_pool, series_values, flip=False):

    out = []
    if values_pool.size == 0:
        return [np.nan] * len(series_values)

    pool = np.array(values_pool, dtype=float)
    if flip:
        pool = -pool
    pool_sorted = np.sort(pool)
    n = pool_sorted.size

    for v in series_values:
        if v is None:
            out.append(np.nan)
            continue
        val = float(v)
        if flip:
            val = -val
        # lower and upper rank
        lo = np.searchsorted(pool_sorted, val, side="left")
        hi = np.searchsorted(pool_sorted, val, side="right")
        mean_rank = (lo + hi) / 2.0
        pct = (mean_rank / n) * 100.0
        out.append(pct)
    return out

def normalize_team_name(name: str) -> str:
    """
    Convert any known team alias to canonical PlayWise team name.
    If no match found, return cleaned original.
    """
    if not name:
        return ""

    name_clean = name.strip().lower()

    for canonical, variants in TEAM_NAME_MAP.items():
        for v in variants:
            if name_clean == v.lower():
                return canonical

    # fallback: title-cased original (safe default)
    return name.strip()



# Create your views here.
def home(request):
    return render(request, 'index.html')

def superuser_required(user):
    return user.is_superuser


def manager_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("manager_dashboard")
        else:
            error = "Invalid username or password"
            return render(request, "login.html", {"error": error})

    return render(request, "login.html")

def manager_logout(request):
    logout(request)
    return redirect("manager_login")

def predict_page(request):
    teams = Team.objects.all().order_by("name")
    return render(request, "predict.html", {"teams": teams})

@csrf_exempt
def predict_api(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    content_type = request.content_type or ""

    if "application/json" in content_type:
        try:
            body = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        home = body.get("home_team")
        away = body.get("away_team")
    else:
        home = request.POST.get("home_team")
        away = request.POST.get("away_team")

    if not home or not away:
        return JsonResponse({"error": "home_team and away_team required"}, status=400)

    try:
        raw = predict_match(home, away)

        # Normalize output for frontend
        response = {
            "home_team": raw["home"],
            "away_team": raw["away"],
            "home_win_prob": round(raw["probs"]["H"] * 100, 1),
            "draw_prob": round(raw["probs"]["D"] * 100, 1),
            "away_win_prob": round(raw["probs"]["A"] * 100, 1),
            "predicted": raw["predicted"],
        }

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required(login_url="manager_login")
def coach_dashboard(request):
    """
    Coach dashboard view — adapted to the new Match model fields.
    Preserves original variable names (y_home_goals, result, matchweek, xg_home, ...)
    by computing/attaching them on each Match instance so templates remain unchanged.
    """
    try:
        coach = Coach.objects.get(user=request.user)
    except Coach.DoesNotExist:
        return render(request, "coach_no_profile.html", {})

    team = coach.team
    today = date.today()

    # --- Upcoming (unplayed) fixtures
    # New model has no 'played' field; treat matches with both goals NULL as unplayed
    upcoming_qs = Match.objects.filter(
        Q(home_team=team) | Q(away_team=team),
        home_goals__isnull=True,
        away_goals__isnull=True,
        date__gte=today
    ).order_by("date", "week")[:5]
    upcoming = list(upcoming_qs)

    # --- Recent played matches (for display)
    recent_qs = Match.objects.filter(
        Q(home_team=team) | Q(away_team=team),
        home_goals__isnull=False,
        away_goals__isnull=False,
        date__lte=today
    ).order_by("-date")[:10]
    recent = list(recent_qs[:5])

    # Attach compatibility attributes on upcoming/recent (so templates referencing old names work)
    def attach_compat_attrs(m):
        # matchweek
        m.matchweek = getattr(m, "week", None)
        # y_home_goals / y_away_goals (old names)
        m.y_home_goals = m.home_goals
        m.y_away_goals = m.away_goals
        # xg compatibility
        m.xg_home = m.home_xg
        m.xg_away = m.away_xg
        # compute result if goals available
        if m.y_home_goals is not None and m.y_away_goals is not None:
            if m.y_home_goals > m.y_away_goals:
                m.result = "H"
            elif m.y_home_goals < m.y_away_goals:
                m.result = "A"
            else:
                m.result = "D"
        else:
            m.result = None
        return m

    for m in upcoming:
        attach_compat_attrs(m)
    for m in recent:
        attach_compat_attrs(m)

    # --- Pick most relevant season (most recent season that has any match for this team)
    season_qs = Match.objects.filter(Q(home_team=team) | Q(away_team=team)).order_by("-season")
    season_to_use = season_qs.first().season if season_qs.exists() else None

    # season matches (all scheduled matches in that season)
    if season_to_use:
        season_matches_qs = Match.objects.filter(
            Q(home_team=team) | Q(away_team=team),
            season=season_to_use
        ).order_by("date")
    else:
        season_matches_qs = Match.objects.none()

    # counts and accumulators (same variables as before)
    played_count = season_matches_qs.filter(home_goals__isnull=False, away_goals__isnull=False).count()

    wins = draws = losses = 0
    goals_for = goals_against = 0
    goals_for_count = goals_against_count = 0
    xg_for = xg_against = 0.0
    xg_for_count = xg_against_count = 0
    clean_sheets = 0

    matches_home = matches_away = 0
    goals_home_for = goals_away_for = 0
    goals_home_for_count = goals_away_for_count = 0
    goals_home_against = goals_away_against = 0
    goals_home_against_count = goals_away_against_count = 0

    xg_home_for = xg_away_for = 0.0
    xg_home_for_count = xg_away_for_count = 0
    xg_home_against = xg_away_against = 0.0
    xg_home_against_count = xg_away_against_count = 0

    wins_home = wins_away = 0
    draws_home = draws_away = 0
    losses_home = losses_away = 0
    clean_sheets_home = clean_sheets_away = 0

    # iterate season matches
    for m in season_matches_qs:
        # attach compatibility attributes for template & logic
        attach_compat_attrs(m)

        is_home = (m.home_team.name == team.name)


        # schedule-level counts (regardless of whether match played)
        if m.home_goals is not None and m.away_goals is not None:
            if is_home:
                matches_home += 1
            else:
                matches_away += 1


        # Goals: use y_home_goals / y_away_goals (mapped to home_goals/away_goals)
        if m.home_goals is not None and m.away_goals is not None:
            gf = m.y_home_goals if is_home else m.y_away_goals
            ga = m.y_away_goals if is_home else m.y_home_goals

            goals_for += gf
            goals_for_count += 1
            goals_against += ga
            goals_against_count += 1

            if is_home:
                goals_home_for += gf
                goals_home_for_count += 1
                goals_home_against += ga
                goals_home_against_count += 1
            else:
                goals_away_for += gf
                goals_away_for_count += 1
                goals_away_against += ga
                goals_away_against_count += 1

            # clean sheet
            if ga == 0:
                clean_sheets += 1
                if is_home:
                    clean_sheets_home += 1
                else:
                    clean_sheets_away += 1

        # xG: use home_xg / away_xg fields
        if is_home:
            if m.home_xg is not None:
                xg_for += float(m.home_xg)
                xg_for_count += 1
                xg_home_for += float(m.home_xg)
                xg_home_for_count += 1
            if m.away_xg is not None:
                xg_against += float(m.away_xg)
                xg_against_count += 1
                xg_home_against += float(m.away_xg)
                xg_home_against_count += 1
        else:
            if m.away_xg is not None:
                xg_for += float(m.away_xg)
                xg_for_count += 1
                xg_away_for += float(m.away_xg)
                xg_away_for_count += 1
            if m.home_xg is not None:
                xg_against += float(m.home_xg)
                xg_against_count += 1
                xg_away_against += float(m.home_xg)
                xg_away_against_count += 1

        # Results: compute from goals (only if both goals present)
        if m.result in ("H", "D", "A"):
            if m.result == "D":
                draws += 1
                if is_home:
                    draws_home += 1
                else:
                    draws_away += 1
            else:
                team_won = (m.result == "H" and is_home) or (m.result == "A" and not is_home)
                if team_won:
                    wins += 1
                    if is_home:
                        wins_home += 1
                    else:
                        wins_away += 1
                else:
                    losses += 1
                    if is_home:
                        losses_home += 1
                    else:
                        losses_away += 1

    # computed metrics
    goal_diff = (goals_for - goals_against) if (goals_for_count or goals_against_count) else None
    points = wins * 3 + draws
    ppg = round(points / played_count, 2) if played_count else None

    avg_xg_for = round(xg_for / xg_for_count, 2) if xg_for_count else None
    avg_xg_against = round(xg_against / xg_against_count, 2) if xg_against_count else None

    conversion_rate = None
    if xg_for > 0 and goals_for_count:
        conversion_rate = round(goals_for / xg_for, 2)
    elif xg_for_count and xg_for == 0 and goals_for_count:
        conversion_rate = 0.0
    else:
        conversion_rate = None

    # last-5 form (use only matches with goals present)
    last5_qs = season_matches_qs.filter(home_goals__isnull=False, away_goals__isnull=False).order_by("-date")[:5]
    last5_matches = list(last5_qs)
    form_points = []
    form_labels = []
    for m in reversed(last5_matches):  # oldest -> newest
        # ensure compatibility attributes
        attach_compat_attrs(m)
        is_home = (m.home_team_id == team.id)
        if m.result == "D":
            pts = 1
            res_label = "D"
        else:
            pts = 3 if ((m.result == "H" and is_home) or (m.result == "A" and not is_home)) else 0
            res_label = "W" if pts == 3 else "L"
        form_points.append(pts)
        opp = m.away_team.name if is_home else m.home_team.name
        venue = "H" if is_home else "A"
        form_labels.append(f"{m.date.strftime('%d %b')} ({venue}) vs {opp}")

    # streak (consecutive same result from most recent played matches)
    streak = 0
    streak_type = None
    played_ordered_qs = season_matches_qs.filter(home_goals__isnull=False, away_goals__isnull=False).order_by("-date")
    for m in played_ordered_qs:
        attach_compat_attrs(m)
        is_home = (m.home_team_id == team.id)
        if m.result == "D":
            cur = "D"
        else:
            cur = "W" if ((m.result == "H" and is_home) or (m.result == "A" and not is_home)) else "L"
        if streak_type is None:
            streak_type = cur
            streak = 1
        elif cur == streak_type:
            streak += 1
        else:
            break

    # recent goals vs xG arrays (last up to 5 played matches)
    recent_played_for_charts = list(season_matches_qs.filter(home_goals__isnull=False, away_goals__isnull=False).order_by("-date")[:5])[::-1]
    recent_goals = []
    recent_xg = []
    recent_labels = []
    for m in recent_played_for_charts:
        attach_compat_attrs(m)
        is_home = (m.home_team_id == team.id)
        gf = (m.y_home_goals if is_home else m.y_away_goals)
        gf_val = int(gf) if gf is not None else None
        xg_val = (m.home_xg if is_home else m.away_xg)
        recent_goals.append(gf_val)
        recent_xg.append(round(float(xg_val), 2) if xg_val is not None else None)
        recent_labels.append(m.date.strftime("%d %b"))

    # next match prediction (if upcoming exists)
    next_pred = None
    if upcoming:
        nxt = upcoming[0]
        try:
            if predict_match:
                raw = predict_match(nxt.home_team.name, nxt.away_team.name)
                # raw likely has format: {"home":..., "away":..., "probs": {"H":..,"D":..,"A":..}, "predicted": "H"}
                probs = raw.get("probs", {})
                # produce both shapes so templates expecting either style will work
                next_pred = {
                    "probs": {
                        "H": float(probs.get("H", 0.0)),
                        "D": float(probs.get("D", 0.0)),
                        "A": float(probs.get("A", 0.0)),
                    },
                    "predicted": raw.get("predicted")
                }
                # convenience percent keys (0-100) for older templates
                next_pred["home_win_prob"] = round(next_pred["probs"]["H"] * 100, 1) if next_pred["probs"]["H"] <= 1 else round(next_pred["probs"]["H"], 1)
                next_pred["draw_prob"] = round(next_pred["probs"]["D"] * 100, 1) if next_pred["probs"]["D"] <= 1 else round(next_pred["probs"]["D"], 1)
                next_pred["away_win_prob"] = round(next_pred["probs"]["A"] * 100, 1) if next_pred["probs"]["A"] <= 1 else round(next_pred["probs"]["A"], 1)
            else:
                next_pred = None
        except Exception:
            next_pred = None

    # Prepare chart payloads
    chart_form = {"labels": form_labels, "data": form_points}
    chart_goals = [g if g is not None else 0 for g in recent_goals]
    chart_xg = [x if x is not None else 0 for x in recent_xg]
    chart_goals_xg = {"labels": recent_labels, "goals": chart_goals, "xg": chart_xg}

    # Context
    context = {
        "coach": coach,
        "team": team,
        "upcoming": upcoming,
        "recent": recent,
        "season": season_to_use,
        "played_count": played_count,
        "wins": wins, "draws": draws, "losses": losses,
        "points": points, "ppg": ppg,
        "goals_for": goals_for if goals_for_count else None,
        "goals_against": goals_against if goals_against_count else None,
        "goals_for_count": goals_for_count, "goals_against_count": goals_against_count,
        "xg_for_total": round(xg_for, 2) if xg_for_count else None,
        "xg_against_total": round(xg_against, 2) if xg_against_count else None,
        "xg_for_count": xg_for_count, "xg_against_count": xg_against_count,
        "avg_xg_for": avg_xg_for, "avg_xg_against": avg_xg_against,
        "conversion_rate": conversion_rate,
        "clean_sheets": clean_sheets,
        "matches_home": matches_home, "matches_away": matches_away,
        "goals_home_for": goals_home_for if goals_home_for_count else None,
        "goals_away_for": goals_away_for if goals_away_for_count else None,
        "goals_home_against": goals_home_against if goals_home_against_count else None,
        "goals_away_against": goals_away_against if goals_away_against_count else None,
        "xg_home_for": round(xg_home_for, 2) if xg_home_for_count else None,
        "xg_away_for": round(xg_away_for, 2) if xg_away_for_count else None,
        "xg_home_against": round(xg_home_against, 2) if xg_home_against_count else None,
        "xg_away_against": round(xg_away_against, 2) if xg_away_against_count else None,
        "wins_home": wins_home, "wins_away": wins_away,
        "draws_home": draws_home, "draws_away": draws_away,
        "losses_home": losses_home, "losses_away": losses_away,
        "clean_sheets_home": clean_sheets_home, "clean_sheets_away": clean_sheets_away,
        "streak": streak, "streak_type": streak_type,
        "chart_form": json.dumps(chart_form),
        "chart_goals_xg": json.dumps(chart_goals_xg),
        "next_pred": next_pred,
        "recent_goals_raw": recent_goals,
        "recent_xg_raw": recent_xg,
        "goal_diff": goals_for - goals_against,
    }

    return render(request, "coach_dashboard.html", context)


def get_match_result(m):
    """
    Derive match result from goals.
    Returns 'H', 'D', 'A', or None.
    """
    if m.home_goals is None or m.away_goals is None:
        return None
    if m.home_goals > m.away_goals:
        return "H"
    if m.home_goals < m.away_goals:
        return "A"
    return "D"

def team_compare(request, match_id):
    """
    Compare home vs away for the given match id.
    Produces raw season stats + normalized radar values (0..1).
    Compatible with the NEW Match model.
    """
    match = get_object_or_404(Match, id=match_id)
    season = match.season

    # ---------- helpers ----------
    def get_result(m):
        if m.home_goals is None or m.away_goals is None:
            return None
        if m.home_goals > m.away_goals:
            return "H"
        elif m.home_goals < m.away_goals:
            return "A"
        return "D"

    def team_season_stats(team, season):
        """
        Compute raw season stats for `team` in `season` using played matches only.
        Calculation logic preserved exactly.
        """
        qs = Match.objects.filter(
            Q(home_team=team) | Q(away_team=team),
            season=season,
            home_goals__isnull=False,
            away_goals__isnull=False
        )

        matches = 0
        points = 0
        goals_for = 0.0
        goals_against = 0.0
        gf_count = 0
        ga_count = 0
        xg_for = 0.0
        xg_against = 0.0
        xg_for_count = 0
        xg_against_count = 0
        clean_sheets = 0

        for m in qs:
            matches += 1
            is_home = (m.home_team_id == team.id)
            res = get_result(m)

            # ---- Goals ----
            gf = m.home_goals if is_home else m.away_goals
            ga = m.away_goals if is_home else m.home_goals

            if gf is not None:
                goals_for += float(gf)
                gf_count += 1
            if ga is not None:
                goals_against += float(ga)
                ga_count += 1
            if ga == 0:
                clean_sheets += 1

            # ---- xG ----
            if is_home:
                if m.home_xg is not None:
                    xg_for += float(m.home_xg)
                    xg_for_count += 1
                if m.away_xg is not None:
                    xg_against += float(m.away_xg)
                    xg_against_count += 1
            else:
                if m.away_xg is not None:
                    xg_for += float(m.away_xg)
                    xg_for_count += 1
                if m.home_xg is not None:
                    xg_against += float(m.home_xg)
                    xg_against_count += 1

            # ---- Points (UNCHANGED LOGIC) ----
            if res == "D":
                points += 1
            elif res in ("H", "A"):
                team_won = (res == "H" and is_home) or (res == "A" and not is_home)
                points += 3 if team_won else 0

        # ---- Derived metrics ----
        ppg = round(points / matches, 3) if matches > 0 else None
        avg_gf = round(goals_for / gf_count, 3) if gf_count > 0 else None
        avg_ga = round(goals_against / ga_count, 3) if ga_count > 0 else None
        avg_xg_for = round(xg_for / xg_for_count, 3) if xg_for_count > 0 else None
        avg_xg_against = round(xg_against / xg_against_count, 3) if xg_against_count > 0 else None
        clean_sheet_pct = round((clean_sheets / matches) * 100, 2) if matches > 0 else None

        conversion = None
        if xg_for > 0:
            conversion = round(goals_for / xg_for, 3)
        elif xg_for == 0 and goals_for > 0:
            conversion = 0.0

        return {
            "team": team,
            "matches": matches,
            "points": points,
            "ppg": ppg,
            "goals_for": int(goals_for) if gf_count > 0 else None,
            "goals_against": int(goals_against) if ga_count > 0 else None,
            "avg_gf": avg_gf,
            "avg_ga": avg_ga,
            "xg_for": round(xg_for, 3) if xg_for_count > 0 else None,
            "xg_against": round(xg_against, 3) if xg_against_count > 0 else None,
            "avg_xg_for": avg_xg_for,
            "avg_xg_against": avg_xg_against,
            "clean_sheets": clean_sheets,
            "clean_sheet_pct": clean_sheet_pct,
            "conversion": conversion
        }

    # ---------- teams ----------
    home_team = match.home_team
    away_team = match.away_team

    home_stats = team_season_stats(home_team, season)
    away_stats = team_season_stats(away_team, season)

    # ---------- radar metrics ----------
    def safe_num(x):
        return float(x) if x is not None else 0.0

    home_raw = [
        safe_num(home_stats["avg_gf"]),
        safe_num(home_stats["avg_ga"]),
        safe_num(home_stats["avg_xg_for"]),
        safe_num(home_stats["avg_xg_against"]),
        safe_num(home_stats["ppg"]),
        safe_num(home_stats["clean_sheet_pct"]),
        safe_num(home_stats["conversion"]),
    ]

    away_raw = [
        safe_num(away_stats["avg_gf"]),
        safe_num(away_stats["avg_ga"]),
        safe_num(away_stats["avg_xg_for"]),
        safe_num(away_stats["avg_xg_against"]),
        safe_num(away_stats["ppg"]),
        safe_num(away_stats["clean_sheet_pct"]),
        safe_num(away_stats["conversion"]),
    ]

    metric_ranges = [
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.0, 100.0),
        (0.0, 2.0),
    ]

    def norm_value(val, vmin, vmax):
        if val is None:
            return 0.0
        v = max(min(float(val), vmax), vmin)
        return (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0

    normalized_home = []
    normalized_away = []

    for i in range(len(home_raw)):
        vmin, vmax = metric_ranges[i]
        na = norm_value(home_raw[i], vmin, vmax)
        nb = norm_value(away_raw[i], vmin, vmax)

        if i in (1, 3):  # invert "against"
            na = 1.0 - na
            nb = 1.0 - nb

        normalized_home.append(round(na, 3))
        normalized_away.append(round(nb, 3))

    radar_labels = [
        "Avg Goals For",
        "Avg Goals Against",
        "Avg xG For",
        "Avg xG Against",
        "Points / game",
        "Clean Sheet %",
        "Conversion",
    ]

    context = {
        "match": match,
        "season": season,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "radar_labels": json.dumps(radar_labels),
        "home_metrics_raw": json.dumps(home_raw),
        "away_metrics_raw": json.dumps(away_raw),
        "home_metrics_norm": json.dumps(normalized_home),
        "away_metrics_norm": json.dumps(normalized_away),
    }

    return render(request, "team_compare.html", context)


def minute_to_int(m):
    if not m:
        return None
    m = str(m)
    if "+" in m:
        a, b = m.split("+", 1)
        return int(a) + int(b)
    return int(m)


def match_detail(request, match_id):
    match = get_object_or_404(Match, id=match_id)

    # -------- result text --------
    if match.home_goals is None or match.away_goals is None:
        result_text = "Unplayed"
    elif match.home_goals > match.away_goals:
        result_text = f"{match.home_team} won"
    elif match.home_goals < match.away_goals:
        result_text = f"{match.away_team} won"
    else:
        result_text = "Draw"

    # -------- xG delta --------
    home_overperformance = (
        round(match.home_goals - match.home_xg, 2)
        if match.home_goals is not None and match.home_xg is not None
        else None
    )
    away_overperformance = (
        round(match.away_goals - match.away_xg, 2)
        if match.away_goals is not None and match.away_xg is not None
        else None
    )

    # -------- FBref (EXACT WORKING CALL) --------
    fb_events = []
    home_lineup = []
    away_lineup = []
    fb_error = None

    try:
        fbref = sd.FBref(
            leagues="ENG-Premier League",
            seasons=2025,
        )

        events_df = fbref.read_events(match_id=match.game_id)
        lineup_df = fbref.read_lineup(match_id=match.game_id)

        # events
        if events_df is not None and not events_df.empty:
            fb_events = events_df.to_dict(orient="records")
            for e in fb_events:
                e["minute_int"] = minute_to_int(e.get("minute"))

        # lineup
        if lineup_df is not None and not lineup_df.empty:
            lineup = lineup_df.to_dict(orient="records")

            home_team_norm = normalize_team_name(match.home_team.name)
            away_team_norm = normalize_team_name(match.away_team.name)

            for p in lineup:
                fb_team_norm = normalize_team_name(p.get("team"))

                if fb_team_norm == home_team_norm:
                    home_lineup.append(p)
                elif fb_team_norm == away_team_norm:
                    away_lineup.append(p)


    
    except Exception as exc:
        fb_error = str(exc)
    
    context = {
        "match": match,
        "result_text": result_text,

        "home_overperformance": home_overperformance,
        "away_overperformance": away_overperformance,

        "fb_events": fb_events,
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
        "fb_error": fb_error,
    }

    return render(request, "match_detail.html", context)

def admin_login(request):

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is None:
            messages.error(request, "Invalid username or password")
            return render(request, "admin_login.html")

        if not user.is_superuser:
            messages.error(request, "You are not authorized as admin")
            return render(request, "admin_login.html")

        login(request, user)
        return redirect("admin_dashboard")

    return render(request, "admin_login.html")

@login_required(login_url="admin_login")
def admin_dashboard(request):

    if not request.user.is_superuser:
        messages.error(request, "Unauthorized access")
        return redirect("admin_login")

    return render(request, "admin_dashboard.html")


@user_passes_test(superuser_required)
def admin_dashboard(request):
    return render(request, "admin_dashboard.html")


@login_required
@user_passes_test(superuser_required)
def admin_run_pipeline(request):
    """Start the full pipeline"""
    if request.method == "POST":
        if pipeline_status["running"]:
            return JsonResponse({"error": "Pipeline already running"}, status=400)
        
        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline_async)
        thread.daemon = True
        thread.start()
        
        return JsonResponse({"status": "started"})
    
    return JsonResponse({"error": "Invalid request"}, status=400)

@login_required
@user_passes_test(superuser_required)
def admin_pipeline_status(request):
    """Get current pipeline status"""
    return JsonResponse(pipeline_status)


def run_player_stats_async():
    """Run player stats update in background"""
    global player_stats_status
    
    player_stats_status["running"] = True
    player_stats_status["error"] = None
    player_stats_status["logs"] = []
    
    project_root = Path(settings.BASE_DIR)
    season_update_script = project_root / "ml_pipeline" / "update" / "update_player_season_stats.py"
    match_update_script = project_root / "ml_pipeline" / "update" / "update_player_match_summary_2526.py"
    
    steps = [
        {
            "name": "Update Season CSV",
            "cmd": ["python3", str(season_update_script)],
            "cwd": str(project_root)
        },
        {
            "name": "Import Season Stats",
            "cmd": ["python", "manage.py", "import_player_season_stats"],
            "cwd": str(project_root)
        },
        {
            "name": "Update Match Summary CSV",
            "cmd": ["python3", str(match_update_script)],
            "cwd": str(project_root)
        },
        {
            "name": "Import Match Stats",
            "cmd": ["python", "manage.py", "import_player_match_stats"],
            "cwd": str(project_root)
        }
    ]
    
    player_stats_status["steps"] = [{"name": step["name"], "status": "pending"} for step in steps]
    
    try:
        for i, step in enumerate(steps):
            player_stats_status["current_step"] = i
            player_stats_status["steps"][i]["status"] = "running"
            player_stats_status["logs"].append(f"⏳ {step['name']}...")
            
            result = subprocess.run(
                step["cmd"],
                cwd=step["cwd"],
                capture_output=True,
                text=True,
                check=True
            )
            
            player_stats_status["logs"].append(f"✓ {step['name']} completed")
            player_stats_status["steps"][i]["status"] = "completed"
            time.sleep(0.5)
        
        player_stats_status["current_step"] = None
        player_stats_status["logs"].append("✓ Player stats update completed successfully!")
        
    except subprocess.CalledProcessError as e:
        player_stats_status["error"] = f"Error in {steps[i]['name']}: {e.stderr}"
        player_stats_status["steps"][i]["status"] = "error"
        player_stats_status["logs"].append(f"✗ Update failed: {e.stderr}")
    
    finally:
        player_stats_status["running"] = False

@login_required
@user_passes_test(superuser_required)
def admin_run_player_stats_update(request):
    """Start player stats update in background"""
    if request.method == "POST":
        if player_stats_status["running"]:
            return JsonResponse({"error": "Player stats update already running"}, status=400)
        
        # Start update in background thread
        thread = threading.Thread(target=run_player_stats_async)
        thread.daemon = True
        thread.start()
        
        return JsonResponse({"status": "started"})
    
    return JsonResponse({"error": "Invalid request"}, status=400)

@login_required
@user_passes_test(superuser_required)
def admin_player_stats_status(request):
    """Get current player stats update status"""
    return JsonResponse(player_stats_status)


@login_required
def player_analysis(request):
    """
    Comprehensive squad analysis with advanced metrics and insights
    """
    coach = Coach.objects.select_related("team").get(user=request.user)
    team = coach.team
    season = "25/26"

    players = PlayerSeasonStat.objects.filter(
        team=team,
        season=season
    ).select_related("team")

    player_rows = []
    
    # Squad-wide metrics for comparison
    squad_avg_goals_p90 = players.aggregate(avg=Avg('p90_goals'))['avg'] or 0
    squad_avg_assists_p90 = players.aggregate(avg=Avg('p90_assists'))['avg'] or 0
    squad_avg_xg_p90 = players.aggregate(avg=Avg('p90_xg'))['avg'] or 0

    for p in players:
        mins = p.playing_time_min or 0
        apps = p.playing_time_mp or 0
        starts = p.playing_time_starts or 0
        min_90s = p.playing_time_90s or 0.01  # Avoid division by zero

        # =====================================================
        # AVAILABILITY & RELIABILITY
        # =====================================================
        availability = round((starts / apps) * 100, 1) if apps else 0
        minutes_per_app = round(mins / apps, 1) if apps else 0
        
        durability_score = min_90s
        durability_rating = (
            "Exceptional" if durability_score >= 20 else
            "High" if durability_score >= 15 else
            "Moderate" if durability_score >= 10 else
            "Low"
        )

        # =====================================================
        # ATTACKING OUTPUT
        # =====================================================
        goals = p.goals or 0
        assists = p.assists or 0
        xg = p.xg or 0
        xag = p.xag or 0
        
        # Goal efficiency
        goal_delta = goals - xg
        finishing_efficiency = round((goals / xg * 100) if xg > 0 else 0, 1)
        finishing_rating = (
            "Clinical" if finishing_efficiency >= 120 else
            "Efficient" if finishing_efficiency >= 90 else
            "Average" if finishing_efficiency >= 70 else
            "Poor"
        )
        
        # Assist efficiency
        assist_delta = assists - xag
        creative_efficiency = round((assists / xag * 100) if xag > 0 else 0, 1)
        
        # Per 90 output
        goals_p90 = p.p90_goals or 0
        assists_p90 = p.p90_assists or 0
        goal_contributions_p90 = p.p90_goals_assists or 0
        
        # Output vs squad average
        goals_vs_avg = ((goals_p90 - squad_avg_goals_p90) / squad_avg_goals_p90 * 100) if squad_avg_goals_p90 > 0 else 0
        assists_vs_avg = ((assists_p90 - squad_avg_assists_p90) / squad_avg_assists_p90 * 100) if squad_avg_assists_p90 > 0 else 0

        # =====================================================
        # CREATIVITY & PROGRESSION
        # =====================================================
        progressive_passes = p.progressive_passes or 0
        progressive_carries = p.progressive_carries or 0
        key_passes = p.key_passes or 0
        passes_final_third = p.passes_into_final_third or 0
        passes_penalty_area = p.passes_into_penalty_area or 0
        
        progression_score = progressive_passes + progressive_carries
        progression_p90 = round(progression_score / min_90s, 2)
        
        creativity_score = key_passes + passes_penalty_area
        creativity_p90 = round(creativity_score / min_90s, 2)
        
        involvement_rating = (
            "Elite" if progression_p90 >= 10 else
            "High" if progression_p90 >= 6 else
            "Moderate" if progression_p90 >= 3 else
            "Low"
        )

        # =====================================================
        # PASSING QUALITY
        # =====================================================
        pass_completion = p.pass_completion_pct or 0
        passes_completed = p.passes_completed or 0
        passes_attempted = p.passes_attempted or 0
        
        passing_volume_p90 = round(passes_attempted / min_90s, 1)
        
        passing_rating = (
            "Excellent" if pass_completion >= 85 else
            "Good" if pass_completion >= 75 else
            "Average" if pass_completion >= 65 else
            "Poor"
        )

        # =====================================================
        # DEFENSIVE CONTRIBUTION
        # =====================================================
        tackles = p.tackles or 0
        interceptions = p.interceptions or 0
        clearances = p.clearances or 0
        blocks = p.blocks or 0
        
        defensive_actions = tackles + interceptions + clearances + blocks
        defensive_actions_p90 = round(defensive_actions / min_90s, 2)
        
        defensive_rating = (
            "Elite" if defensive_actions_p90 >= 8 else
            "Strong" if defensive_actions_p90 >= 5 else
            "Moderate" if defensive_actions_p90 >= 3 else
            "Limited"
        )

        # =====================================================
        # DISCIPLINE
        # =====================================================
        yellows = p.yellow_cards or 0
        reds = p.red_cards or 0
        total_cards = yellows + reds
        cards_per_90 = round(total_cards / min_90s, 2)
        
        discipline_rating = (
            "Concerning" if cards_per_90 >= 0.5 or reds >= 1 else
            "Monitor" if cards_per_90 >= 0.3 else
            "Good" if cards_per_90 >= 0.15 else
            "Excellent"
        )

        # =====================================================
        # OVERALL IMPACT SCORE
        # =====================================================
        # Weighted composite score
        impact_score = (
            (goal_contributions_p90 * 3) +  # Attacking output (highest weight)
            (progression_p90 * 1.5) +        # Ball progression
            (defensive_actions_p90 * 1.2) +  # Defensive work
            (creativity_p90 * 2) +            # Chance creation
            (pass_completion / 20)            # Passing accuracy
        )
        
        impact_tier = (
            "World Class" if impact_score >= 15 else
            "Elite" if impact_score >= 10 else
            "Reliable" if impact_score >= 6 else
            "Rotation" if impact_score >= 3 else
            "Fringe"
        )

        # =====================================================
        # RECENT FORM (Last 5 matches)
        # =====================================================
        recent_matches = PlayerMatchStat.objects.filter(
            player=p
        ).order_by("-match__date_parsed")[:5]

        recent_minutes = sum(m.minutes or 0 for m in recent_matches)
        recent_goals = sum(m.goals or 0 for m in recent_matches)
        recent_assists = sum(m.assists or 0 for m in recent_matches)
        
        form_score = recent_goals + (recent_assists * 0.8)
        form_rating = (
            "Excellent" if form_score >= 3 else
            "Good" if form_score >= 2 else
            "Average" if form_score >= 1 else
            "Declining"
        )

        # =====================================================
        # VALUE ANALYSIS
        # =====================================================
        # Minutes utilization
        utilization_pct = round((mins / (apps * 90)) * 100, 1) if apps > 0 else 0
        
        # Player importance score
        importance_score = (
            (starts / apps * 100) if apps > 0 else 0 +
            (mins / 2000 * 100) if mins > 0 else 0
        ) / 2

        # =====================================================
        # TACTICAL FLAGS & INSIGHTS
        # =====================================================
        flags = []
        insights = []
        
        # Performance flags
        if goal_delta < -2:
            flags.append("Underperforming xG significantly")
        elif goal_delta > 2:
            insights.append("Overperforming xG - clinical finisher")
            
        if assist_delta < -1.5:
            flags.append("Creating but not converting to assists")
            
        # Availability flags
        if availability < 60:
            flags.append("Low starting rate - rotation risk")
        if mins < 500 and impact_tier in ["Elite", "World Class"]:
            flags.append("High-quality player underutilized")
            
        # Discipline flags
        if reds >= 1:
            flags.append("Red card received - suspension served")
        if cards_per_90 >= 0.5:
            flags.append("High card rate - discipline concern")
            
        # Form flags
        if form_rating == "Declining" and mins >= 1000:
            flags.append("Form drop despite regular playing time")
            
        # Output flags
        if goal_contributions_p90 < 0.3 and p.position in ["FW", "MF"]:
            flags.append("Low attacking output for position")
            
        # Efficiency insights
        if finishing_efficiency >= 120:
            insights.append("Clinical finishing - exceeding expectations")
        if creative_efficiency >= 110:
            insights.append("High assist conversion rate")
        if pass_completion >= 90:
            insights.append("Elite passing accuracy")
        if defensive_actions_p90 >= 8:
            insights.append("Strong defensive contribution")

        player_rows.append({
            # Identity
            "player_id": p.player_id,
            "name": p.player_name,
            "position": p.position,
            "age": p.age,
            "nation": p.nation,
            
            # Playing Time
            "minutes": mins,
            "appearances": apps,
            "starts": starts,
            "minutes_per_app": minutes_per_app,
            "availability": availability,
            "utilization": utilization_pct,
            
            # Durability
            "durability_score": round(durability_score, 1),
            "durability": durability_rating,
            
            # Attacking
            "goals": goals,
            "assists": assists,
            "goals_p90": round(goals_p90, 2),
            "assists_p90": round(assists_p90, 2),
            "goal_contributions": goals + assists,
            "goal_contributions_p90": round(goal_contributions_p90, 2),
            "xg": round(xg, 2),
            "xag": round(xag, 2),
            "goal_delta": round(goal_delta, 2),
            "finishing_efficiency": finishing_efficiency,
            "finishing": finishing_rating,
            "creative_efficiency": creative_efficiency,
            "goals_vs_avg": round(goals_vs_avg, 1),
            
            # Progression & Creativity
            "progression_score": progression_score,
            "progression_p90": progression_p90,
            "creativity_score": creativity_score,
            "creativity_p90": creativity_p90,
            "involvement": involvement_rating,
            "key_passes": key_passes,
            
            # Passing
            "pass_completion": round(pass_completion, 1),
            "passing_volume_p90": passing_volume_p90,
            "passing": passing_rating,
            
            # Defensive
            "defensive_actions": defensive_actions,
            "defensive_actions_p90": defensive_actions_p90,
            "defensive": defensive_rating,
            "tackles": tackles,
            "interceptions": interceptions,
            
            # Discipline
            "yellows": yellows,
            "reds": reds,
            "cards_per_90": cards_per_90,
            "discipline": discipline_rating,
            
            # Impact
            "impact_score": round(impact_score, 2),
            "impact": impact_tier,
            
            # Form
            "recent_form": form_rating,
            "form_score": round(form_score, 1),
            
            # Flags & Insights
            "flags": flags,
            "insights": insights,
        })

    # Sort by impact score descending
    player_rows.sort(key=lambda x: x['impact_score'], reverse=True)

    # Squad statistics
    squad_stats = {
        "total_players": len(player_rows),
        "avg_age": round(statistics.mean([float(p['age'].split('-')[0]) for p in player_rows if p['age']]), 1),
        "total_goals": sum(p['goals'] for p in player_rows),
        "total_assists": sum(p['assists'] for p in player_rows),
        "elite_count": sum(1 for p in player_rows if p['impact'] in ["World Class", "Elite"]),
        "avg_pass_completion": round(statistics.mean([p['pass_completion'] for p in player_rows if p['pass_completion']]), 1),
        "disciplinary_issues": sum(1 for p in player_rows if p['discipline'] in ["Concerning", "Monitor"]),
    }

    context = {
        "team": team,
        "season": season,
        "players": player_rows,
        "squad_stats": squad_stats,
    }

    return render(request, "player_analysis.html", context)

PERCENTILE_STATS = [
    # Attacking output
    ("Goals per 90", "p90_goals", "attack"),
    ("Assists per 90", "p90_assists", "attack"),
    ("xG per 90", "p90_xg", "attack"),
    ("xA per 90", "p90_xag", "attack"),
    ("Goal Contributions per 90", "p90_goals_assists", "attack"),
    
    # Progression & involvement
    ("Progressive Passes", "progressive_passes", "progression"),
    ("Progressive Carries", "progressive_carries", "progression"),
    ("Key Passes", "key_passes", "creativity"),
    ("Passes into Final Third", "passes_into_final_third", "creativity"),
    ("Passes into Penalty Area", "passes_into_penalty_area", "creativity"),
    
    # Passing quality
    ("Pass Completion %", "pass_completion_pct", "passing"),
    ("Short Pass Completion %", "short_pass_completion_pct", "passing"),
    ("Medium Pass Completion %", "medium_pass_completion_pct", "passing"),
    ("Long Pass Completion %", "long_pass_completion_pct", "passing"),
    
    # Defensive contribution
    ("Tackles + Interceptions", "tackles_interceptions", "defense"),
    ("Interceptions", "interceptions", "defense"),
    ("Blocks", "blocks", "defense"),
    ("Clearances", "clearances", "defense"),
    ("Tackles Won", "tackles_won", "defense"),
    
    # Discipline
    ("Yellow Cards", "yellow_cards", "discipline"),
    ("Errors Leading to Shot", "errors", "discipline"),
]

LESS_IS_BETTER = ["yellow_cards", "red_cards", "errors"]

def percentile_rank(value, pool, flip=False):
    """Calculate percentile rank for a value within a pool"""
    if value is None or not pool:
        return None
    arr = np.array([p for p in pool if p is not None], dtype=float)
    if len(arr) == 0:
        return None
    val = float(value)
    if flip:
        return round((np.sum(arr >= val) / len(arr)) * 100, 1)
    return round((np.sum(arr <= val) / len(arr)) * 100, 1)


@login_required
def player_detail(request, player_id):
    player = get_object_or_404(PlayerSeasonStat, player_id=player_id)
    
    # Permission check
    if not request.user.is_superuser:
        coach = get_object_or_404(Coach, user=request.user)
        if coach.team_id != player.team_id:
            return HttpResponseBadRequest("Not allowed")

    season = player.season
    mins = player.playing_time_min or 0
    min_90s = player.playing_time_90s or 0.01
    apps = player.playing_time_mp or 0
    starts = player.playing_time_starts or 0

    # =====================================================
    # PLAYER OVERVIEW
    # =====================================================
    overview = {
        "name": player.player_name,
        "team": player.team.name,
        "league": player.league.name,
        "season": player.season,
        "position": player.position,
        "age": player.age.split('-')[0] if player.age else "—",
        "nation": player.nation,
        "minutes": mins,
        "starts": starts,
        "appearances": apps,
        "minutes_per_game": round(mins / apps, 1) if apps > 0 else 0,
        "start_percentage": round((starts / apps * 100), 1) if apps > 0 else 0,
    }

    # Usage classification
    if starts >= apps * 0.8:
        overview["role_type"] = "Key Player"
        overview["role_desc"] = "Regular starter with consistent minutes"
    elif starts >= apps * 0.5:
        overview["role_type"] = "Rotation"
        overview["role_desc"] = "Regular rotation with mixed starts"
    elif mins >= 500:
        overview["role_type"] = "Impact Sub"
        overview["role_desc"] = "Frequent substitute appearances"
    else:
        overview["role_type"] = "Fringe"
        overview["role_desc"] = "Limited playing time"

    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================
    metrics = {
        # Attacking
        "goals": player.goals or 0,
        "assists": player.assists or 0,
        "goal_contributions": (player.goals or 0) + (player.assists or 0),
        "goals_p90": round(player.p90_goals or 0, 2),
        "assists_p90": round(player.p90_assists or 0, 2),
        "goal_contributions_p90": round(player.p90_goals_assists or 0, 2),
        "xg": round(player.xg or 0, 2),
        "xag": round(player.xag or 0, 2),
        "goal_delta": round((player.goals or 0) - (player.xg or 0), 2),
        "assist_delta": round((player.assists or 0) - (player.xag or 0), 2),
        
        # Progression
        "progressive_passes": player.progressive_passes or 0,
        "progressive_carries": player.progressive_carries or 0,
        "progressive_actions_p90": round(((player.progressive_passes or 0) + (player.progressive_carries or 0)) / min_90s, 2),
        
        # Creativity
        "key_passes": player.key_passes or 0,
        "passes_final_third": player.passes_into_final_third or 0,
        "passes_penalty_area": player.passes_into_penalty_area or 0,
        
        # Passing
        "pass_completion": round(player.pass_completion_pct or 0, 1),
        "passes_completed": player.passes_completed or 0,
        "passes_attempted": player.passes_attempted or 0,
        
        # Defensive
        "tackles": player.tackles or 0,
        "interceptions": player.interceptions or 0,
        "clearances": player.clearances or 0,
        "blocks": player.blocks or 0,
        "defensive_actions": (player.tackles or 0) + (player.interceptions or 0),
        "defensive_actions_p90": round(((player.tackles or 0) + (player.interceptions or 0)) / min_90s, 2),
        
        # Discipline
        "yellows": player.yellow_cards or 0,
        "reds": player.red_cards or 0,
        "errors": player.errors or 0,
    }

    # =====================================================
    # STRENGTHS & DEVELOPMENT AREAS
    # =====================================================
    strengths = []
    development = []
    
    # Finishing
    if metrics["goal_delta"] >= 2:
        strengths.append({
            "title": "Clinical Finishing",
            "detail": f"+{metrics['goal_delta']} goals above xG - exceptional conversion"
        })
    elif metrics["goal_delta"] <= -2:
        development.append({
            "title": "Finishing Efficiency",
            "detail": f"{metrics['goal_delta']} goals below xG - conversion needs improvement"
        })
    
    # Creativity
    if metrics["assist_delta"] >= 1.5:
        strengths.append({
            "title": "Elite Playmaking",
            "detail": f"+{metrics['assist_delta']} assists above xA - creates quality chances"
        })
    
    # Progression
    if metrics["progressive_actions_p90"] >= 8:
        strengths.append({
            "title": "Ball Progression",
            "detail": f"{metrics['progressive_actions_p90']}/90 progressive actions - strong carrier"
        })
    
    # Passing
    if metrics["pass_completion"] >= 88:
        strengths.append({
            "title": "Passing Accuracy",
            "detail": f"{metrics['pass_completion']}% completion - reliable in possession"
        })
    elif metrics["pass_completion"] < 75 and metrics["pass_completion"] > 0:
        development.append({
            "title": "Passing Accuracy",
            "detail": f"{metrics['pass_completion']}% completion - room for improvement"
        })
    
    # Defensive work
    if metrics["defensive_actions_p90"] >= 6:
        strengths.append({
            "title": "Defensive Contribution",
            "detail": f"{metrics['defensive_actions_p90']}/90 tackles+interceptions - strong defender"
        })
    
    # Discipline
    if metrics["yellows"] + metrics["reds"] >= 6:
        development.append({
            "title": "Discipline",
            "detail": f"{metrics['yellows']} yellows, {metrics['reds']} reds - tactical control needed"
        })
    
    # Output
    if metrics["goal_contributions_p90"] >= 0.8:
        strengths.append({
            "title": "High Output",
            "detail": f"{metrics['goal_contributions_p90']}/90 goal contributions - decisive impact"
        })
    elif metrics["goal_contributions_p90"] < 0.2 and player.position in ["FW", "MF"]:
        development.append({
            "title": "Attacking Output",
            "detail": f"{metrics['goal_contributions_p90']}/90 contributions - increase involvement"
        })
    
    # Availability
    if overview["start_percentage"] < 50 and mins >= 500:
        development.append({
            "title": "Starting Consistency",
            "detail": f"{overview['start_percentage']}% start rate - compete for regular spot"
        })

    # =====================================================
    # RECENT FORM (Last 5 matches)
    # =====================================================
    recent_matches = list(
        PlayerMatchStat.objects
        .filter(player=player, season=season)
        .select_related("match", "match__home_team", "match__away_team")
        .order_by("-match__date_parsed")[:5]
    )

    recent = {
        "matches": len(recent_matches),
        "minutes": sum(m.minutes or 0 for m in recent_matches),
        "goals": sum(m.goals or 0 for m in recent_matches),
        "assists": sum(m.assists or 0 for m in recent_matches),
        "xg": round(sum(m.xg or 0 for m in recent_matches), 2),
        "xag": round(sum(m.xag or 0 for m in recent_matches), 2),
    }

    # Form classification
    contributions = recent["goals"] + recent["assists"]
    if contributions >= 3:
        recent["form"] = "Excellent"
        recent["form_desc"] = "High recent output"
    elif contributions >= 2:
        recent["form"] = "Good"
        recent["form_desc"] = "Consistent contributor"
    elif contributions >= 1:
        recent["form"] = "Average"
        recent["form_desc"] = "Moderate impact"
    else:
        recent["form"] = "Quiet"
        recent["form_desc"] = "Limited recent output"

    # =====================================================
    # MATCH LOGS
    # =====================================================
    match_logs = []
    for m in recent_matches:
        match = m.match
        is_home = match.home_team.name == m.team.name
        opponent = match.away_team.name if is_home else match.home_team.name

        match_logs.append({
            "date": match.date_parsed,
            "opponent": opponent,
            "home": is_home,
            "minutes": m.minutes or 0,
            "goals": m.goals or 0,
            "assists": m.assists or 0,
            "xg": round(m.xg or 0, 2) if m.xg else 0,
            "xag": round(m.xag or 0, 2) if m.xag else 0,
            "sca": m.sca or 0,
            "gca": m.gca or 0,
            "touches": m.touches or 0,
            #"pass_completion": round(m.pass_completion_pct or 0, 1) if m.pass_completion_pct else 0,
        })

    # =====================================================
    # PERCENTILE ANALYSIS
    # =====================================================
    league_pool = PlayerSeasonStat.objects.filter(
        league=player.league,
        season=player.season,
        playing_time_90s__gte=5  # Min 5 full matches for fair comparison
    )

    percentile_data = {
        "attack": [],
        "progression": [],
        "creativity": [],
        "passing": [],
        "defense": [],
        "discipline": []
    }

    for label, field, category in PERCENTILE_STATS:
        pool_values = [getattr(p, field, None) for p in league_pool]
        player_value = getattr(player, field, None)
        
        pct = percentile_rank(
            player_value,
            pool_values,
            flip=(field in LESS_IS_BETTER)
        )

        if pct is not None:
            percentile_data[category].append({
                "label": label,
                "percentile": pct,
                "value": player_value
            })

    context = {
        "player": player,
        "overview": overview,
        "metrics": metrics,
        "strengths": strengths,
        "development": development,
        "recent": recent,
        "match_logs": match_logs,
        "percentile_data": json.dumps(percentile_data),
    }
    
    return render(request, "player_detail.html", context)


def player_comparison(request):
    competitions = Competition.objects.all().order_by("name")

    return render(request, "player_comparison.html", {
        "competitions": competitions
    })

def player_comparison_radar(request):

    # ---------------------- INPUT ----------------------
    p1_id = request.GET.get("p1")
    p2_id = request.GET.get("p2")

    p1 = PlayerSeasonStat.objects.get(player_id=p1_id, season="25/26")
    p2 = PlayerSeasonStat.objects.get(player_id=p2_id, season="25/26")

    # ---------------------- PARAMETERS (ONLY AVAILABLE STATS) ----------------------
    params = [
        "npxG",
        "Non-Penalty Goals",
        "xA",
        "Goals + Assists",
        "Progressive Passes",
        "Progressive Carries",
        "Progressive Runs",
        "npxG + xA",
        "Minutes (90s)"
    ]

    low  = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    high = [0.6, 0.6, 0.6, 1.2, 10, 8, 8, 1.5, 40]

    radar = Radar(
        params,
        low,
        high,
        round_int=[False] * len(params),
        num_rings=4,
        ring_width=1,
        center_circle_radius=1
    )

    # ---------------------- FONTS (UNCHANGED) ----------------------
    URL_REG = "https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab%5Bwght%5D.ttf"
    URL_THIN = "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf"

    font_reg = FontManager(URL_REG)
    font_num = FontManager(URL_THIN)

    def safe(v):
        return float(v) if v is not None else 0.0

    # ---------------------- PLAYER VALUES ----------------------
    p1_vals = [
        safe(p1.p90_npxg),
        safe(p1.p90_goals_non_penalty),
        safe(p1.p90_xag),
        safe(p1.p90_goals_assists),
        safe(p1.progressive_passes) / (safe(p1.playing_time_90s) or 1),
        safe(p1.progressive_carries) / (safe(p1.playing_time_90s) or 1),
        safe(p1.progressive_runs) / (safe(p1.playing_time_90s) or 1),
        safe(p1.p90_npxg_xag),
        safe(p1.playing_time_90s),
    ]

    p2_vals = [
        safe(p2.p90_npxg),
        safe(p2.p90_goals_non_penalty),
        safe(p2.p90_xag),
        safe(p2.p90_goals_assists),
        safe(p2.progressive_passes) / (safe(p2.playing_time_90s) or 1),
        safe(p2.progressive_carries) / (safe(p2.playing_time_90s) or 1),
        safe(p2.progressive_runs) / (safe(p2.playing_time_90s) or 1),
        safe(p2.p90_npxg_xag),
        safe(p2.playing_time_90s),
    ]

    # ---------------------- FIGURE (DESIGN UNCHANGED) ----------------------
    fig, axs = grid(
        figheight=13,
        grid_height=0.85,
        title_height=0.09,
        endnote_height=0.04,
        title_space=0,
        endnote_space=0,
        grid_key="radar",
        axis=False
    )

    fig.patch.set_facecolor("#ffffff")
    axs["radar"].set_facecolor("#ffffff")

    radar.setup_axis(ax=axs["radar"])

    ring_colors = ["#e5e7eb", "#cbd5e1"]
    for i in range(radar.num_rings):
        radar.draw_circles(
            ax=axs["radar"],
            facecolor=ring_colors[i % 2],
            edgecolor="#d1d5db",
            lw=1.1,
            zorder=0
        )

    radar_output = radar.draw_radar_compare(
        p1_vals,
        p2_vals,
        ax=axs["radar"],
        kwargs_radar=dict(
            facecolor="#2563eb",
            edgecolor="#1e40af",
            lw=2,
            alpha=0.45
        ),
        kwargs_compare=dict(
            facecolor="#dc2626",
            edgecolor="#991b1b",
            lw=2,
            alpha=0.45
        )
    )

    poly1, poly2, verts1, verts2 = radar_output

    axs["radar"].scatter(
        verts1[:, 0], verts1[:, 1],
        s=90, color="#2563eb",
        edgecolors="white", linewidth=0.8, zorder=3
    )
    axs["radar"].scatter(
        verts2[:, 0], verts2[:, 1],
        s=90, color="#dc2626",
        edgecolors="white", linewidth=0.8, zorder=3
    )

    # ---------------------- LABELS (VISIBLE) ----------------------
    radar.draw_param_labels(
        ax=axs["radar"],
        fontsize=13,
        fontproperties=font_reg.prop,
        color="#111827"
    )

    radar.draw_range_labels(
        ax=axs["radar"],
        fontsize=12,
        fontproperties=font_num.prop,
        color="#374151",
        zorder=5
    )

    # ---------------------- TITLES ----------------------
    axs["title"].text(
        0.01, 0.65,
        f"{p1.player_name} ({p1.team.name})",
        fontsize=18,
        fontproperties=font_reg.prop,
        color="#1e40af",
        ha="left"
    )

    axs["title"].text(
        0.99, 0.65,
        f"{p2.player_name} ({p2.team.name})",
        fontsize=18,
        fontproperties=font_reg.prop,
        color="#991b1b",
        ha="right"
    )

    axs["endnote"].text(
        0.99, 0.5,
        "Season 25/26 · PlayWise",
        fontsize=10,
        fontproperties=font_num.prop,
        color="#4b5563",
        ha="right"
    )

    # ---------------------- RESPONSE ----------------------
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return HttpResponse(buf.getvalue(), content_type="image/png")



def team_comparison(request):
    season = "25/26"

    # All competitions that actually have matches this season
    competitions = Competition.objects.filter(
        teams__home_matches__season=season
    ).distinct()

    # Default: no teams until competition selected
    teams = Team.objects.none()

    competition_id = request.GET.get("competition")

    if competition_id:
        teams = Team.objects.filter(
            competition_id=competition_id
        ).filter(
            Q(home_matches__season=season) |
            Q(away_matches__season=season)
        ).distinct().order_by("name")

    context = {
        "season": season,
        "competitions": competitions,
        "teams": teams,
        "selected_competition": competition_id,
    }

    return render(request, "team_comparison.html", context)


def team_comparison_bumpy(request):
    """
    Render a Bumpy chart (PNG) showing week-by-week league positions for the season,
    using only played matches (home_goals and away_goals both not null).

    Query params:
      teams=<id>&teams=<id>&teams=<id>  (up to 3 team ids)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from urllib.request import urlopen
    from PIL import Image
    from highlight_text import fig_text
    from mplsoccer import Bumpy, FontManager, add_image

    from .models import Match, Team

    # ----- Input -----
    team_ids = request.GET.getlist("teams")[:3]  # up to 3 teams
    season = request.GET.get("season", "25/26")

    # ----- Fonts (use stable gstatic URLs to avoid 404 issues) -----
    font_normal = FontManager("https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf")
    font_bold = FontManager("https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf")

    # ----- EPL image (kept from example design) -----
    try:
        epl = Image.open(urlopen("https://raw.githubusercontent.com/andrewRowlinson/mplsoccer-assets/main/epl.png"))
    except Exception:
        epl = None

    # ----- Matches: only played ones for the season -----
    matches_qs = Match.objects.filter(
        season=season,
        home_goals__isnull=False,
        away_goals__isnull=False
    ).order_by("week", "date_parsed")

    # If no matches, return a small message-image
    if not matches_qs.exists():
        fig = plt.figure(figsize=(8, 3))
        fig.text(0.5, 0.5, f"No played matches for season {season}", ha="center", va="center", fontsize=14)
        buf_empty = io.BytesIO()
        plt.savefig(buf_empty, format="png", bbox_inches="tight")
        plt.close(fig)
        return HttpResponse(buf_empty.getvalue(), content_type="image/png")

    # ----- Compute weeks present (sorted) -----
    weeks = sorted({m.week for m in matches_qs if m.week is not None})
    if not weeks:
        # fallback: assign weeks by unique dates if week field absent
        dates = sorted({m.date for m in matches_qs})
        # create pseudo-weeks by date order
        date_to_week = {d: i + 1 for i, d in enumerate(dates)}
        for m in matches_qs:
            m.week = date_to_week.get(m.date, None)
        weeks = sorted({m.week for m in matches_qs if m.week is not None})

    week_labels = [f"Week {w}" for w in weeks]

    # ----- All teams participating this season (from matches) -----
    teams_in_season = set()
    for m in matches_qs:
        teams_in_season.add(m.home_team.name)
        teams_in_season.add(m.away_team.name)
    teams_in_season = sorted(list(teams_in_season))  # stable order

    # ----- Helper: build league table from a queryset of matches -----
    def build_table(ms):
        table = {team: {"pts": 0, "gd": 0, "gf": 0} for team in teams_in_season}
        for m in ms:
            h = m.home_team.name
            a = m.away_team.name
            if m.home_goals is None or m.away_goals is None:
                continue
            # goals for / gd
            table[h]["gf"] += m.home_goals
            table[h]["gd"] += (m.home_goals - m.away_goals)
            table[a]["gf"] += m.away_goals
            table[a]["gd"] += (m.away_goals - m.home_goals)
            # points
            if m.home_goals > m.away_goals:
                table[h]["pts"] += 3
            elif m.home_goals < m.away_goals:
                table[a]["pts"] += 3
            else:
                table[h]["pts"] += 1
                table[a]["pts"] += 1
        return table

    # ----- Build week-by-week positions for all teams -----
    season_dict = {team: [] for team in teams_in_season}

    for w in weeks:
        # matches up to and including week w
        ms = matches_qs.filter(week__lte=w)
        table = build_table(ms)
        # sort table: points desc, gd desc, gf desc, then name
        sorted_table = sorted(
            table.items(),
            key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"], -kv[1]["gf"], kv[0])
        )
        # make mapping team -> position
        pos_map = {team: pos for pos, (team, _) in enumerate(sorted_table, start=1)}
        # append positions for all teams (ensures same length)
        for team in teams_in_season:
            season_dict[team].append(pos_map.get(team, len(teams_in_season)))

    # ----- Prepare highlight dict: color selected teams (keep example colors) -----
    selected_teams = []
    if team_ids:
        selected_qs = Team.objects.filter(id__in=team_ids)
        selected_teams = [t.name for t in selected_qs if t.name in teams_in_season]

    base_colors = ["crimson", "skyblue", "gold"]
    highlight_dict = {selected_teams[i]: base_colors[i] for i in range(len(selected_teams))}

    # ----- Bumpy design EXACTLY as example (keeps fontproperties usage) -----
    bumpy = Bumpy(
        scatter_color="#282A2C", line_color="#252525",
        rotate_xticks=90,
        ticklabel_size=17, label_size=30,
        scatter_primary='D',
        show_right=True,
        plot_labels=True,
        alignment_yvalue=0.1, alignment_xvalue=0.065
    )

    # ensure y_list covers possible positions (1..Nteams)
    num_teams = max(20, len(teams_in_season))  # keep at least 20 for visual similarity
    y_list = np.linspace(1, num_teams, num_teams).astype(int)

    fig, ax = bumpy.plot(
        x_list=week_labels,
        y_list=y_list,
        values=season_dict,
        secondary_alpha=0.5,
        highlight_dict=highlight_dict,
        figsize=(20, 16),
        x_label='Week',
        y_label='Position',
        ylim=(-0.1, num_teams + 3),
        lw=2.5,
        fontproperties=font_normal.prop,
    )

    # Title + subtitle (keep example structure, highlight selected teams in subtitle)
    TITLE = f"2025/26 week-wise standings:"
    if selected_teams:
        # build subtitle with highlighted tags as in example: <Team>
        tags = ", ".join([f"<{t}>" for t in selected_teams])
        SUB_TITLE = f"A comparison between {tags}s"
    else:
        SUB_TITLE = "League position comparison"

    fig.text(0.09, 0.95, TITLE, size=29, color="#F2F2F2", fontproperties=font_bold.prop)
    # highlight_text expects a list of highlight props; map selected colors
    highlight_colors = [highlight_dict.get(t, "white") for t in selected_teams]
    # fallback: if no selected teams, show subtitle without highlights
    if selected_teams:
        fig_text(
            0.09, 0.94, SUB_TITLE, color="#F2F2F2",
            highlight_textprops=[{"color": c} for c in highlight_colors],
            size=25, fig=fig, fontproperties=font_bold.prop
        )
    else:
        fig_text(0.09, 0.94, SUB_TITLE, color="#F2F2F2",
                 size=25, fig=fig, fontproperties=font_bold.prop)

    # add EPL image if available (keeps design)
    if epl is not None:
        try:
            fig = add_image(epl, fig, 0.02, 0.9, 0.08, 0.08)
        except Exception:
            pass

    plt.tight_layout(pad=0.5)

    # ----- Return PNG response -----
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close("all")
    buf.seek(0)
    return HttpResponse(buf.getvalue(), content_type="image/png")

def stat_leaders(request):
    competitions = Competition.objects.order_by("name")
    return render(request, "stat_leaders.html", {
        "competitions": competitions,
        "stat_options": STAT_REGISTRY
    })

def stat_leaders_api(request):
    competition_id = request.GET.get("competition_id")
    stat = request.GET.get("stat")
    season = "25/26"

    if stat not in STAT_REGISTRY:
        return JsonResponse({"error": "Invalid stat"}, status=400)

    meta = STAT_REGISTRY[stat]

    qs = PlayerSeasonStat.objects.filter(
        season=season,
        league_id=competition_id,
        **{f"{stat}__isnull": False},
        playing_time_min__gte=300
    ).select_related("team").order_by(meta["order"])[:10]

    rows = []
    for p in qs:
        rows.append({
            "Player": p.player_name,
            "Squad": p.team.name,
            "Age": p.age,
            "Mins": p.playing_time_min,
            "value": round(getattr(p, stat), 2)
        })

    return JsonResponse({
        "label": meta["label"],
        "less_is_better": meta["less_is_better"],
        "rows": rows
    })


def player_pizza_page(request):
    competitions = Competition.objects.all().order_by("name")

    context = {
        "competitions": competitions,
        "season": "25/26",
    }
    return render(request, "pizza_player.html", context)

def player_pizza_api(request):
    player_id = request.GET.get("player_id")
    season = "25/26"

    if not player_id:
        return JsonResponse({"error": "Missing player_id"}, status=400)

    player = PlayerSeasonStat.objects.select_related(
        "team", "league"
    ).get(player_id=player_id, season=season)

    league_players = PlayerSeasonStat.objects.filter(
        league=player.league,
        season=season,
        playing_time_min__gte=600
    )

    # --- Stats used in pizza (ORDER MATTERS) ---
    stats = [
        ("Non-Penalty Goals", "goals_non_penalty"),
        ("npxG", "npxg"),
        ("xA", "xag"),
        ("xG + xA", "npxg_xag"),
        ("Goals + Assists", "goals_assists"),

        ("Progressive Passes", "progressive_passes"),
        ("Progressive Carries", "progressive_carries"),
        ("Progressive Runs", "progressive_runs"),
        ("xA per 90", "p90_xag"),
        ("xG per 90", "p90_npxg"),

        ("Goals per 90", "p90_goals"),
        ("Assists per 90", "p90_assists"),
        ("G+A per 90", "p90_goals_assists"),
        ("NP G+A per 90", "p90_goals_assists_non_penalty"),
        ("xG+xA per 90", "p90_xg_xag"),
    ]

    percentiles = []

    for _, field in stats:
        values = list(
            league_players.exclude(**{f"{field}__isnull": True})
            .values_list(field, flat=True)
        )

        if not values:
            percentiles.append(0)
            continue

        player_value = getattr(player, field)
        pctl = percentileofscore(values, player_value, kind="rank")
        percentiles.append(round(pctl))

    # ---------------- PIZZA PLOT ----------------

    font_normal = FontManager(
        "https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf"
    )
    font_bold = FontManager(
        "https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf"
    )

    slice_colors = (
        ["#1A78CF"] * 5 +
        ["#FF9300"] * 5 +
        ["#D70232"] * 5
    )

    baker = PyPizza(
        params=[s[0] for s in stats],
        background_color="#EBEBE9",
        straight_line_color="#EBEBE9",
        straight_line_lw=1,
        last_circle_lw=0,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        percentiles,
        figsize=(8, 8.5),
        slice_colors=slice_colors,
        value_colors=["#000000"] * 10 + ["#F2F2F2"] * 5,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(edgecolor="#F2F2F2", linewidth=1),
        kwargs_params=dict(
            color="#000000",
            fontsize=11,
            fontproperties=font_normal.prop,
            va="center",
        ),
        kwargs_values=dict(
            color="#000000",
            fontsize=11,
            fontproperties=font_normal.prop,
            bbox=dict(
                edgecolor="#000000",
                facecolor="cornflowerblue",
                boxstyle="round,pad=0.2",
                lw=1,
            ),
        ),
    )

    # Title
    fig.text(
        0.5, 0.97,
        f"{player.player_name} — {player.team.name}",
        ha="center",
        fontproperties=font_bold.prop,
        fontsize=16
    )

    fig.text(
        0.5, 0.945,
        f"Percentile Rank vs {player.league.name} Players | Season 25/26",
        ha="center",
        fontproperties=font_normal.prop,
        fontsize=12
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close("all")
    buf.seek(0)

    return HttpResponse(buf.getvalue(), content_type="image/png")


def api_teams_by_competition(request):
    competition_id = request.GET.get("competition")

    if not competition_id:
        return JsonResponse([], safe=False)

    teams = Team.objects.filter(
        competition_id=competition_id
    ).order_by("name")

    data = [
        {"id": t.id, "name": t.name}
        for t in teams
    ]

    return JsonResponse(data, safe=False)


def api_players_by_team(request):
    team_id = request.GET.get("team")
    season = "25/26"

    if not team_id:
        return JsonResponse([], safe=False)

    players = PlayerSeasonStat.objects.filter(
        team_id=team_id,
        season=season,
        playing_time_min__gte=300   # important filter
    ).order_by("player_name")

    data = [
        {
            "player_id": p.player_id,
            "name": p.player_name
        }
        for p in players
    ]

    return JsonResponse(data, safe=False)

@login_required
def coach_role_leaders_page(request):
    """
    Render page for coach; coach is required to have a Team via Coach.profile.
    Page supplies role options only (we compute top-5 via AJAX).
    """
    try:
        coach = Coach.objects.select_related("team").get(user=request.user)
    except Coach.DoesNotExist:
        # redirect or show error — keep simple: redirect to dashboard
        return redirect("dashboard")

    roles = sorted(POSITION_WEIGHTS.keys())
    context = {
        "roles": roles,
        "team": coach.team,
        "season": "25/26",
    }
    return render(request, "coach_role_leaders.html", context)


@login_required
def coach_role_leaders_api(request):
    """
    API: compute role scores for players in the coach's team and return top 5.
    Query params: ?role=ROLE_NAME
    """

    role = request.GET.get("role")
    season = "25/26"

    if not role or role not in POSITION_WEIGHTS:
        return JsonResponse({"error": "Invalid or missing role"}, status=400)

    # ---------------- Coach & team ----------------
    try:
        coach = Coach.objects.select_related("team").get(user=request.user)
    except Coach.DoesNotExist:
        return JsonResponse({"error": "Coach profile not found"}, status=403)

    team = coach.team
    league = team.competition

    # ---------------- Thresholds ----------------
    MIN_POOL_MINS = 600     # players used to build percentile pool
    MIN_TEAM_MINS = 200     # players eligible to appear in results

    # ---------------- Querysets ----------------
    pool_qs = PlayerSeasonStat.objects.filter(
        league=league,
        season=season,
        playing_time_min__gte=MIN_POOL_MINS
    )

    team_qs = PlayerSeasonStat.objects.filter(
        team=team,
        season=season,
        playing_time_min__gte=MIN_TEAM_MINS
    ).select_related("team").order_by("-playing_time_min")

    # ---------------- Role stats & weights ----------------
    role_weights = POSITION_WEIGHTS[role]
    stats = list(role_weights.keys())

    if not stats:
        return JsonResponse({"error": "No stats available for this role"}, status=400)

    # ---------------- Build pool arrays ----------------
    pool_arrays = {}
    for stat in stats:
        vals = list(
            pool_qs
            .exclude(**{f"{stat}__isnull": True})
            .values_list(stat, flat=True)
        )
        pool_arrays[stat] = np.asarray(vals, dtype=float) if vals else np.array([], dtype=float)

    # ---------------- Build team stat matrix ----------------
    players = list(team_qs)
    team_values = {stat: [] for stat in stats}

    for p in players:
        for stat in stats:
            v = getattr(p, stat, None)
            team_values[stat].append(np.nan if v is None else float(v))

    # ---------------- Percentile helper ----------------
    def percentiles_vs_pool(pool, values, flip=False):
        """
        Vectorized percentile computation.
        Returns array aligned with values.
        """
        values = np.asarray(values, dtype=float)

        if pool.size == 0:
            return np.full_like(values, np.nan)

        if flip:
            pool = -pool
            values = -values

        pool_sorted = np.sort(pool)
        out = np.full_like(values, np.nan)

        mask = ~np.isnan(values)
        ranks = np.searchsorted(pool_sorted, values[mask], side="left")
        out[mask] = (ranks / pool_sorted.size) * 100.0
        return out

    # ---------------- Compute percentiles ----------------
    percentiles = {}
    for stat in stats:
        percentiles[stat] = percentiles_vs_pool(
            pool_arrays[stat],
            team_values[stat],
            flip=(stat in LESS_IS_BETTER_KEYS)
        )

    # ---------------- Compute role scores ----------------
    results = []

    for idx, player in enumerate(players):
        num = 0.0
        den = 0.0

        for stat in stats:
            pct = percentiles[stat][idx]
            if np.isnan(pct):
                continue

            w = float(role_weights[stat])
            num += pct * w
            den += abs(w)

        score = (num / den) if den > 0 else 0.0

        # Role-specific adjustment (small, principled)
        adj_fn = POSITION_ADJUSTMENTS.get(role)
        if adj_fn:
            score = adj_fn(score)

        score = max(0.0, min(100.0, score))

        results.append({
            "player_id": player.player_id,
            "player_name": player.player_name,
            "team": player.team.name,
            "age": player.age,
            "mins": int(player.playing_time_min or 0),
            "score": round(score, 2)
        })

    # ---------------- Sort & respond ----------------
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    return JsonResponse({
        "role": role,
        "team": team.name,
        "season": season,
        "rows": results
    })


@login_required
def coach_role_suitability_page(request):
    coach = Coach.objects.select_related("team").get(user=request.user)
    team = coach.team
    season = "25/26"

    players = PlayerSeasonStat.objects.filter(
        team=team,
        season=season
    ).order_by("player_name")

    roles = sorted(POSITION_WEIGHTS.keys())

    return render(request, "role_suitability.html", {
        "players": players,
        "roles": roles
    })

POSITION_GROUPS = {
    "GK": {"GK"},

    "DF": {"CB", "FB", "WB"},
    "MF": {"DM", "CM", "AM"},
    "FW": {"ST", "W"},
}

def get_player_role_groups(pos_str: str) -> set[str]:
    """
    Converts FBref-style pos string (DF, MF, FW, MF,FW)
    into allowed role groups.
    """
    if not pos_str:
        return set()

    tokens = {p.strip() for p in pos_str.upper().split(",")}

    groups = set()
    if "DF" in tokens:
        groups |= POSITION_GROUPS["DF"]
    if "MF" in tokens:
        groups |= POSITION_GROUPS["MF"]
    if "FW" in tokens:
        groups |= POSITION_GROUPS["FW"]

    return groups

ROLE_GROUPS = {
    # --- Centre backs ---
    "CB - Stopper": {"CB"},
    "CB - Ball-Playing": {"CB"},
    "CB - Sweeper": {"CB"},

    # --- Fullbacks ---
    "Fullback - Defensive": {"FB", "WB"},
    "Fullback - Overlapping": {"FB", "WB"},
    "Fullback - Inverted": {"FB", "WB"},

    # --- Defensive midfield ---
    "DM - Ball Winner": {"DM"},
    "DM - Deep-Lying Playmaker": {"DM"},

    # --- Central midfield ---
    "CM - Water Carrier": {"CM"},
    "CM - Progresser": {"CM"},
    "CM - Box-to-Box": {"CM"},
    "CM - Mezzala": {"CM"},

    # --- Attacking midfield ---
    "AM - Classic 10": {"AM"},
    "AM - Primary Creator": {"AM"},
    "AM - Secondary Scorer": {"AM"},
    "AM - Shadow Striker": {"AM"},

    # --- Wingers ---
    "Winger - Classic": {"W"},
    "Winger - Inverted": {"W"},
    "Winger - Ball Progressor": {"W"},

    # --- Strikers ---
    "ST - Pure Finisher": {"ST"},
    "ST - Target Man": {"ST"},
    "ST - Creative Forward": {"ST"},
    "ST - Complete Forward": {"ST"},

    # --- Global ---
    "Overall": {"CB","FB","WB","DM","CM","AM","W","ST"},
    "Risk-Aware Player": {"CB","FB","WB","DM","CM","AM","W","ST"},
    "Durability Asset": {"CB","FB","WB","DM","CM","AM","W","ST"},
}

@login_required
def coach_role_suitability_api(request):
    player_id = request.GET.get("player_id")
    if not player_id:
        return JsonResponse({"error": "player_id required"}, status=400)

    season = "25/26"

    try:
        player = PlayerSeasonStat.objects.select_related(
            "team", "league"
        ).get(player_id=player_id, season=season)
    except PlayerSeasonStat.DoesNotExist:
        return JsonResponse({"error": "Player not found"}, status=404)

    league = player.league

    MIN_POOL_MINS = 600

    # -----------------------------
    # Percentile reference pool
    # -----------------------------
    pool_qs = PlayerSeasonStat.objects.filter(
        league=league,
        season=season,
        playing_time_min__gte=MIN_POOL_MINS
    )

    # -----------------------------
    # Player position groups (CRITICAL)
    # -----------------------------
    player_groups = get_player_role_groups(player.position)

    role_scores = []

    for role, weights in POSITION_WEIGHTS.items():

        # -----------------------------
        # HARD ROLE FILTER (Step 2)
        # -----------------------------
        allowed_groups = ROLE_GROUPS.get(role)
        if allowed_groups and not (player_groups & allowed_groups):
            continue  # role not football-valid for this player

        numer = 0.0
        denom = 0.0

        for field, w in weights.items():
            val = getattr(player, field, None)
            if val is None:
                continue

            pool_vals = list(
                pool_qs.filter(**{f"{field}__isnull": False})
                       .values_list(field, flat=True)
            )

            if not pool_vals:
                continue

            pool_vals = np.asarray(pool_vals, dtype=float)

            # Percentile logic
            if field in LESS_IS_BETTER_KEYS:
                pct = 100.0 * (pool_vals > val).mean()
            else:
                pct = 100.0 * (pool_vals < val).mean()

            numer += pct * abs(w)
            denom += abs(w)

        if denom == 0:
            continue

        score = numer / denom

        # -----------------------------
        # Role adjustment (small nudges only)
        # -----------------------------
        adj = POSITION_ADJUSTMENTS.get(role)
        if adj:
            try:
                score = adj(score)
            except Exception:
                pass

        score = round(max(0.0, min(100.0, score)), 1)

        role_scores.append({
            "role": role,
            "score": score
        })

    role_scores.sort(key=lambda x: x["score"], reverse=True)

    return JsonResponse({
        "player": player.player_name,
        "position": player.position,
        "roles": role_scores
    })

# A small set of formations to support and the slots required.
# Each slot string is descriptive; scoring function will interpret it.
FORMATIONS = {
    "4-3-3": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "ST", "RW"],
    "4-2-3-1": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "LAM", "CAM", "RAM", "ST"],
    "4-4-2": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"],
    "3-5-2": ["GK", "CB", "CB", "CB", "LWB", "CM", "CM", "CM", "RWB", "ST", "ST"],
    "4-1-4-1": ["GK", "LB", "CB", "CB", "RB", "CDM", "LM", "CM", "CM", "RM", "ST"],
}

# Per-slot stat weights (fields must exist on PlayerSeasonStat)
# Small, interpretable mixes. You can tune these weights later.
SLOT_WEIGHTS = {
    # defenders
    "CB": {"clearances": 1.2, "interceptions": 1.0, "tackles": 0.9, "blocks": 0.6, "playing_time_min": 0.2},
    "LB": {"progressive_passes": 0.6, "crosses": 0.8, "tackles": 0.9, "passes_completed": 0.4, "playing_time_min": 0.2},
    "RB": {"progressive_passes": 0.6, "crosses": 0.8, "tackles": 0.9, "passes_completed": 0.4, "playing_time_min": 0.2},
    "LWB": {"progressive_passes": 0.7, "crosses": 0.9, "tackles": 0.8, "playing_time_min": 0.2},
    "RWB": {"progressive_passes": 0.7, "crosses": 0.9, "tackles": 0.8, "playing_time_min": 0.2},

    # full generic defense fallback
    "DEF": {"clearances": 1.0, "interceptions": 0.9, "tackles": 0.9, "playing_time_min": 0.2},

    # midfield
    "CM": {"progressive_passes": 1.0, "progressive_carries": 0.6, "key_passes": 0.7, "p90_goals": 0.6, "p90_assists": 0.6, "playing_time_min": 0.2},
    "CDM": {"interceptions": 1.0, "tackles": 1.0, "clearances": 0.6, "progressive_passes": 0.5, "playing_time_min": 0.2},
    "CM_DEF": {"interceptions": 1.0, "tackles": 1.0, "progressive_passes": 0.5, "playing_time_min": 0.2},
    "LM": {"progressive_carries": 0.9, "crosses": 0.8, "key_passes": 0.7, "p90_assists": 0.6, "playing_time_min": 0.2},
    "RM": {"progressive_carries": 0.9, "crosses": 0.8, "key_passes": 0.7, "p90_assists": 0.6, "playing_time_min": 0.2},
    "LAM": {"key_passes": 1.0, "expected_assists": 0.9, "p90_assists": 0.8, "p90_goals": 0.5, "playing_time_min": 0.2},
    "RAM": {"key_passes": 1.0, "expected_assists": 0.9, "p90_assists": 0.8, "p90_goals": 0.5, "playing_time_min": 0.2},
    "CAM": {"key_passes": 1.1, "expected_assists": 0.9, "p90_goals": 0.6, "p90_assists": 0.6, "playing_time_min": 0.2},

    # wide attackers
    "LW": {"progressive_carries": 0.8, "crosses": 0.7, "p90_goals": 0.8, "p90_assists": 0.6, "playing_time_min": 0.2},
    "RW": {"progressive_carries": 0.8, "crosses": 0.7, "p90_goals": 0.8, "p90_assists": 0.6, "playing_time_min": 0.2},

    # forwards
    "ST": {"p90_goals": 1.5, "p90_xg": 1.2, "p90_goals_assists": 0.6, "progressive_carries": 0.4, "playing_time_min": 0.2},
    "CF": {"p90_goals": 1.3, "p90_xg": 1.1, "p90_goals_assists": 0.8, "progressive_carries": 0.6, "playing_time_min": 0.2},

    # GK (very simple because GK stats are not stored here — choose by minutes for now)
    "GK": {"playing_time_min": 1.0},
}

def _normalize_array(arr):
    """Normalize a 1D numpy array to 0..1 (max-based). Returns zeros when all nan/zero."""
    a = np.array(arr, dtype=float)
    # replace nan with 0
    a = np.nan_to_num(a, nan=0.0)
    mx = a.max() if a.size > 0 else 0.0
    if mx <= 0:
        return np.zeros_like(a, dtype=float)
    return a / float(mx)

def _slot_stat_score(player, slot, team_maxes):
    """
    Compute a raw score (0..1) for a player for the given slot using SLOT_WEIGHTS.
    team_maxes: dict mapping field -> max value across team (for safe division).
    """
    weights = SLOT_WEIGHTS.get(slot) or SLOT_WEIGHTS.get(slot.split("_")[0]) or {}
    if not weights:
        return 0.0, {}

    ssum = 0.0
    weight_sum = 0.0
    debug_breakdown = {}
    for field, w in weights.items():
        # defensive fallback naming: playing_time_min is numeric field
        val = getattr(player, field, None)
        if val is None:
            val = 0.0
        try:
            val_f = float(val)
        except Exception:
            val_f = 0.0

        maxv = team_maxes.get(field) or 0.0
        contrib = 0.0
        if maxv > 0:
            contrib = (val_f / maxv) * float(abs(w))
        else:
            # if no max available, use raw scaled by 0.0 (so it's negligible)
            contrib = 0.0

        ssum += contrib if w >= 0 else -contrib
        weight_sum += abs(w)
        debug_breakdown[field] = round(contrib, 4)

    if weight_sum <= 0:
        score = 0.0
    else:
        # scale to 0..100
        score = (ssum / (weight_sum + 1e-9)) * 100.0
        score = max(score, 0.0)
    return score, debug_breakdown

def _player_position_matches_slot(player, slot):
    """
    Return a soft multiplier (1.0 when player position broadly matches slot,
    lower when mismatch). player.position stored like "DF", "MF", "FW", "MF,FW".
    """
    pos_raw = (player.position or "").upper()
    # if nothing, be permissive but penalize
    if not pos_raw:
        return 0.6

    tokens = {p.strip() for p in pos_raw.split(",") if p.strip()}
    # slot grouping
    if slot in ("GK",):
        return 1.0 if "GK" in tokens else 0.0
    if slot in ("CB", "LB", "RB", "LWB", "RWB", "DEF"):
        # accept DF broad tokens
        return 1.0 if any(tok == "DF" for tok in tokens) else 0.6 if any(tok == "MF" for tok in tokens) else 0.3
    if slot.startswith("C") or slot in ("CM", "CDM", "CAM", "CM_DEF"):
        return 1.0 if any(tok == "MF" for tok in tokens) else 0.6 if any(tok == "DF" for tok in tokens) else 0.4
    if slot in ("LM", "RM", "LW", "RW", "LAM", "RAM"):
        return 1.0 if any(tok == "FW" for tok in tokens) or any(tok == "MF" for tok in tokens) else 0.5
    if slot in ("ST", "CF"):
        return 1.0 if any(tok == "FW" for tok in tokens) else 0.5
    return 0.6


@login_required
def xi_suggestion_page(request):
    """
    Page for coach to select an upcoming match (or past match) and try formations.
    Shows matches where coach.team is either home or away in chosen season.
    """
    try:
        coach = Coach.objects.select_related("team").get(user=request.user)
    except Coach.DoesNotExist:
        return render(request, "xi_suggestion.html", {"error": "Coach profile not found."})

    team = coach.team

    now = timezone.now()

    matches = Match.objects.filter(
        season=SEASON,
        date_parsed__gt=now   
    ).filter(
        models.Q(home_team=team) | models.Q(away_team=team)
    ).order_by("date_parsed")[:100]  

    competitions = Competition.objects.all()

    context = {
        "team": team,
        "matches": matches,
        "formations": list(FORMATIONS.keys()),
        "competitions": competitions,
        "season": SEASON,
    }
    return render(request, "xi_suggestion.html", context)


@login_required
def xi_suggestion_api(request):
    """
    API that returns a suggested XI JSON for a given match_id and formation.
    GET params: match_id (optional) and formation (required)
    If match_id is provided, it can be used to look at opponent info in future.
    """
    user = request.user
    formation = request.GET.get("formation")
    match_id = request.GET.get("match_id")  # optional

    if not formation or formation not in FORMATIONS:
        return JsonResponse({"error": "formation missing or invalid"}, status=400)

    # coach & team
    try:
        coach = Coach.objects.select_related("team").get(user=user)
    except Coach.DoesNotExist:
        return JsonResponse({"error": "Coach profile missing"}, status=403)
    team = coach.team

    # team players pool (only season players)
    players_qs = PlayerSeasonStat.objects.filter(team=team, season=SEASON)
    # optionally filter by minutes (we still include low-minute as fallback)
    players = list(players_qs)

    if not players:
        return JsonResponse({"error": "No players found for your team/season."}, status=404)

    # compute team maxes for fields used in weights
    fields_needed = set()
    for w in SLOT_WEIGHTS.values():
        fields_needed.update(w.keys())
    team_maxes = {}
    for fld in fields_needed:
        vals = [getattr(p, fld, 0) or 0 for p in players]
        try:
            team_maxes[fld] = float(max(vals)) if vals else 0.0
        except Exception:
            team_maxes[fld] = 0.0

    # function to score a player for a particular slot with position multiplier
    def score_player_for_slot(player, slot):
        raw_score, breakdown = _slot_stat_score(player, slot, team_maxes)
        pos_mul = _player_position_matches_slot(player, slot)
        # penalize players with extremely low minutes (soft)
        mins = float(player.playing_time_min or 0)
        mins_mul = 1.0 if mins >= 900 else (0.6 + 0.4 * (mins / 900.0))  # up to 900 mins is full
        score = raw_score * pos_mul * mins_mul
        return round(score, 2), breakdown, pos_mul, mins

    # greedy assignment function
    def assign_formation_slots(formation_slots):
        selected = []
        used_player_ids = set()
        candidate_table = {}  # slot -> list of (score, player, breakdown)
        # build candidate lists
        for slot in formation_slots:
            # try exact slot, fallback to family
            candidates = []
            for p in players:
                if (p.player_id in used_player_ids):
                    continue
                s, breakdown, pos_mul, mins = score_player_for_slot(p, slot)
                candidates.append((s, p, breakdown, pos_mul, mins))
            # sort desc
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidate_table[slot] = candidates

        # pick in order: GK first, then defenders, midfield, attack to reduce conflicts
        ordered_slots = sorted(
            list(formation_slots),
            key=lambda s: (0 if s == "GK" else (1 if s in ["CB","LB","RB","LWB","RWB","DEF"] else (2 if s.startswith("C") or s in ["CM","CDM","CAM","LM","RM"] else 3)))
        )

        for slot in ordered_slots:
            picks = [c for c in candidate_table.get(slot, []) if c[1].player_id not in used_player_ids]
            if picks:
                score, player, breakdown, pos_mul, mins = picks[0]
                selected.append({
                    "slot": slot,
                    "player_id": player.player_id,
                    "player_name": player.player_name,
                    "pos_raw": player.position,
                    "score": float(round(score, 2)),
                    "breakdown": breakdown,
                    "pos_match_multiplier": pos_mul,
                    "mins": int(mins)
                })
                used_player_ids.add(player.player_id)
            else:
                # no candidate available — pick best leftover player by minutes
                leftovers = [p for p in players if p.player_id not in used_player_ids]
                leftovers.sort(key=lambda x: (x.playing_time_min or 0), reverse=True)
                if leftovers:
                    p = leftovers[0]
                    selected.append({
                        "slot": slot,
                        "player_id": p.player_id,
                        "player_name": p.player_name,
                        "pos_raw": p.position,
                        "score": 0.0,
                        "breakdown": {},
                        "pos_match_multiplier": 0.0,
                        "mins": int(p.playing_time_min or 0)
                    })
                    used_player_ids.add(p.player_id)
                else:
                    selected.append({
                        "slot": slot,
                        "player_id": None,
                        "player_name": "N/A",
                        "pos_raw": "",
                        "score": 0.0,
                        "breakdown": {},
                        "pos_match_multiplier": 0.0,
                        "mins": 0
                    })
        # compute formation score
        formation_score = sum(item["score"] for item in selected)
        return selected, formation_score

    # produce suggested lineup for requested formation
    requested_slots = FORMATIONS[formation]
    lineup, fscore = assign_formation_slots(requested_slots)

    # also compute best formation among known ones
    best_choice = {"formation": formation, "lineup": lineup, "score": fscore}
    for fname, slots in FORMATIONS.items():
        sel, sc = assign_formation_slots(slots)
        if sc > best_choice["score"]:
            best_choice = {"formation": fname, "lineup": sel, "score": sc}

    # optional: include match/opponent info for context
    match = None
    if match_id:
        try:
            match = Match.objects.get(id=match_id)
        except Match.DoesNotExist:
            match = None

    resp = {
        "team": team.name,
        "requested_formation": formation,
        "requested_lineup": lineup,
        "requested_score": round(fscore, 2),
        "best_formation": best_choice["formation"],
        "best_lineup": best_choice["lineup"],
        "best_score": round(best_choice["score"], 2),
        "match": {
            "id": match.game_id if match else None,
            "opponent": (match.away_team.name if match and match.home_team == team else (match.home_team.name if match else None)) if match else None,
            "date": match.date.isoformat() if match else None
        } if match else None
    }
    
    return JsonResponse(resp, safe=True)


@login_required
def xi_pitch_image_api(request):
    """
    Render mplsoccer pitch image for requested or best XI.

    GET params:
      formation (required)
      match_id  (optional)
      mode      = requested | best (default: best)
    """
    mode = request.GET.get("mode", "best")

    # Call existing XI logic (JSON API)
    response = xi_suggestion_api(request)

    if response.status_code != 200:
        return response

    data = json.loads(response.content)

    # Decide which lineup to draw
    if mode == "requested":
        formation = data["requested_formation"]
        lineup = data["requested_lineup"]
        title_suffix = "Requested XI"
    else:
        formation = data["best_formation"]
        lineup = data["best_lineup"]
        title_suffix = "Optimized XI"

    # Draw pitch → returns BytesIO
    try:
        buf = draw_xi(
            team_name=data["team"],
            formation=formation,
            lineup=lineup,
            title_suffix=title_suffix
        )
    except ValueError as e:
        return HttpResponse(
            f"Formation not supported by pitch renderer: {formation}",
            status=400
        )

    return HttpResponse(buf.getvalue(), content_type="image/png")


