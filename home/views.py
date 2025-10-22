from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
import json
from .utils import predictor
from .models import *
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Sum, Count, Q
from decimal import Decimal
from datetime import date


# Create your views here.
def home(request):
    return render(request, 'index.html')



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

def predict_api(request):
    """API endpoint used by Axios from the front-end.
    Accepts JSON or form-encoded POSTs with keys: home_team, away_team
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    # parse JSON body or form body
    if request.content_type and request.content_type.startswith("application/json"):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"error": "invalid JSON"}, status=400)
        home = body.get("home_team")
        away = body.get("away_team")
    else:
        home = request.POST.get("home_team")
        away = request.POST.get("away_team")

    if not home or not away:
        return JsonResponse({"error": "home_team and away_team required"}, status=400)

    try:
        # call your predictor's production-ready function
        result = predictor.predict_match(home, away)
        return JsonResponse(result)
    except Exception as e:
        # return helpful error for debugging (you can log this on server)
        return JsonResponse({"error": str(e)}, status=500)


@login_required(login_url="manager_login")
def coach_dashboard(request):
    """
    Coach dashboard view — robust handling of missing stats.
    Aggregates only real values (no forced 0s), exposes totals and counts.
    """
    try:
        coach = Coach.objects.get(user=request.user)
    except Coach.DoesNotExist:
        return render(request, "coach_no_profile.html", {})

    team = coach.team
    today = date.today()

    # Upcoming (unplayed) fixtures
    upcoming_qs = Match.objects.filter(
        Q(home_team=team) | Q(away_team=team),
        played=False,
        date__gte=today
    ).order_by("date", "matchweek")[:5]
    upcoming = list(upcoming_qs)

    # Recent played matches (for display) - fetch a little more for robust charts
    recent_qs = Match.objects.filter(
        Q(home_team=team) | Q(away_team=team),
        played=True,
        date__lte=today
    ).order_by("-date")[:10]
    recent = list(recent_qs[:5])

    # pick most relevant season (most recent season that has any match for this team)
    season_qs = Match.objects.filter(Q(home_team=team) | Q(away_team=team)).order_by("-season")
    season_to_use = season_qs.first().season if season_qs.exists() else None

    # season matches that are played (we compute season metrics from played=True only)
    if season_to_use:
        season_matches_qs = Match.objects.filter(
            Q(home_team=team) | Q(away_team=team),
            season=season_to_use
        ).order_by("date")
    else:
        season_matches_qs = Match.objects.none()

    # counts and accumulators
    played_count = season_matches_qs.filter(played=True).count()

    # aggregated counters (only count where data present)
    wins = draws = losses = 0
    goals_for = goals_against = 0
    goals_for_count = goals_against_count = 0  # how many matches had goals recorded
    xg_for = xg_against = 0.0
    xg_for_count = xg_against_count = 0
    clean_sheets = 0
    # splits
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
        # skip matches that are marked not played when computing some stats? we will still use played flag where appropriate
        is_home = (m.home_team_id == team.id)

        # increment home/away match counters regardless of played flag (this is schedule-level)
        if is_home:
            matches_home += 1
        else:
            matches_away += 1

        # Goals: use y_home_goals / y_away_goals when present
        if m.y_home_goals is not None and m.y_away_goals is not None:
            # consider this match as "played" for stats
            gf = m.y_home_goals if is_home else m.y_away_goals
            ga = m.y_away_goals if is_home else m.y_home_goals

            # overall totals
            goals_for += gf
            goals_for_count += 1
            goals_against += ga
            goals_against_count += 1

            # splits
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

        # xG: count only when xG field exists for that side
        # home_xg and away_xg may be None
        home_xg_val = m.home_xg
        away_xg_val = m.away_xg

        if is_home:
            # team is home
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
            # team is away
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


     
        # Results: only if result is present and match marked played (we rely on result field)
        if m.result in ("H", "D", "A"):
            # classification relative to this team
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

    # computed metrics with safe guards (only compute averages when count > 0)
    goal_diff = (goals_for - goals_against) if (goals_for_count or goals_against_count) else None
    points = wins * 3 + draws
    ppg = round(points / played_count, 2) if played_count else None

    avg_xg_for = round(xg_for / xg_for_count, 2) if xg_for_count else None
    avg_xg_against = round(xg_against / xg_against_count, 2) if xg_against_count else None

    # conversion: goals / xG, only when we have xg_for_count and goals_for_count
    conversion_rate = None
    if xg_for > 0 and goals_for_count:
        conversion_rate = round(goals_for / xg_for, 2)
    elif xg_for_count and xg_for == 0 and goals_for_count:
        # xg is present but zero (rare) — show float 0.0 conversion explicitly
        conversion_rate = 0.0
    else:
        conversion_rate = None

    # last-5 form (use only matches with result present and played=True)
    last5_matches = list(season_matches_qs.filter(played=True, result__in=("H","D","A")).order_by("-date")[:5])
    form_points = []
    form_labels = []
    for m in reversed(last5_matches):  # oldest -> newest
        is_home = (m.home_team_id == team.id)
        if m.result == "D":
            pts = 1
            res_label = "D"
        else:
            pts = 3 if ((m.result == "H" and is_home) or (m.result == "A" and not is_home)) else 0
            res_label = "W" if pts == 3 else "L"
        form_points.append(pts)
        # label shows opponent and venue
        opp = m.away_team.name if is_home else m.home_team.name
        venue = "H" if is_home else "A"
        form_labels.append(f"{m.date.strftime('%d %b')} ({venue}) vs {opp}")

    # streak (consecutive same result from most recent played matches)
    streak = 0
    streak_type = None
    for m in season_matches_qs.filter(played=True, result__in=("H","D","A")).order_by("-date"):
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
    recent_played_for_charts = list(season_matches_qs.filter(played=True).order_by("-date")[:5])[::-1]
    recent_goals = []
    recent_xg = []
    recent_labels = []
    for m in recent_played_for_charts:
        is_home = (m.home_team_id == team.id)
        # goals: fall back to 0 if missing to keep chart lengths consistent, but record counts separately
        gf = (m.y_home_goals if is_home else m.y_away_goals)
        gf_val = int(gf) if gf is not None else None
        # xg: only the side's xG if present
        xg_val = (m.home_xg if is_home else m.away_xg)
        recent_goals.append(gf_val)
        recent_xg.append(round(float(xg_val), 2) if xg_val is not None else None)
        recent_labels.append(m.date.strftime("%d %b"))

    # next match prediction (if upcoming exists)
    next_pred = None
    if upcoming:
        nxt = upcoming[0]
        try:
            next_pred = predictor.predict_match(nxt.home_team.name, nxt.away_team.name)
        except Exception:
            next_pred = None

    # Prepare chart payloads, convert None -> null in JSON by leaving None (json.dumps handles None->null)
    chart_form = {"labels": form_labels, "data": form_points}
    # for charts we want numeric arrays; replace None with 0 for plotting but keep counts in context
    chart_goals = [g if g is not None else 0 for g in recent_goals]
    chart_xg = [x if x is not None else 0 for x in recent_xg]
    chart_goals_xg = {"labels": recent_labels, "goals": chart_goals, "xg": chart_xg}

    # Context: expose both totals and counts so template can show '—' when data missing
    context = {
        "coach": coach,
        "team": team,
        "upcoming": upcoming,
        "recent": recent,
        "season": season_to_use,
        # counts
        "played_count": played_count,
        # calculated result metrics (counts reflect only matches with result)
        "wins": wins, "draws": draws, "losses": losses,
        "points": points, "ppg": ppg,
        # goals totals & counts
        "goals_for": goals_for if goals_for_count else None,
        "goals_against": goals_against if goals_against_count else None,
        "goals_for_count": goals_for_count, "goals_against_count": goals_against_count,
        # xG totals & counts
        "xg_for_total": round(xg_for, 2) if xg_for_count else None,
        "xg_against_total": round(xg_against, 2) if xg_against_count else None,
        "xg_for_count": xg_for_count, "xg_against_count": xg_against_count,
        "avg_xg_for": avg_xg_for, "avg_xg_against": avg_xg_against,
        "conversion_rate": conversion_rate,
        "clean_sheets": clean_sheets,
        # splits (home / away) totals & counts
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
        # form & streak
        "streak": streak, "streak_type": streak_type,
        "chart_form": json.dumps(chart_form),
        "chart_goals_xg": json.dumps(chart_goals_xg),
        "next_pred": next_pred,
        # helper raw arrays for debugging if needed
        "recent_goals_raw": recent_goals,
        "recent_xg_raw": recent_xg,
    }

    return render(request, "coach_dashboard.html", context)

def team_compare(request, match_id):
    """
    Compare home vs away for the given match id.
    Produces raw season stats + normalized radar values (0..1).
    """
    match = get_object_or_404(Match, id=match_id)
    season = match.season

    def team_season_stats(team, season):
        """
        Compute raw season stats for `team` in `season` using played matches only.
        Returns dict with totals and derived metrics.
        """
        qs = Match.objects.filter(
            Q(home_team=team) | Q(away_team=team),
            season=season,
            played=True
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

            # Goals (y_home_goals / y_away_goals) - careful with None
            if m.y_home_goals is not None and m.y_away_goals is not None:
                gf = (m.y_home_goals if is_home else m.y_away_goals)
                ga = (m.y_away_goals if is_home else m.y_home_goals)
                if gf is not None:
                    goals_for += float(gf); gf_count += 1
                if ga is not None:
                    goals_against += float(ga); ga_count += 1
                if ga == 0:
                    clean_sheets += 1

            # xG values (home_xg / away_xg) - careful with None
            home_xg_val = m.home_xg
            away_xg_val = m.away_xg
            if is_home:
                if home_xg_val is not None:
                    xg_for += float(home_xg_val); xg_for_count += 1
                if away_xg_val is not None:
                    xg_against += float(away_xg_val); xg_against_count += 1
            else:
                if away_xg_val is not None:
                    xg_for += float(away_xg_val); xg_for_count += 1
                if home_xg_val is not None:
                    xg_against += float(home_xg_val); xg_against_count += 1

            # points from match result (H/D/A are relative to the home team)
            if m.result == "D":
                points += 1
            elif m.result in ("H", "A"):
                team_won = (m.result == "H" and is_home) or (m.result == "A" and not is_home)
                points += 3 if team_won else 0

        # Derived metrics (safely)
        ppg = round(points / matches, 3) if matches > 0 else None
        avg_gf = round(goals_for / gf_count, 3) if gf_count > 0 else None
        avg_ga = round(goals_against / ga_count, 3) if ga_count > 0 else None
        avg_xg_for = round(xg_for / xg_for_count, 3) if xg_for_count > 0 else None
        avg_xg_against = round(xg_against / xg_against_count, 3) if xg_against_count > 0 else None
        clean_sheet_pct = round((clean_sheets / matches) * 100, 2) if matches > 0 else None

        conversion = None
        # conversion = total_goals / total_xg (avoid div by zero)
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

    # Get teams
    home_team = match.home_team
    away_team = match.away_team

    home_stats = team_season_stats(home_team, season)
    away_stats = team_season_stats(away_team, season)

    # Radar metrics (raw)
    def safe_num(x):
        return float(x) if (x is not None) else 0.0

    home_raw = [
        safe_num(home_stats["avg_gf"]),
        safe_num(home_stats["avg_ga"]),
        safe_num(home_stats["avg_xg_for"]),
        safe_num(home_stats["avg_xg_against"]),
        safe_num(home_stats["ppg"]),
        safe_num(home_stats["clean_sheet_pct"]),
        safe_num(home_stats["conversion"])
    ]
    away_raw = [
        safe_num(away_stats["avg_gf"]),
        safe_num(away_stats["avg_ga"]),
        safe_num(away_stats["avg_xg_for"]),
        safe_num(away_stats["avg_xg_against"]),
        safe_num(away_stats["ppg"]),
        safe_num(away_stats["clean_sheet_pct"]),
        safe_num(away_stats["conversion"])
    ]

    # Fixed sensible metric ranges (min, max) — absolute scale for normalization
    metric_ranges = [
        (0.0, 3.0),    # Avg Goals For
        (0.0, 3.0),    # Avg Goals Against  (invert)
        (0.0, 3.0),    # Avg xG For
        (0.0, 3.0),    # Avg xG Against (invert)
        (0.0, 3.0),    # Points / game
        (0.0, 100.0),  # Clean Sheet % (0-100)
        (0.0, 2.0),    # Conversion (goals/xG) allow up to 2
    ]

    def norm_value(val, vmin, vmax):
        """Clip to [vmin, vmax] then scale to [0,1]. None -> 0.0"""
        if val is None:
            return 0.0
        v = max(min(float(val), vmax), vmin)
        if vmax - vmin <= 0:
            return 0.0
        return (v - vmin) / (vmax - vmin)

    normalized_home = []
    normalized_away = []

    for i in range(len(home_raw)):
        vmin, vmax = metric_ranges[i]
        a = home_raw[i]
        b = away_raw[i]

        na = norm_value(a, vmin, vmax)
        nb = norm_value(b, vmin, vmax)

        # invert for "Against" metrics (smaller is better)
        if i in (1, 3):
            na = 1.0 - na
            nb = 1.0 - nb

        normalized_home.append(round(max(0.0, min(1.0, na)), 3))
        normalized_away.append(round(max(0.0, min(1.0, nb)), 3))

    radar_labels = [
        "Avg Goals For",
        "Avg Goals Against",
        "Avg xG For",
        "Avg xG Against",
        "Points / game",
        "Clean Sheet %",
        "Conversion"
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


def match_detail(request, match_id):
    """
    Display detailed info about a single match with enhanced analytics.
    """
    match = get_object_or_404(Match, id=match_id)
    
    # Get recent form (last 5 matches before this match)
    home_recent = Match.objects.filter(
        Q(home_team=match.home_team) | Q(away_team=match.home_team),
        date__lt=match.date,
        season=match.season
    ).order_by('-date')[:5]
    
    away_recent = Match.objects.filter(
        Q(home_team=match.away_team) | Q(away_team=match.away_team),
        date__lt=match.date,
        season=match.season
    ).order_by('-date')[:5]
    
    # Calculate form (points from last 5)
    def calculate_form(team, matches):
        points = 0
        results = []
        for m in matches:
            if m.home_team == team:
                if m.result == 'H':
                    points += 3
                    results.append('W')
                elif m.result == 'D':
                    points += 1
                    results.append('D')
                else:
                    results.append('L')
            else:
                if m.result == 'A':
                    points += 3
                    results.append('W')
                elif m.result == 'D':
                    points += 1
                    results.append('D')
                else:
                    results.append('L')
        return points, results[::-1]  # Reverse to show oldest to newest
    
    home_form_points, home_form_results = calculate_form(match.home_team, home_recent)
    away_form_points, away_form_results = calculate_form(match.away_team, away_recent)
    
    # Head-to-head history (last 5 meetings)
    h2h_matches = Match.objects.filter(
        Q(home_team=match.home_team, away_team=match.away_team) |
        Q(home_team=match.away_team, away_team=match.home_team),
        date__lt=match.date
    ).order_by('-date')[:5]
    
    # Calculate h2h stats
    home_wins = 0
    away_wins = 0
    draws = 0
    for h2h in h2h_matches:
        if h2h.home_team == match.home_team:
            if h2h.result == 'H':
                home_wins += 1
            elif h2h.result == 'A':
                away_wins += 1
            else:
                draws += 1
        else:
            if h2h.result == 'H':
                away_wins += 1
            elif h2h.result == 'A':
                home_wins += 1
            else:
                draws += 1
    
    # Season averages
    home_season_stats = Match.objects.filter(
        Q(home_team=match.home_team) | Q(away_team=match.home_team),
        season=match.season,
        date__lt=match.date
    ).aggregate(
        avg_gf=Avg('home_gf', filter=Q(home_team=match.home_team)),
        avg_ga=Avg('home_ga', filter=Q(home_team=match.home_team)),
        avg_xg=Avg('home_xg', filter=Q(home_team=match.home_team)),
    )
    
    away_season_stats = Match.objects.filter(
        Q(home_team=match.away_team) | Q(away_team=match.away_team),
        season=match.season,
        date__lt=match.date
    ).aggregate(
        avg_gf=Avg('away_gf', filter=Q(away_team=match.away_team)),
        avg_ga=Avg('away_ga', filter=Q(away_team=match.away_team)),
        avg_xg=Avg('away_xg', filter=Q(away_team=match.away_team)),
    )
    
    # Calculate performance vs expectation
    home_overperformance = None
    away_overperformance = None
    if match.home_xg and match.y_home_goals is not None:
        home_overperformance = float(match.y_home_goals) - float(match.home_xg)
    if match.away_xg and match.y_away_goals is not None:
        away_overperformance = float(match.y_away_goals) - float(match.away_xg)
    
    context = {
        "match": match,
        "home_form_points": home_form_points,
        "home_form_results": home_form_results,
        "away_form_points": away_form_points,
        "away_form_results": away_form_results,
        "h2h_matches": h2h_matches,
        "h2h_home_wins": home_wins,
        "h2h_away_wins": away_wins,
        "h2h_draws": draws,
        "home_season_stats": home_season_stats,
        "away_season_stats": away_season_stats,
        "home_overperformance": home_overperformance,
        "away_overperformance": away_overperformance,
    }
    return render(request, "match_detail.html", context)

