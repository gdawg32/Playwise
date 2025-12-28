from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('manager_login/', manager_login, name='manager_login'),
    path('manager_logout/', manager_logout, name='manager_logout'),
    path('manager_dashboard/', coach_dashboard, name='manager_dashboard'),

    path("coach/role-leaders/", coach_role_leaders_page, name="coach_role_leaders_page"),
    path("api/coach/role-leaders/", coach_role_leaders_api, name="coach_role_leaders_api"),

    path(
        "coach/role-suitability/",
        coach_role_suitability_page,
        name="coach_role_suitability"
    ),
    path(
        "coach/api/role-suitability/",
        coach_role_suitability_api,
        name="coach_role_suitability_api"
    ),



    path("predict-page/", predict_page, name="predict_page"),  
    path("predict/", predict_api, name="predict_api"),

    path('compare/<int:match_id>/', team_compare, name='team_compare'),
    path("match/<int:match_id>/", match_detail, name="match_detail"),
    path(
        "admin/player-analysis/",
        player_analysis,
        name="admin_player_analysis"
    ),
    path("player/<str:player_id>/", player_detail, name="player_detail"),


    

    

    path("api/teams/", api_teams_by_competition, name="api_teams"),
    path("api/players/", api_players_by_team, name="api_players"),



    path("admin-login/", admin_login, name="admin_login"),
    path("admin-dashboard/", admin_dashboard, name="admin_dashboard"),

    path('admin/run-pipeline/', admin_run_pipeline, name='admin_run_pipeline'),
    path('admin/pipeline-status/', admin_pipeline_status, name='admin_pipeline_status'),
    path('admin/run-player-stats/', admin_run_player_stats_update, name='admin_run_player_stats_update'),
    path('admin/player-stats-status/', admin_player_stats_status, name='admin_player_stats_status'),


    path("player_comparison/", player_comparison, name="player_comparision"),
    path("player-comparison/radar/", player_comparison_radar, name="player_comparison_radar"),
    path("team-comparison/", team_comparison, name="team_comparison"),
    path("team-comparison/bumpy/", team_comparison_bumpy, name="team_comparison_bumpy"),
    path("pizza/", player_pizza_page, name="player_pizza"),
    path("api/pizza/", player_pizza_api, name="player_pizza_api"),
    path("stat-leaders/", stat_leaders, name="stat_leaders"),
    path("api/stat-leaders/", stat_leaders_api, name="stat_leaders_api"),

    path("coach/xi-suggestion/", xi_suggestion_page, name="xi_suggestion_page"),
    path("coach/xi-suggestion-api/", xi_suggestion_api, name="xi_suggestion_api"),
    path("api/xi/pitch/", xi_pitch_image_api, name="xi_pitch_image_api"),



]
