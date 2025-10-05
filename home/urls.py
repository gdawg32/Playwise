from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('manager_login/', manager_login, name='manager_login'),
    path('manager_logout/', manager_logout, name='manager_logout'),
    path('manager_dashboard/', coach_dashboard, name='manager_dashboard'),

    path("predict-page/", predict_page, name="predict_page"),  
    path("predict/", predict_api, name="predict_api"),

    path('compare/<int:match_id>/', team_compare, name='team_compare'),

]
