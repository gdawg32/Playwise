from django.contrib import admin
from .models import *
# Register your models here.

admin.site.register(Competition)
admin.site.register(Team)
admin.site.register(Match)
admin.site.register(Coach)
admin.site.register(PlayerMatchStat)
admin.site.register(PlayerSeasonStat)