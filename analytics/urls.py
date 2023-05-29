# Created by 'shubham.mallick' on 25-April-2023 4:45 pm


from django.contrib import admin
from django.urls import path, re_path, include
from . import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('simple_analytics', views.simple_analytics, name='simple_analytics'),
    re_path(r'^get_simple_analytics_data/$', views.get_simple_analytics_data, name='get_simple_analytics_data'),
    path('weekly_analytics', views.weekly_analytics, name='weekly_analytics'),
    re_path(r'^get_weekly_analytics_data/$', views.get_weekly_analytics_data, name='get_weekly_analytics_data'),
]


