from django.urls import path

from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('register/', views.register_view, name='register'),
    path('about/', views.about_view, name='about'),
    path('signin/', views.signin_view, name='signin'),
    path('signout/', views.signout_view, name='signout'),
    path('result/', views.result_view, name='result'),
    path('simulation/', views.simulation_view, name='simulation'),
    path('run-simulation/', views.run_simulation_view, name='run_simulation'),
    path('simulation-output/', views.simulation_output_view, name='simulation_output'),
    path('sse-updates/', views.sse_updates, name='sse_updates'),
]
