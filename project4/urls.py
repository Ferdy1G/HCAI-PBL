from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("design1/", views.design1_pairwise, name="design1"),
    path("design2/", views.design2_ranking, name="design2"),
    path("get_candidates/", views.get_candidates, name="get_candidates"),
    path("submit_preference/", views.submit_preference, name="submit_preference"),
    path("complete/", views.complete, name="complete"),
]