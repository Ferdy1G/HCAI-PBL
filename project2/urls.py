from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_model, name='train_model'),
    path("get-sample/", views.get_sample, name="get_sample"),
    path("generate-counterfactuals/", views.generate_counterfactuals, name="generate_counterfactuals"),
    path('get-pdp-plot/', views.get_pdp_plot, name='get_pdp_plot'),
]