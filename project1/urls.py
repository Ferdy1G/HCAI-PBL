from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.workspace_view, name='workspace_view'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('explore/', views.explore_view, name='explore_view'),
    path('train/', views.train_view, name='train_view'),
    
    # Data Cleaning Routes
    path('clean/reset/', views.clean_reset_view, name='clean_reset'),
    path('clean/drop-nulls/', views.clean_drop_nulls_view, name='clean_drop_nulls'),
    path('clean/fill-missing/', views.clean_fill_missing_view, name='clean_fill_missing'),
    path('clean/filter-distribution/', views.clean_filter_distribution_view, name='clean_filter_distribution'),
]