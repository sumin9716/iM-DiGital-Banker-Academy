"""
URL configuration for the Segment Radar API.
"""
from django.urls import path
from . import views

urlpatterns = [
    # Root & Health
    path('', views.RootView.as_view(), name='root'),
    path('health/', views.HealthCheckView.as_view(), name='health'),
    
    # Dashboard
    path('overview/', views.OverviewView.as_view(), name='overview'),
    path('kpi-trends/', views.KPITrendsView.as_view(), name='kpi-trends'),
    
    # Segments
    path('segments/', views.SegmentListView.as_view(), name='segment-list'),
    path('segments/<str:segment_id>/', views.SegmentDetailView.as_view(), name='segment-detail'),
    path('segments/<str:segment_id>/history/', views.SegmentHistoryView.as_view(), name='segment-history'),
    
    # Forecasts
    path('forecasts/', views.ForecastListView.as_view(), name='forecast-list'),
    
    # Watchlist
    path('watchlist/', views.WatchlistView.as_view(), name='watchlist'),
    
    # Recommendations
    path('recommendations/', views.RecommendationsView.as_view(), name='recommendations'),
    
    # Filters
    path('filters/', views.FilterOptionsView.as_view(), name='filters'),
    
    # Admin
    path('refresh/', views.RefreshDataView.as_view(), name='refresh'),
]
