from django.urls import path
from . import views


urlpatterns = [
    path('', views.main_page, name='main_page'),
    path('tutorial/', views.tutorial, name='tutorial'),
    path('image/', views.image_upload, name='image_upload'),
    path('image/prediction/', views.display, name='display'),
    path('video/', views.video_stream, name='video_stream'),
    path('videoreader/', views.videoreader, name='videoreader'),
]
