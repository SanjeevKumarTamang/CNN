from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/catdog/', views.LeadListCreate.as_view()),
    path('api/img/', views.FileUploadView.as_view()),
]
