from django.urls import path
from .views import UploadMediaView

urlpatterns = [
    path('upload/', UploadMediaView.as_view(), name='upload'),
]
