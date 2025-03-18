from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from colors.views import home  # Import the new view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('colors.urls')),
    path('', home),  # Add this line to handle the root URL
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
