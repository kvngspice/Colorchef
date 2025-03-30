from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.UploadMediaView.as_view(), name='upload'),
    path('posterize/', views.posterize_image, name='posterize_image'),
    path('posterize-svg/', views.posterize_svg, name='posterize_svg'),
    path('line-art/', views.convert_to_line_art, name='convert_to_line_art'),
    path('line-art-svg/', views.convert_to_line_art_svg, name='convert_to_line_art_svg'),
    path('blend-art/', views.blend_art, name='blend_art'),
]
