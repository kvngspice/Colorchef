from rest_framework import serializers
from .models import UploadedImage

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedImage
        fields = '__all__'
