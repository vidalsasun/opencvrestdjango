# serializers.py
from rest_framework import serializers

from .models import ImageCV

class ImageCVSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = ImageCV
        fields = ('name', 'base64')        