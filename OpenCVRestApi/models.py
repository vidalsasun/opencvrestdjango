# models.py
from django.db import models
class ImageCV(models.Model):
    name = models.CharField(max_length=60)
    base64 = models.TextField()
    def __str__(self):
        return self.name        