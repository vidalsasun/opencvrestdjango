# OpenCVRestApi/urls.py
from django.urls import include, path
from rest_framework import routers
from . import views
from .models import ImageCV
from .views import ImageCVSet

router = routers.DefaultRouter()
router.register(r'imagesCV', views.ImageCVSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),    
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]