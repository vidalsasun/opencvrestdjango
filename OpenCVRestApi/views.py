# Create your views here.
from rest_framework.decorators import  action
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import ImageCVSerializer
from .mrz import detect_mrz
from .models import ImageCV
from rest_framework import viewsets
import io
from rest_framework.parsers import JSONParser
from rest_framework.permissions import IsAuthenticated 
from django.core.serializers import serialize
import json

def convert_to_dict(obj):
    """
    A function takes in a custom object and returns a dictionary representation of the object.
    This dict representation includes meta data such as the object's module and class names.
    """    
    #  Populate the dictionary with object meta data 
    obj_dict = {
        "__class__": obj.__class__.__name__,
        "__module__": obj.__module__
    }    
    #  Populate the dictionary with object properties
    obj_dict.update(obj.__dict__)    
    return obj_dict

class ImageCVSet(viewsets.ModelViewSet):
    permission_classes = (IsAuthenticated,)
    queryset = ImageCV.objects.all().order_by('name')
    serializer_class = ImageCVSerializer

    def create(self, request, *args, **kwargs):
        serializer = ImageCVSerializer(data=request.data)
        serializer.is_valid()
        # True
        serializer.validated_data                
        base64ret = detect_mrz.detect(serializer.data["base64"])       
        jsonData = json.dumps(base64ret,default=convert_to_dict,indent=4, sort_keys=True)
        return Response(jsonData)
    
