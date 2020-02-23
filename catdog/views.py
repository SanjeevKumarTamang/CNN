from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView

from .MLModels import CatVsDogTester
from .models import Lead, SampleImage
from .serializers import LeadSerializer
from rest_framework import generics, status
import json
import logging

logger = logging.getLogger(__name__)
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


class LeadListCreate(generics.ListCreateAPIView):
    print('inside this')
    queryset = Lead.objects.all()
    serializer_class = LeadSerializer


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, format=None):
        if 'image' not in request.data:
            logger.error('Something went wrong!')
            raise ParseError("Empty content")
        try:
            img = request.data['image']
            predictedAnimal=CatVsDogTester.guessTheImage(img)
            x = {
                "name": predictedAnimal
            }
            return Response(data=json.dumps(x), content_type='application/json')
        except Exception as e:
            logger.error(e)

        # return HttpResponse(predictedAnimal)

    def delete(self, request, format=None):
        SampleImage.image.delete(save=True)
        return Response(status=status.HTTP_204_NO_CONTENT)