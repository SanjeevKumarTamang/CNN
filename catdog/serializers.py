from rest_framework import serializers
from .models import Lead, SampleImage


class LeadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Lead
        fields = ('id', 'name', 'email', 'message')

class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = SampleImage
        fields = ('id','name')