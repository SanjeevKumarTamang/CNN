from django.db import models

# Create your models here.
from django.db import models


class Lead(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.CharField(max_length=300)
    created_at = models.DateTimeField(auto_now_add=True)


class SampleImage(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to="imagestest/", null=True, blank=True)

    def __str__(self):
        return "{}".format(self.name)


