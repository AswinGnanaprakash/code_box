from django.db import models

# Create your models here.
class plant(models.Model):
    upload_your_image = models.ImageField(upload_to='images/')
