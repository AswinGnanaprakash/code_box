from django.db import models

# Create your models here.
class plant(models.Model):
    Main_Img = models.ImageField(upload_to='images/')
