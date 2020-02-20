from django import forms
from .models import *

class ImageUpload(forms.ModelForm):

    class Meta:
        model = plant
        fields = ['Main_Img']
