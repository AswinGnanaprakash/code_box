from django.shortcuts import render
from django.shortcuts import render,get_object_or_404,redirect, HttpResponse


import json
import requests
import os
import base64
from .models import plant
from .form import *

# import settings
# Create your views here.


def image_as_base64(image_file, format='png'):
    """
    :param `image_file` for the complete path of image.
    :param `format` is format for image, eg: `png` or `jpg`.
    """
    if not os.path.isfile(image_file):
        return None

    encoded_string = ''
    print("working")
    with open(image_file, 'rb') as img_f:
        encoded_string = base64.b64encode(img_f.read())
    image_encoded = 'data:image/{};base64,'.format(format) + str(encoded_string)

    image_file = image_encoded.replace("b'", "")


    url = "http://127.0.0.1:8000/predict"
    data = {'plant_image':image_file}
    x = requests.post(url, data)
    print(x)
    values = x.json()
    
    return values



def home_page(request):
    if request.method == 'POST':
        form = ImageUpload(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            issue_list = plant.objects.all()
            image_file = str(issue_list.last().upload_your_image)
            cwd = os.getcwd()
            val = image_as_base64(cwd+'/media/'+image_file)
            answers  = val['data']
            suggestion = val['suggestion']
            return render(request, 'show.html', {'values' :answers, 'suggestion' : suggestion})

    else:
        form = ImageUpload()
    return render(request, 'base.html', {'form' : form})

# def
