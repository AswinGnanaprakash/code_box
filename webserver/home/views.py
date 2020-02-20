from django.shortcuts import render
from django.shortcuts import render,get_object_or_404,redirect, HttpResponse
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
    print("woprking")
    if not os.path.isfile(image_file):
        return None

    encoded_string = ''
    with open(image_file, 'rb') as img_f:
        encoded_string = base64.b64encode(img_f.read())
        print(encoded_string)
    print('data:image/%s;base64,%s' % (format, encoded_string))


def home_page(request):
    if request.method == 'POST':
        form = ImageUpload(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            issue_list = plant.objects.all()
            image_file = str(issue_list.last().Main_Img)
            cwd = os.getcwd()
            image_as_base64(cwd+'/media/'+image_file)

    else:
        form = ImageUpload()
    return render(request, 'base.html', {'form' : form})

# def
