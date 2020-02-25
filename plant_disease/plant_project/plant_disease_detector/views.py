import datetime
import pickle
import json
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from api.settings import BASE_DIR
from custom_code import image_converter
# from tensorflow import keras
from tensorflow.python.keras.models import load_model

import os
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@api_view(['GET'])
def __index__function(request):
    start_time = datetime.datetime.now()
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_time_ms = (elapsed_time.days * 86400000) + (elapsed_time.seconds * 1000) + (elapsed_time.microseconds / 1000)
    return_data = {
        "error" : "0",
        "message" : "Successful",
        "restime" : elapsed_time_ms
    }
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')

@api_view(['POST','GET'])
def predict_plant_disease(request):
    # try:
    if request.method == "GET" :
        return_data = {
            "error" : "0",
            "message" : "Working!!!"
        }
    else:
        if request.body:
            request_data = request.data["plant_image"]
            header, image_data = request_data.split(';base64,')
            image_array, err_msg = image_converter.convert_image(image_data)
            if err_msg == None :
                model_file = f"{BASE_DIR}/ml_files/sequential.pkl"
                saved_classifier_model = pickle.load(open(model_file, 'rb'))
                prediction = saved_classifier_model.predict(image_array)
                label_binarizer = pickle.load(open(f"{BASE_DIR}/ml_files/label_bin.pkl",'rb'))
    
                return_data = {
                    "error" : "0",
                    "data" : f"{label_binarizer.inverse_transform(prediction)[0]}"
                }            
    
    return HttpResponse(json.dumps(return_data), content_type='application/json; charset=utf-8')
