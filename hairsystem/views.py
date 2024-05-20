import cv2
import logging
import mediapipe as mp
import numpy as np
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.shortcuts import render
import joblib
from itertools import chain
logger = logging.getLogger(__name__)



def webcam_page(request):
    return render(request,'webcam.html')

def photo_page(request):
    return render(request,'photo.html')

def home(request):
    return render(request,'home.html')

def rule_based_photo_page(request):
    return render(request,'rule_photo.html')

def rule_based_webcam_page(request):
    return render(request,'rule_webcam.html')

def add_data_page(request):
    return render(request,'add_data.html')

def add_hair(request):
    return render(request,'add_hairstyle.html')

def error_handle(request):
    return render(request,'error_handle.html')


