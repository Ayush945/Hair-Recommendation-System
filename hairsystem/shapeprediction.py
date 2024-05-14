from .models import FaceShape, Hairstyle
import joblib
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
from keras_facenet import FaceNet
from django.shortcuts import render
import tkinter as tk
from PIL import Image,ImageTk
import os
from django.http import JsonResponse,HttpResponse
import base64
from django.views.decorators.csrf import csrf_exempt

#Load the model and label
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
svm_model = joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\FaceNetTrainedSVM.joblib')
label=joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\label_encoder_SVM.joblib')


def predict_face_shape(request):
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.FILES.get('image_file')
        
        # Load the image file using OpenCV
        image_data = image_file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess the image
        preprocessed_image = preprocess_image(frame)

        # Make a prediction
        prediction = svm_model.predict(preprocessed_image)

        face_shape=label.inverse_transform(prediction)
        
        # Render the prediction in the template
        hairstyles = get_hairstyles_for_face_shape(face_shape[0])
        return render(request, 'predictedFace.html', {'prediction': face_shape[0],'hairstyles': hairstyles})
    else:
        return render(request, 'predictedFace.html',{'prediction':'Unable To Classify'})

def preprocess_image(image):
   detector =MTCNN()
   t_im=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   x,y,w,h=detector.detect_faces(t_im)[0]['box']
   t_im=t_im[y:y+h,x:x+w]
   t_im=cv2.resize(t_im,(160,160))
   test_im=get_embedding(t_im)
   test_im=[test_im]
   return test_im



def get_embedding(face_img):
    embedder=FaceNet()
    face_img=face_img.astype('float32')#3D(160X160X3)
    face_img=np.expand_dims(face_img,axis=0)
    #4D (NONEX160X160X3)
    yhat=embedder.embeddings(face_img)
    return yhat[0]  # 512D image (1x1x12)

def get_hairstyles_for_face_shape(face_shape_name):
    try:
        face_shape = FaceShape.objects.get(faceShape=face_shape_name)
        hairstyles = Hairstyle.objects.filter(face_shape=face_shape)
        return hairstyles
    except FaceShape.DoesNotExist:
        return None
