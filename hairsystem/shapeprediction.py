import joblib
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
from keras_facenet import FaceNet
from django.shortcuts import render
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def predict_face_shape(request):
    #Load the model and label
    svm_model = joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\FaceNetTrainedSVM.joblib')
    label=joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\label_encoder_SVM.joblib')

    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Preprocess the image
    preprocessed_image = preprocess_image(frame)

    # Make a prediction
    prediction = svm_model.predict(preprocessed_image)

    face_shape=label.inverse_transform(prediction)
    # Render the prediction in the template
    return render(request, 'faceshape.html', {'prediction': face_shape})


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