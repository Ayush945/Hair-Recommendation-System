from .models import FaceShape, Hairstyle
import joblib
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from pathlib import Path
from keras_facenet import FaceNet
from django.shortcuts import render
import os
import base64
from django.views.decorators.csrf import csrf_exempt
#from .swaphair import swap_hair
from .anotherswaphair import swap_hair

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#Load the SVM model and the label decoder
svm_model = joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\FaceNetTrainedSVM.joblib')
label=joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\label_encoder_SVM.joblib')

#Function to process the input image and return the results
def webcam_face(request):
    if request.method == 'POST':
        try:
            image_file = request.POST.get('captured_image')
            if image_file:
                image_file=base64.b64decode(image_file.split(',')[1])
                nparr = np.frombuffer(image_file, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


                preprocessed_image = preprocess_image(frame)


                prediction = svm_model.predict(preprocessed_image)

                face_shape=label.inverse_transform(prediction)

                hairstyles = get_hairstyles_for_face_shape(face_shape[0])
                # image_store = []
                # for hairstyle in hairstyles:
                #     swapped_image = swap_hair(frame, hairstyle.image_path)
                #     ret, buffer = cv2.imencode('.jpg', swapped_image)
                #     image_as_string = base64.b64encode(buffer).decode('utf-8')
                #     image_store.append(image_as_string)
                    
                return render(request, 'preditedFace_webcam.html', {
                    'prediction': face_shape[0],
                    'hairstyles': hairstyles
                })
            else:
                return render(request, 'predictedFace.html', {'prediction': 'No image data received'})
        except:
            return render(request, 'error_handle.html')
    else:
        return render(request, 'predictedFace.html', {'prediction': 'Unable To Classify'})

#function to preprocess the image
def preprocess_image(image):
   detector =MTCNN()
   t_im=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   x,y,w,h=detector.detect_faces(t_im)[0]['box']
   t_im=t_im[y:y+h,x:x+w]
   t_im=cv2.resize(t_im,(160,160))
   test_im=get_embedding(t_im)
   test_im=[test_im]
   return test_im


#function to processes a facial image to generate a 512-dimensional embedding (feature vector)
def get_embedding(face_img):
    embedder=FaceNet()
    face_img=face_img.astype('float32')#3D(160X160X3)
    face_img=np.expand_dims(face_img,axis=0)
    #4D (NONEX160X160X3)
    yhat=embedder.embeddings(face_img)
    return yhat[0]  # 512D image (1x1x12)

#Function to return harstyle based on face shape
def get_hairstyles_for_face_shape(face_shape_name):
    try:
        face_shape = FaceShape.objects.get(faceShape=face_shape_name)
        hairstyles = Hairstyle.objects.filter(face_shape=face_shape)
        return hairstyles
    except FaceShape.DoesNotExist:
        return None