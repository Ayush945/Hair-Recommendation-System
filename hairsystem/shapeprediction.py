from .models import FaceShape, Hairstyle
import joblib
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from django.shortcuts import render
import os
import base64
from django.views.decorators.csrf import csrf_exempt
#from .swaphair import swap_hair
from .anotherswaphair import swap_hair
#Load the model and label
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
svm_model = joblib.load(r'E:\Class\Course Material\L6\Sem 2\Models\FaceNetTrainedSVM.joblib')
label=joblib.load(r'E:\Class\Course Material\L6\Sem 2\Models\label_encoder_SVM.joblib')


def predict_face_shape(request):
    if request.method == 'POST':
        try:
        
            image_file = request.FILES.get('image_file')
            image_data = image_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            preprocessed_image = preprocess_image(frame)
            prediction = svm_model.predict(preprocessed_image)
            
            face_shape=label.inverse_transform(prediction)
            
            hairstyles = get_hairstyles_for_face_shape(face_shape[0])
            image_store = []
            for hairstyle in hairstyles:
                swapped_image = swap_hair(frame, hairstyle.image_path)
                ret, buffer = cv2.imencode('.jpg', swapped_image)
                image_as_string = base64.b64encode(buffer).decode('utf-8')
                image_store.append(image_as_string)
                
            return render(request, 'predictedFace.html', {
                'prediction': face_shape[0],
                'hairstyles': hairstyles,
                'image_data': image_store,
            })
        except:
            return render(request, 'error_handle.html')

    else:
        return render(request, 'predictedFace.html', {'prediction': 'Unable To Classify'})


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
    face_img=face_img.astype('float32')
    face_img=np.expand_dims(face_img,axis=0)
    yhat=embedder.embeddings(face_img)
    return yhat[0]

def get_hairstyles_for_face_shape(face_shape_name):
    try:
        face_shape = FaceShape.objects.get(faceShape=face_shape_name)
        hairstyles = Hairstyle.objects.filter(face_shape=face_shape)
        return hairstyles
    except FaceShape.DoesNotExist:
        return None

