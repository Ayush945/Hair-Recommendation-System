import cv2
import logging
import mediapipe as mp
import numpy as np
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.shortcuts import render
import joblib
from itertools import chain
from .forms import UploadImageForm
logger = logging.getLogger(__name__)

def upload_image(request):
    if request.method=='POST':
        form=UploadImageForm(request.POST,request.FILES)
        if form.is_valid():
            image_file=request.FILES['image']
            image_data=image_file.read()

            results=facemesh_method(image_data)
            if results:
                 context = {'landmarks': results}
                # context={'landmarks':'nice'}
            else:
                context={'landmarks':0,'message':'No face detected'}
            return render(request,'upload_result.html',context)
    else:
        form=UploadImageForm()
    context={'form':form}
    return render(request,'Upload_form.html',context)

def facemesh_method(image_data):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # Convert image data to OpenCV image
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Preprocess image (e.g., convert to RGB, resize)
    # ... (preprocessing code)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        # Convert image to BGR format (expected by Face Mesh)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(image)

        # Extract and count landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                logger.error(len(face_landmarks.landmark))
                return landmarks

    return None 