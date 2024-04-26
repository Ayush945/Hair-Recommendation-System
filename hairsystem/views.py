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

def capture_image(request):
    try:
        svm_model = joblib.load('C:\\\\Users\\\\admin\\\\Downloads\\\\svm_model.pkl')
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        cap1 = cv2.VideoCapture(cv2.CAP_DSHOW)
        ret, frame = cap1.read()
        if not ret:
            return HttpResponse(status=500)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_image)
            feature_array = []
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                face_features = [coord or 0  # Replace None with 0 (or other default value)
                                for landmark in face_landmarks.landmark
                                for coord in [landmark.x, landmark.y, landmark.z]]
                feature_array.append(face_features)
                logger.error(len(face_landmarks.landmark))
                
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            else:
                logger.error("No landmark detected")

            feature_array = np.array(feature_array)
            logger.error(feature_array.shape[0])  # Should be 1
            logger.error(feature_array.shape[1])  # Should be 1404

            _, jpeg = cv2.imencode('.jpg', frame)
            cap1.release()
            return HttpResponse(jpeg.tobytes(), content_type='image/jpeg')
    except Exception as e:
        logger.error(f"Error in capture_image view: {e}")
        return HttpResponse(status=500)

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