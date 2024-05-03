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

            _, jpeg = cv2.imencode('.jpg', frame)
            cap1.release()
            return HttpResponse(jpeg.tobytes(), content_type='image/jpeg')
    except Exception as e:
        logger.error(f"Error in capture_image view: {e}")
        return HttpResponse(status=500)

