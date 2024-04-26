import cv2
import logging
import openpyxl
import mediapipe as mp
import numpy as np
from django.http import HttpResponse, StreamingHttpResponse,JsonResponse
from django.shortcuts import render

logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'home.html')

def webcam_feed(request):
    try:
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error("Failed to open webcam.")
            return HttpResponse(status=500)

        def gen_frames():
            while True:
                success, frame = cap.read()
                if not success:
                    logger.error("Failed to read frame from webcam.")
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    

        return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in webcam_feed view: {e}")
        return HttpResponse(status=500)

