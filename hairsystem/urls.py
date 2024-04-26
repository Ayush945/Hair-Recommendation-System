from django.contrib import admin
from django.urls import path
from . import views,webcam

urlpatterns = [
    path('',webcam.home,name='home'),
    path('webcam_feed/',webcam.webcam_feed,name='webcam_feed'),
    path('capture_image/',views.capture_image,name='capture_image'),
    path('upload/',views.upload_image,name='upload_image'),
    ]


# landmarks = np.array([[landmark.x / frame.shape[1], landmark.y / frame.shape[0]]
# for landmark in face_landmarks.landmark])
                    