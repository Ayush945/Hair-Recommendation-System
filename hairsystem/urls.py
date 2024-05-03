from django.contrib import admin
from django.urls import path
from . import views,webcam,upload_image,shapeprediction

urlpatterns = [
    path('',webcam.home,name='home'),
    ]


# landmarks = np.array([[landmark.x / frame.shape[1], landmark.y / frame.shape[0]]
# for landmark in face_landmarks.landmark])
                    