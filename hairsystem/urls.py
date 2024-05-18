from django.urls import path
from . import shapeprediction
from .views import webcam_page,photo_page,home,rule_based_photo_page,rule_based_webcam_page,add_data_page
from .upload_image import webcam_face
from .rule_based_photo import ruleBasedPredictPhoto
from.rule_based_webcam import ruleBasedPredictWebcam
from .data_addition import add_data,add_hair_data
urlpatterns = [
    #path for home page
    path('',home),

    #path for SVM photo and webcam prediction 
    path('predict_face_shape/', shapeprediction.predict_face_shape, name='predict_face_shape'),
    path('photo_predict/',webcam_face,name='photo_predict'),

    #path for html page of svm photo and webcam
    path('svm_photo',photo_page),
    path('svm_webcam/',webcam_page),

    #path for html page for Rule Based photo and webcam
    path('rule_based_photo',rule_based_photo_page),
    path('rule_based_webcam/',rule_based_webcam_page),

    #path for Rule Based photo and webcam prediction
    path('rule_based_predict_photo',ruleBasedPredictPhoto,name='rule_based_predict_photo'),
    path('rule_based_predict_webcam',ruleBasedPredictWebcam,name='rule_based_predict_webcam'),

    #path to add face shape and hairstyle to database
    path('add_data/', add_data, name='add_data'),
    path('form',add_data_page),
    path('add_hair_data',add_hair_data)
]                   