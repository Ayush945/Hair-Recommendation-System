from django import forms
from .models import FaceShape, Hairstyle

class FaceShapeForm(forms.ModelForm):
    class Meta:
        model = FaceShape
        fields = '__all__' 

class HairstyleForm(forms.ModelForm):
    class Meta:
        model = Hairstyle
        fields = ['hairName', 'image_path', 'face_shape'] 
