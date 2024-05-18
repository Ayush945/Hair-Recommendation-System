from .models import Hairstyle, FaceShape
from django.shortcuts import render, redirect
from .forms import FaceShapeForm

def add_data(request):
  if request.method == 'POST':
    givenFace=request.POST.get('face_shape')
    faceshape=FaceShape(faceShape=givenFace)
    faceshape.save()
    return render(request, 'add_data.html')
  return render(request, 'add_data.html')

def add_hair_data(request):
  return False
