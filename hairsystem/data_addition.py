from .models import Hairstyle, FaceShape
from django.shortcuts import render, redirect
from .forms import FaceShapeForm


#function to add face to database
def add_data(request):
  if request.method == 'POST':
    givenFace=request.POST.get('face_shape')
    faceshape=FaceShape(faceShape=givenFace)
    faceshape.save()
    return render(request, 'add_data.html')
  return render(request, 'add_data.html')

#function to add hairstyle to database
def add_hair_data(request):
  if request.method == 'POST':
    givenHairStyle=request.POST.get('hair_style')
    image=request.POST.get('file')
    hairStyle=Hairstyle(hairName=givenHairStyle,face_shape_id=10,image_path=image)
    hairStyle.save()
    return render(request, 'add_hairstyle.html')
  return render(request, 'add_hairstyle.html')
