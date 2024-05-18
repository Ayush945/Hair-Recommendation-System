from django.db import models

# Create your models here.
class FaceShape(models.Model):
    faceShape=models.CharField(max_length=100)


class Hairstyle(models.Model):
    hairName=models.CharField(max_length=100)
    image_path=models.ImageField(upload_to='hairstyles/', default='default.jpg')
    face_shape=models.ForeignKey(FaceShape,on_delete=models.CASCADE,related_name='face_hairstyle')
