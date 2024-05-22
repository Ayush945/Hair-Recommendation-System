import numpy as np
import cv2 
import dlib 
from sklearn.cluster import KMeans
from math import degrees
from .models import FaceShape, Hairstyle
from django.shortcuts import render
import base64
from .swaphair import swap_hair

#haarcascade for detecting faces
face_cascade_path = r'E:\Class\Course Material\L6\Sem 2\Models\haarcascade_frontalface_default.xml'

#File for detecting facial landmarks
predictor_path = r'E:\Class\Course Material\L6\Sem 2\Models\shape_predictor_68_face_landmarks.dat'


def ruleBasedPredictWebcam(request):
    if request.method == 'POST':
        try:
            image_file = request.POST.get('captured_image')
            if image_file:
            
                image_file=base64.b64decode(image_file.split(',')[1])
                nparr = np.frombuffer(image_file, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                preprocessedImage = preprocess_image(frame)
                
                hairstyles = get_hairstyles_for_face_shape(preprocessedImage)
                # image_store = []
                # for hairstyle in hairstyles:
                #     swapped_image = swap_hair(frame, hairstyle.image_path)
                #     ret, buffer = cv2.imencode('.jpg', swapped_image)
                #     image_as_string = base64.b64encode(buffer).decode('utf-8')
                #     image_store.append(image_as_string)
                    
                return render(request, 'preditedFace_webcam.html', {
                    'prediction': preprocessedImage,
                    'hairstyles': hairstyles
                })
        except:
            return render(request,"error_handle.html")
    else:
        return render(request, 'predictedFace.html',{'prediction':'Unable To Classify'})

def preprocess_image(frame):
    faceCascade = cv2.CascadeClassifier(face_cascade_path)
    predictor = dlib.shape_predictor(predictor_path)
    image=frame
    image=cv2.resize(image,(500,500))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gauss=cv2.GaussianBlur(gray,(3,3),0)
    faces = faceCascade.detectMultiScale(
    gauss,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100,100),
    flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        rect_dlib=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        detected_landmarks=predictor(image,rect_dlib).parts()
        landmarks=np.matrix([[i.x,i.y] for i in detected_landmarks])
        temp=image.copy()

        forehead = temp[y:y+int(0.25*h), x:x+w]
        rows,cols, bands = forehead.shape
        print(rows,cols)
        X = forehead.reshape(rows*cols,bands)
        
        #kmeans 
        kmeans=KMeans(n_clusters=2,init="k-means++",max_iter=300,n_init=10,random_state=0)
        y_kmeans=kmeans.fit_predict(X)
        for i in range(0,rows):
            for j in range(0,cols):
                    if y_kmeans[i*cols+j]==True:
                        forehead[i][j]=[255,255,255] #white
                    if y_kmeans[i*cols+j]==False:
                        forehead[i][j]=[0,0,0] #black
   
        #Midpoint of forehead and find right and left boundaries of the forehead based on pixel value.
    forehead_mid=[int(cols/2),int(rows/2)]
    lef=0

    pixel_value=forehead[forehead_mid[1],forehead_mid[0]]
    for i in range(0,cols):
        if forehead[forehead_mid[1],forehead_mid[0]-i].all()!=pixel_value.all():
            lef=forehead_mid[0]-i
            break
    left=[lef,forehead_mid[1]]
    rig=0
    for i in range(0,cols):
        if forehead[forehead_mid[1],forehead_mid[0]+i].all()!=pixel_value.all():
            rig=forehead_mid[0]+i
            break
    right=[rig,forehead_mid[1]]
    
    face_shape=classify_face_shape(landmarks,right,left,y,x)
    return face_shape


def classify_face_shape(landmarks,right,left,y,x):
    line1 = np.subtract(right + y, left + x)[0]
    linepointleft = (landmarks[1, 0], landmarks[1, 1])
    linepointright = (landmarks[15, 0], landmarks[15, 1])
    line2 = np.subtract(linepointright, linepointleft)[0]


    linepointleft = (landmarks[3, 0], landmarks[3, 1])
    linepointright = (landmarks[13, 0], landmarks[13, 1])
    line3 = np.subtract(linepointright, linepointleft)[0]

    linepointbottom = (landmarks[8, 0], landmarks[8, 1])
    linepointtop = (landmarks[8, 0], y)
    line4 = np.subtract(linepointbottom, linepointtop)[1]

    ratio1 = line1 / line2
    ratio2 = line1 / line3
    ratio3 = line2 / line3
    ratio4 = line4 / line1
    
    if ratio1 > 1.1:
        return "Oval"
    elif ratio1 < 0.9:
        return "Round"
    elif ratio2 > 1.1 and ratio3 > 1.1:
        return "Square"
    elif ratio4 > 1.1:
        return "Oblong"
    else:
        return "Heart"


def get_hairstyles_for_face_shape(face_shape_name):
    try:
        face_shape = FaceShape.objects.get(faceShape=face_shape_name)
        hairstyles = Hairstyle.objects.filter(face_shape=face_shape)
        return hairstyles
    except FaceShape.DoesNotExist:
        return None
