import cv2
from keras.models import load_model
from keras.losses import mean_squared_error
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from django.shortcuts import render
import base64

# Load the model
model = load_model(r'E:\Class\Course Material\L6\Sem 2\Models\model.h5', custom_objects={'mse': mean_squared_error})

def load_image(path, as_gray=False):
    return io.imread(path, plugin='matplotlib', as_gray=as_gray)

def get_gray_scale(image):
    return rgb2gray(image)

def set_scale(image, shape):
    return resize(image, shape)

def get_image_for_network(path):
    if isinstance(path, str):
        img = load_image(path, True)
    else:
        img = get_gray_scale(path)
    img = set_scale(img, (224, 224))
    img = img.reshape((224, 224, 1))
    return img

def get_prediction(path, original_shape):
    img = get_image_for_network(path)
    res = model.predict(np.asarray([img]))[0]
    return resize(res.reshape((224, 224)), original_shape)

def get_rgb_tuple(h):
    h = h.lstrip('#')
    if len(h) != 6:
        raise ValueError("Invalid color format. Expected a 6-character hexadecimal string.")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
def colorize_image(image_path, color):
    thresh = 0.5
    if isinstance(image_path, str):
        source_img = load_image(image_path)
    else:
        source_img = image_path
    
    original_shape = source_img.shape[:2]
    res = get_prediction(image_path, original_shape)
    
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    erosion_dst = cv2.erode(res, element)
    
    r, g, b = get_rgb_tuple(color) 
    bgr_color = (b, g, r)
    
    mask = np.zeros((*original_shape, 3), dtype=np.uint8)
    mask[:, :, 0] = np.full(original_shape, bgr_color[0], dtype=np.uint8)
    mask[:, :, 1] = np.full(original_shape, bgr_color[1], dtype=np.uint8) 
    mask[:, :, 2] = np.full(original_shape, bgr_color[2], dtype=np.uint8)

    real_img = source_img.copy()
    blend_factor = 0.35
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            if erosion_dst[i, j] > thresh:
                real_img[i, j, 0] = bgr_color[0] * blend_factor + real_img[i, j, 0] * (1 - blend_factor)
                real_img[i, j, 1] = bgr_color[1] * blend_factor + real_img[i, j, 1] * (1 - blend_factor)
                real_img[i, j, 2] = bgr_color[2] * blend_factor + real_img[i, j, 2] * (1 - blend_factor)

    real_img = cv2.medianBlur(real_img, 3)
    return real_img

#Accepts photo and changes its hair
def accept_input(request):
    if request.method == 'POST':
        image_file = request.POST.get('captured_image')
        selected_color = request.POST.get('selected_color')
        if image_file and selected_color:
            image_file = base64.b64decode(image_file.split(',')[1])
            nparr = np.frombuffer(image_file, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            prediction = colorize_image(frame, selected_color)
            ret, buffer = cv2.imencode('.jpg', prediction)
            image_as_string = base64.b64encode(buffer).decode('utf-8')
            return render(request, 'colorChange.html', {
                'prediction': image_as_string,
            })
    else:
        return render(request, 'predictedFace.html', {'prediction': 'Unable To Colorize'})
