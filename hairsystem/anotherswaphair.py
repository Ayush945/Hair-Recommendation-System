from keras.models import load_model
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from skimage import io 
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.losses import mean_squared_error

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import spsolve

#load the model
model = load_model(r'E:\Class\Course Material\L6\Sem 2\Models\model.h5', custom_objects={'mse': mean_squared_error})

def load_image(path,as_gray = False):
    return io.imread(path,plugin='matplotlib',as_gray=as_gray)

def get_gray_scale(image):
    return rgb2gray(image)

def set_scale(image,shape):
    return resize(image,shape)

def get_image_for_network(path):
    try:
        img = load_image(path,True)
        img = set_scale(img,(224,224))
        img = img.reshape((224,224,1))
        img = img
    except:
        img=path
        img=get_gray_scale(img)
        img = set_scale(img,(224,224))
        img = img.reshape((224,224,1))
        img = img
    return img

def get_prediction(path,original_shape):
    img = get_image_for_network(path)
    res = model.predict(np.asarray([img]))[0]
    return resize(res.reshape((224,224)),original_shape) 

def resize_and_swap_hair(source_image, target_image):
  
  # Get the larger image size
  max_height, max_width = max(source_image.shape[:2]), max(target_image.shape[:2])

  # Resize both images to the same size
  source_image = cv2.resize(source_image, (max_width, max_height))
  target_image = cv2.resize(target_image, (max_width, max_height))

  return source_image,target_image

def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    return mat_A

def poisson_blending(source, target, mask, with_gamma=True):
    if with_gamma:
        gamma_value = 2.2
    else:
        gamma_value = 1
    source = source.astype('float')
    target = target.astype('float')
    source = np.power(source, 1 / gamma_value)
    target = np.power(target, 1 / gamma_value)

    res = target.copy()
    y_range, x_range = source.shape[:2]
    mat_A = laplacian_matrix(y_range, x_range)
    laplacian = mat_A.tocsc()
    mask[mask != 0] = 1

    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()

    y_min, y_max = 0, y_range
    x_min, x_max = 0, x_range

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()
        alpha = 1
        mat_b = laplacian.dot(source_flat) * alpha
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        res[:, :, channel] = x

    res = np.power(res, gamma_value)

    res[res > 255] = 255
    res[res < 0] = 0
    res = res.astype('uint8')
    return res

def refine_hair_mask(mask):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

  eroded_mask = cv2.erode(mask, kernel)

  refined_mask = cv2.dilate(eroded_mask, kernel)

  return refined_mask

#Function to swap the hair from target to source
def swap_hair(source_filename, target_filename):
    
    source_img = source_filename
    target_img = load_image(target_filename)
    source_img,target_img=resize_and_swap_hair(source_img,target_img)

    # Get hair segmentation masks
    source_mask = get_prediction(source_filename, source_img.shape[:2])
    target_mask = get_prediction(target_filename, target_img.shape[:2])

    # Ensure masks are single-channel
    if len(source_mask.shape) == 3:
        source_mask = rgb2gray(source_mask)
    if len(target_mask.shape) == 3:
        target_mask = rgb2gray(target_mask)
    
    # Apply erosion to refine the masks
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    source_mask = cv2.erode(source_mask, element)
    target_mask = cv2.erode(target_mask, element)

    # Resize masks to match image shapes
    resized_source_mask = resize(source_mask, source_img.shape[:2], anti_aliasing=True)
    resized_target_mask = resize(target_mask, target_img.shape[:2], anti_aliasing=True)

    if len(resized_source_mask.shape)==3:
        resized_source_mask=rgb2gray(resized_source_mask)
    if len(resized_target_mask.shape)==3:
        resized_target_mask=rgb2gray(resized_target_mask)
    
    print(target_img.shape)
    print(source_img.shape)
    print(resized_source_mask.shape)
    
    resized_source_mask = refine_hair_mask(resized_source_mask) 
    hair_region = target_img.copy()
    if len(hair_region.shape) == 3:
        for c in range(3):
            hair_region[:, :, c] = hair_region[:, :, c] * resized_target_mask
    else:
        hair_region = hair_region * resized_target_mask
    print(hair_region.shape)
    

    poisson_blended_source = poisson_blending(hair_region, source_img,resized_source_mask,with_gamma=True)

    poisson_blended_source=cv2.medianBlur(poisson_blended_source,3)
    poisson_blended_source=cv2.resize(poisson_blended_source,(224,224))
    return poisson_blended_source
    