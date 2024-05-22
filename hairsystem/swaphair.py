from keras.models import load_model
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from skimage import io,transform
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.losses import mean_squared_error
import scipy.sparse
from scipy.sparse.linalg import spsolve

#pretrained U-Net model for hair segmentation
model = load_model(r'E:\Class\Course Material\L6\Sem 2\Models\model.h5', custom_objects={'mse': mean_squared_error})

#function for Laplacian matrix
def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

    mat_A.setdiag(-1, 1 * m)
    mat_A.setdiag(-1, -1 * m)

    return mat_A

#function for poisson blending 
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


#function to swap hair
def swap_hair(source_filename, target_filename):
    print('here')
    original_image=source_filename

    new_image=source_filename
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    new_image = transform.resize(new_image, (224, 224))
    new_image = np.expand_dims(new_image, axis=-1)
    new_image = np.expand_dims(new_image, axis=0)
    
    predicted_mask = model.predict(new_image)   
    _, predicted_mask_binary = cv2.threshold(predicted_mask.squeeze(), 0.5, 1, cv2.THRESH_BINARY)
    predicted_mask_resized = transform.resize(predicted_mask_binary, (original_image.shape[0], original_image.shape[1]), order=0, preserve_range=True)
    masked_image = original_image.copy()
    if len(masked_image.shape) == 3:
        for c in range(3):
            masked_image[:, :, c] = masked_image[:, :, c] * (1 - predicted_mask_resized)
    else:
        masked_image = masked_image * (1 - predicted_mask_resized)
    
    imagepath2 = target_filename
    second_image = io.imread(imagepath2)

    second_image_gray = io.imread(imagepath2, as_gray=True)
    second_image_resized = transform.resize(second_image_gray, (224, 224))
    second_image_resized = np.expand_dims(second_image_resized, axis=-1)
    second_image_resized = np.expand_dims(second_image_resized, axis=0)

    predicted_mask2 = model.predict(second_image_resized)
    _, predicted_mask2_binary = cv2.threshold(predicted_mask2.squeeze(), 0.5, 1, cv2.THRESH_BINARY)

    predicted_mask2_binary = cv2.GaussianBlur(predicted_mask2_binary, (5, 5), 0)
    print("here 4")
    # Resize the predicted mask of the second image back to original second image size
    predicted_mask2_resized = transform.resize(predicted_mask2_binary, (second_image.shape[0], second_image.shape[1]), order=0, preserve_range=True)

    # Extract the hair region from the second image
    hair_region = second_image.copy()
    if len(hair_region.shape) == 3:
        for c in range(3):
            hair_region[:, :, c] = hair_region[:, :, c] * predicted_mask2_resized
    else:
        hair_region = hair_region * predicted_mask2_resized

    hair_region_resized = transform.resize(hair_region, (original_image.shape[0], original_image.shape[1]), order=0, preserve_range=True, anti_aliasing=True)
    mask = predicted_mask2_resized.copy()
    mask[mask > 0] = 1
    print("here 5")
    # Resize the hair region to match the first image size
    hair_region_resized = transform.resize(hair_region, (original_image.shape[0], original_image.shape[1]), order=0, preserve_range=True, anti_aliasing=True)
    center = (original_image.shape[1] // 2, original_image.shape[0] // 2)
    mask=transform.resize(mask, (original_image.shape[0], original_image.shape[1]), order=0, preserve_range=True, anti_aliasing=True)
    
    blended_image = poisson_blending(hair_region_resized, masked_image, mask, with_gamma=True)
    blended_image=cv2.cvtColor(blended_image,cv2.COLOR_BGR2GRAY)
    print("Completed this process")
    return blended_image