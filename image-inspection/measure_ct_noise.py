import cv2
import numpy as np
from pathlib import Path
import os
# code credit: GPT-4 (2023.05.14)
image_dir = Path('QCR_PubMedCLIP/data/data_rad/images_classified')

# load ct images
ct_dir = image_dir/'mod'/ 'CT'
ct_images = list(ct_dir.glob('*.jpg'))
print('number of ct images:', len(ct_images))

def calc_noise_std(image_path: os.PathLike):
    image_path = str(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    w, h = image.shape
    # ROI: 4/5 width * 4/5 height
    roi = image[
        int(1/10 * w): int(9/10 * w),
        int(1/10 * h): int(9/10 * h)]
    # Compute the standard deviation of the ROI
    noise_std = np.std(roi)
    return noise_std

# save noise std in an array and then calc the mean, std, etc.
# finally identify those images in 4th quartile
noise_std_list = []
for ct_img in ct_images:
    noise_std = calc_noise_std(ct_img)
    # print('noise std:', noise_std)
    noise_std_list.append(noise_std)

# convert to numpy array
# save noise std in an array and then calc the mean, std, etc.
# finally identify those images in 4th quartile
noise_std_list = np.array(noise_std_list)
print('noise std\'s mean:', np.mean(noise_std_list))
print('noise std\'s td:', np.std(noise_std_list))
# 4th quartile threshold
threshold = np.quantile(noise_std_list, 0.75)
# identify index
index = np.where(noise_std_list > threshold)
# use index to trace back image names in ct_images
noisy_Q1_image_names = [ct_images[i].name for i in index[0]]
noisy_Q4_image_names = [ct_images[i].name for i in index[0]]
print('Q1 noisy image names (less noisy):', noisy_Q1_image_names)
print('Q4 noisy image names (noise-prone):', noisy_Q4_image_names)









