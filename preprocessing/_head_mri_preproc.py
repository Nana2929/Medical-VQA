import numpy as np
from scipy import ndimage
from skimage import morphology
from skfuzzy import cmeans


def _remove_noise(img):
    # img = img[:, :, 0]
    mask = img <= 20
    selem = morphology.disk(2)
    segmentation = morphology.dilation(mask, selem)
    labels, _ = ndimage.label(segmentation)

    mask = labels == 0

    mask = morphology.dilation(mask, selem)
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, selem)

    clean_img = mask * img

    return clean_img


def _fcm_class_mask(img, brain_mask):
    # We want 3 classes with a maximum of 50 iterations
    [class_centers, partitioned_matrix, _, _, _, _, _] = cmeans(img[brain_mask].reshape(-1, len(brain_mask[brain_mask])),
                                              3, 2, 0.005, 50)

    # We use 'key' to sort by the values and not the index
    matrix_list = [partitioned_matrix[class_num] for class_num, _ in sorted(enumerate(class_centers),
                                                                          key=lambda x: x[1])]
    mask = np.zeros(img.shape + (3,))

    for index in range(3):
        mask[..., index][brain_mask] = matrix_list[index]

    return mask


def _get_mask(img):
    mask = img == 0
    selem = morphology.disk(2)
    segmentation = morphology.dilation(mask, selem)
    labels, _ = ndimage.label(segmentation)

    mask = labels == 0
    mask = morphology.dilation(mask, selem)
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, selem)

    # brain segmentation
    matrix_mask = _fcm_class_mask(img, mask)
    # white matter
    white_matter_mask = matrix_mask[..., 2] > 0.8

    ''' If other masks are needed:
    # gray matter
    m1 = matrix_mask[..., 2] > 0.018
    m2 = matrix_mask[..., 2] > 0.41
    gray_matter_mask = m1 ^ m2

    # bone
    m1 = matrix_mask[..., 2] > 0
    m2 = matrix_mask[..., 2] > 0.018
    bone_mask = m1 ^ m2

    # not gray matter, white matter or bone
    m1 = matrix_mask[..., 2] > 0.41
    m2 = matrix_mask[..., 2] > 0.8
    other_mask = m1 ^ m2
    '''

    return white_matter_mask


def fcm_norm(img, norm_value):
    clean_img = _remove_noise(img)
    white_matter_mask = _get_mask(clean_img)
    wm_mean = clean_img[white_matter_mask == 1].mean()

    normalized_img = (clean_img / wm_mean) * norm_value
    # *= 255
    return normalized_img*255