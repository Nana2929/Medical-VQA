import cv2


def _remove_words(img, kernal, canny_range):
    # CANNY
    canny_min, canny_max = canny_range
    canny_img = cv2.Canny(img, canny_min, canny_max)

    # dilation
    dilate_img = cv2.dilate(canny_img, kernal, iterations=1)

    # erosion
    erode_img = cv2.erode(dilate_img, kernal, iterations=1)

    # repair
    repair_img = cv2.inpaint(img, erode_img, 3, cv2.INPAINT_TELEA)

    return repair_img


def remove_text(img, morphology_kernel, guass, canny_range):
    word_removed = _remove_words(img, morphology_kernel, canny_range)
    blur = cv2.GaussianBlur(word_removed, (guass, guass), 0)

    return blur
