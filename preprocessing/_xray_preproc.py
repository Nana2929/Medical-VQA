
import pathlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def norm(img):
    # if not gray, gray-scale it
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def equal_hist(img):
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    img = cv2.equalizeHist(img)
    return img

def gauss_blur(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def create_clahe(img, clip_limit: float=1, tile_grid_size: float=16):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size,
                                                        tile_grid_size))
    img = clahe.apply(img)
    return img