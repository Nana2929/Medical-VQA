import cv2
import pathlib 
import numpy as np
import os
from  removeword import *

PATH = "./preprocess/"
INPUT = "./Xray/"
OUTPUT_REMVE_WORD = PATH + "Xray_remove/"
OUTPUT_REMOVE_MASK = PATH + "Xray_remove_mask/"
OUTPUT_PREPROCESSED = "./Xray_preprocessed/"

remove_word = True
save_img = True

def new_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# read image
filelist=os.listdir(INPUT)

for image in filelist:
    # .png or .jpg
    if image.endswith('.png') or image.endswith('.jpg'):
        if remove_word:
            removeWords(image,INPUT,True,OUTPUT_REMOVE_MASK,True,OUTPUT_REMVE_WORD)

        # read image
        img = cv2.imread(OUTPUT_REMVE_WORD + image)

        # gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # normalize
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # histogram equalization
        img = cv2.equalizeHist(img)

        # guass
        blur = cv2.GaussianBlur(img, (guass, guass), 0)



        # save
        if save_img:
            new_folder(OUTPUT_PREPROCESSED)
            cv2.imwrite(OUTPUT_PREPROCESSED + image, blur)










    




