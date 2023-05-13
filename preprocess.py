import cv2
import pathlib 
import numpy as np
import os
from  removeword import *
import matplotlib.pyplot as plt



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

        # hist
        his = cv2.calcHist([img], [0], None, [256], [0, 256])
        # cv2.imshow("hist", his)
        # plt.plot(his)
        # plt.show()



        # save
        if save_img:
            clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16,16))
            imgs = clahe.apply(img)
            new_folder("./Xray_preprocessed_05/")
            cv2.imwrite("./Xray_preprocessed_05/" + image, imgs)

        if save_img:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
            imgs = clahe.apply(img)
            new_folder("./Xray_preprocessed_1_16/")
            cv2.imwrite("./Xray_preprocessed_1_16/" + image, imgs)


        img = cv2.equalizeHist(img)
        # guass
        blur = cv2.GaussianBlur(img, (guass, guass), 0)



        # save
        if save_img:
            new_folder(OUTPUT_PREPROCESSED)
            cv2.imwrite(OUTPUT_PREPROCESSED + image, blur)










    




