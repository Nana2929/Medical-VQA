import cv2
import pathlib 
import numpy as np
import os

PATH = "./preprocess/"
INPUT = "./Xray/"
OUTPUT_REMVE_WORD = PATH + "Xray_remove/"
OUTPUT_REMOVE_MASK = PATH + "Xray_remove_mask/"

kernal = np.ones((5,5),np.uint8)
guass = 5
canny_min = 230
canny_max = 250

save_mask  = True
save_img = True


def new_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def removeWords(image,imput_path,save_mask=True,output_mask_path=None,save_img=True,output_path=None):
    # read image
    img = cv2.imread(imput_path + image)

    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #normalize
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # CANNY
    canny_img = cv2.Canny(img,canny_min,canny_max)

    # dilation
    dilate_img = cv2.dilate(canny_img, kernal, iterations=1)

    # erosion
    erode_img = cv2.erode(dilate_img, kernal, iterations=1)
    if save_mask :
        new_folder(output_mask_path)
        cv2.imwrite(output_mask_path + image, erode_img)
    # repair
    repair_img = cv2.inpaint(img, erode_img, 3, cv2.INPAINT_TELEA)
    if save_img :
        new_folder(output_path)
        cv2.imwrite(output_path + image, repair_img)

if __name__ == '__main__':
    # read image
    filelist=os.listdir(INPUT)

    for image in filelist:
        # .png or .jpg
        if image.endswith('.png') or image.endswith('.jpg'):
            removeWords(image,INPUT,save_mask,OUTPUT_REMOVE_MASK,save_img,OUTPUT_REMVE_WORD)
