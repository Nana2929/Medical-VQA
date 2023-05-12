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

save_img = True
save_mask = True
remove_word = True




def new_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def removeWords(img,save_mask=True,output_mask_path=None,save_img=True,output_path=None):

    # read image
    img = cv2.imread(INPUT + image)

    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #normalize
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # CANNY
    canny_img = cv2.Canny(img,230,250)

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
            if remove_word:
                removeWords(image,save_mask,OUTPUT_REMOVE_MASK,save_img,OUTPUT_REMVE_WORD)

            # read image
            img = cv2.imread(OUTPUT_REMVE_WORD + image)

            # guass
            blur = cv2.GaussianBlur(img, (guass, guass), 0)

            # save
            if save_img:
                new_folder("Xray_preprocessed/")
                cv2.imwrite("Xray_preprocessed/" + image, blur)

            







#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     his = cv2.equalizeHist(gray)
#     cv2.imwrite("./Xray_Histogram/"+image,his)

#     blur = cv2.GaussianBlur(gray,kernal_size,0)
#     cv2.imwrite("./Xray_Gauss/"+image,blur)

#     sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
#     cv2.imwrite("./Xray_Sobel/"+image,sobel)

#     blur_his = cv2.GaussianBlur(his,kernal_size,0)
#     cv2.imwrite("./Xray_Gauss_Histogram/"+image,blur_his)

#     sobel_his_blur = cv2.GaussianBlur(blur_his,kernal_size,0)
#     cv2.imwrite("./Xray_Gauss_Histogram_Sobel/"+image,sobel_his_blur)










    




