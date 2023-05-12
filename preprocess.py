import cv2
import pathlib 
import numpy as np
import os

PATH = "./preprocess/"
INPUT = "./Xray/"
OUTPUT_REMVE_WORD = PATH + "Xray_remove/"
OUTPUT_REMOVE_MASK = PATH + "Xray_remove_mask/"

kernal = np.ones((3,3),np.uint8)
guass = 5

save_img = True
save_mask = True
remove_word = True




def new_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def SSR(img, size):
    res = img

    L_blur = cv2.GaussianBlur(res, (size, size), 0)
    log_R = np.log(res + 0.001) - np.log(L_blur + 0.001)

    minvalue, maxvalue, minloc, maxloc = cv2.minMaxLoc(log_R)
    log_R = (log_R - minvalue) * 255.0 / (maxvalue - minvalue)

    dst = cv2.convertScaleAbs(log_R)
    dst = cv2.add(img, dst)
    return dst

def MSR(img, scales):
    number = len(scales)
    res = img

    h, w = img.shape[:2]
    dst_R = np.zeros((h, w), dtype=np.float32)
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(number):
        L_blur = cv2.GaussianBlur(res, (scales[i], scales[i]), 0)
        log_R += np.log(res + 0.001) - np.log(L_blur + 0.001)

    log_R = log_R / number
    cv2.normalize(log_R, dst_R, 0, 255, cv2.NORM_MINMAX)
    dst_R = cv2.convertScaleAbs(dst_R)
    dst = cv2.add(img, dst_R)

    return dst

def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def removeWords(img,save_mask=True,output_mask_path=None,save_img=True,output_path=None):
        # read image
    img = cv2.imread(INPUT + image)

    # inhance image
    b_gray, g_gray, r_gray = cv2.split(img)
    b_gray = SSR(b_gray, guass)
    g_gray = SSR(g_gray, guass)
    r_gray = SSR(r_gray, guass)
    result = cv2.merge([b_gray, g_gray, r_gray])


    # CANNY
    canny_img = cv2.Canny(img, 120,300)

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
            
            img = cv2.imread(OUTPUT_REMVE_WORD+image)

            # normalize
            normal_



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










    




