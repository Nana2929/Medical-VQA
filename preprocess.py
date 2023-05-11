import cv2
import os
kernal_size = (5,5)
if not os.path.exists("./Xray_Gauss"):
    os.mkdir("./Xray_Gauss")

if not os.path.exists("./Xray_Histogram"):
    os.mkdir("./Xray_Histogram")

if not os.path.exists("./Xray_Sobel"):
    os.mkdir("./Xray_Sobel")

if not os.path.exists("./Xray_Gauss_Histogram"):
    os.mkdir("./Xray_Gauss_Histogram")

if not os.path.exists("./Xray_Gauss_Histogram_Sobel"):
    os.mkdir("./Xray_Gauss_Histogram_Sobel")



for image in os.listdir("./Xray"):
    img = cv2.imread("./Xray/"+image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    his = cv2.equalizeHist(gray)
    cv2.imwrite("./Xray_Histogram/"+image,his)

    blur = cv2.GaussianBlur(gray,kernal_size,0)
    cv2.imwrite("./Xray_Gauss/"+image,blur)

    sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    cv2.imwrite("./Xray_Sobel/"+image,sobel)

    blur_his = cv2.GaussianBlur(his,kernal_size,0)
    cv2.imwrite("./Xray_Gauss_Histogram/"+image,blur_his)

    sobel_his_blur = cv2.GaussianBlur(blur_his,kernal_size,0)
    cv2.imwrite("./Xray_Gauss_Histogram_Sobel/"+image,sobel_his_blur)










    




