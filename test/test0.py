import os
base_path = os.path.dirname(os.path.abspath(__file__))
bin_path = r"C:\Program Files (x86)\Tesseract-OCR"
os.environ["PATH"] = bin_path + os.pathsep + os.environ['PATH']

import cv2
import imutils  
import pytesseract as ts
import re 
import numpy as np

ts.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

image = cv2.imread('test.jpg')
#image = imutils.resize(image, width = 500)

cv2.imshow("Original Image",image)
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray)
cv2.waitKey(0)

#gray = cv2.GaussianBlur(gray, (9, 9), 0)

#reduce noice from our image and make it smooth
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Smoother Image", gray)
cv2.waitKey(0)

#find the egdes of images
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny edge", edged)
cv2.waitKey(0)



#find the contours based on the image
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0,255,0),3)
cv2.imshow("Canny after contouring", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
NumberPlateCount = None

image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0,255,0),3)
cv2.imshow('TOP 30 Contours', image2)
cv2.waitKey(0)

count = 0
name = 1
for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
    if (len(approx) == 4):
        NumberPlateCount = approx
        x,y,w,h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        cv2.imwrite(str(name)+'.png',crp_img)
        name += 1
        break
    
    cv2.drawContours(image, [NumberPlateCount], -1, (0,255,0),3)
    cv2.imshow("Final Image", image)
    cv2.waitKey(0)