import os

base_path = os.path.dirname(os.path.abspath(__file__))
bin_path = r"C:\Program Files (x86)\Tesseract-OCR"
os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
import cv2
import imutils

import pytesseract
import re

# pytesseract.pytesseract.tesseract_cmd = r"/Users/shenmengjie/Downloads/tesseract-ocr-w32-setup-v5.0.0-alpha.20201127.exe "


# for imageName in glob.glob('/content/yolov5/runs/detect/exp2/crops/Carplate/*.jpg'): #assuming JPG
#     image = cv.imread(imageName)
#     image = imutils.resize(image, width = 500)

#     cv2_imshow("Original Image", image)
#     #cv2.waitKey(0)

# image = cv2.imread('-A9YU86_jpg.rf.29d88d38edb692fa1174f8703883a2b9.jpg')
# #image = imutils.resize(image, width = 500)

# cv2.imshow("Original Image",image)
# cv2.waitKey(0)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Scale Image", gray)
# cv2.waitKey(0)

# #reduce noice from our image and make it smooth
# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# cv2.imshow("Smoother Image", gray)
# cv2.waitKey(0)

# #find the egdes of images
# edged = cv2.Canny(gray, 170, 200)
# cv2.imshow("Canny edge", edged)
# cv2.waitKey(0)

# #find the contours based on the image
# cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# #we will create a copy of our original image to draw all the contours
# image1 = image.copy()
# cv2.drawContours(image1, cnts, -1, (0,255,0),3)
# cv2.imshow('Canny after contouring', image1)
# cv2.waitKey(0)

# text = pytesseract.image_to_string(image1, lang='eng')
# print("Number is :", text)
# cv2.waitKey(0)

def recognize_plate(img):
    # grayscale region within bounding box
    box = img
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    # resize image to three times as large as original for better readability
    gray = cv2.resize(gray, None, fx=7, fy=7, interpolation=cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)

    # threshold the image using Otsus method to preprocess for tesseract
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15,0)
    cv2.imshow("Otsu Threshold", thresh)
    cv2.waitKey(0)


    '''
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    cv2.imshow("Dilation", dilation)
    cv2.waitKey(0)
    '''


    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = gray.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.5: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 15: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # grab character region of image
        roi = thresh[y - 25:y + h + 25, x - 25:x + w + 25]
        # perfrom bitwise not to flip image to black text on white background
        #roi = cv2.bitwise_not(roi)
        # perform another blur on character region
        #roi = cv2.medianBlur(roi, 5)
        #cv2.imshow("roi", roi)
        #cv2.waitKey(0)
        #cv2.imwrite("./roi.jpg", roi)

        try:
            custom_config = '--psm 8 --oem 1'
            text = pytesseract.image_to_string(roi, config=custom_config)#.strip()[:1]
            print(text)
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except:
            text = None

    if plate_num != None:
        print("License Plate #: ", plate_num)
    cv2.imshow("Character's Segmented", im2)
    cv2.waitKey(0)
    return plate_num


# image = cv2.imread('-A9YU86_jpg.rf.29d88d38edb692fa1174f8703883a2b9.jpg')
image = cv2.imread('test.jpg')
recognize_plate(image)
