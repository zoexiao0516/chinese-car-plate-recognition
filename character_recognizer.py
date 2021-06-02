#coding = utf -8
# test file if you want to quickly try tesseract on a chinese character
import os
base_path = os.path.dirname(os.path.abspath(__file__))
bin_path = r"C:\Program Files (x86)\Tesseract-OCR"
os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
import pytesseract
import cv2
import numpy as np 
import imutils
import pandas as pd

def add_padding(img):
    # img = cv2.imread(img)
    # print(img.shape)
    ht, wd = img.shape
    print(img.shape)

    # create new image of desired size and color (blue) for padding
    ww = 1100
    hh = 1100
    color = (255)
    result = np.full((hh, ww), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy + ht, xx:xx + wd] = img
    return result

def process_image(img):
    image = cv2.imread("train/images/" + img)
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(0)
    image = imutils.resize(image, width = 500)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    #reduce noice from our image and make it smooth
    gray = cv2.bilateralFilter(gray, 40, 27, 27)
    #图片binary
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #padding the image
    image = add_padding(binary)
    image = imutils.resize(image, width = 64)
    # cv2.imshow("Final Image", image)
    # cv2.waitKey(0)
    cv2.imwrite('Lenet Training set/'+img, image)
    # reverse the color of the image
    # gray = cv2.bitwise_not(gray)
    # cv2.imshow("Reversed Color Image", gray)
    # cv2.waitKey(0)


def main():
    #find file directory and put into a list:
    # for file in os.listdir("train/images"):
    #     process_image(file)


    id =[]
    for file in os.listdir("Lenet Training set"):
        id.append(file[:-4])

    dic = {}
    dic["id"] = []
    dic["code"] = []
    for i in id:
        with open("labels/" + i + ".txt", "r") as file1:
            truth_code = file1.readline().split()[0]
        dic["id"].append(i)
        dic["code"].append(truth_code)
    df = pd.DataFrame(dic)
    df.to_csv("Province.csv")

if __name__ == "__main__":
    main()
