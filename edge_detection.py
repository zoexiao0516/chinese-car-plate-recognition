import cv2
import imutils
import numpy as np

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def edge_detection(img):
    image = cv2.imread(img)
    image = imutils.resize(image, width=500)
    cv2.imshow("original image", image)
    cv2.waitKey(0)
    # convert to gray image and Gaussian Blur
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # reduce noice from our image and make it smooth
    gray = cv2.bilateralFilter(gray, 40, 27, 27)

    # reverse the color of the image
    gray = cv2.bitwise_not(gray)

    # find the egdes of images automatic dectect
    v = np.median(gray)
    sigma = 0.4
    # ---- apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # edge detect
    edged = cv2.Canny(gray, lower, upper)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilation = cv2.dilate(edged, rect_kern, iterations=1)
    cv2.imshow("edged image", edged)
    cv2.waitKey(0)
    # find the contours based on the image
    cnts, new = cv2.findContours(dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 1)
    # filter out the contours not needed
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    NumberPlateCount = None

    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("contoured image", image2)
    cv2.waitKey(0)

    # finding the best possible contour
    name = 1
    # print(cnts)
    for i in cnts:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
        if (len(approx) == 4):
            NumberPlateCount = approx
            x, y, w, h = cv2.boundingRect(i)
            break
    if NumberPlateCount is not None:
        # cv2.drawContours(image, [NumberPlateCount], -1, (0,255,0),3)
        # collect the four points and order them
        pts = NumberPlateCount.reshape(4, 2)
        # perform perspective transformation on the image
        final = four_point_transform(image, pts)
        cv2.imshow("final image", final)
        cv2.waitKey(0)
    else:
        print("NO Carplates found in the image!")


def main():
    edge_detection("test11.jpg")

if __name__ == '__main__':
    main()
