import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    images = os.listdir(folder)
    '''
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    '''
    return images

def main():
    images = load_images_from_folder("C:/Users/Henry Dan Victor And/Desktop/train")
    for i in images:
        print(i)
        img = cv2.imdecode(np.fromfile('C:/Users/Henry Dan Victor And/Desktop/train/%s' %(i), dtype=np.uint8), -1)
        blur = cv2.blur(img, (10, 10))
        cv2.imencode(i, blur)[1].tofile('C:/Users/Henry Dan Victor And/Desktop/Gaussian_train/%s' %(i))


'''
img = cv2.imdecode(np.fromfile('京A88731.jpg', dtype=np.uint8), -1)

blur = cv2.blur(img,(10,10))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
'''
#cv2.imwrite('C:/Users/Henry Dan Victor And/Desktop/Test.jpg', blur)
#cv2.imencode('京A88731.jpg', blur)[1].tofile('C:/Users/Henry Dan Victor And/Desktop/京A88731.jpg')
if __name__ == "__main__":
    main()