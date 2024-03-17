import cv2
import numpy as np

def show_image(img, im_name='-'):
    cv2.imshow(im_name, np.uint8(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (9, 9), 0)
# show_image(img)

canny_img = cv2.Canny(img, img.max() * 0.3, img.max() * 0.6)
show_image(canny_img)