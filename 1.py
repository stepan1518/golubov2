import cv2
import numpy as np

def show_image(img, im_name='-'):
    cv2.imshow(im_name, np.uint8(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (7, 7), 0)
# show_image(img)

canny_img = cv2.Canny(img, img.mean() * 2 / 3, img.mean() * 4 / 3)
# show_image(canny_img)

max_detection = cv2.Canny(img, 0, 0)
# show_image(max_detection)

canny_img = cv2.Canny(img, img.mean() * 1.3, img.mean() * 2)
# show_image(canny_img)