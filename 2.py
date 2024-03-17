import cv2
import numpy as np

def show_image(img, im_name='-'):
    cv2.imshow(im_name, np.uint8(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image1 = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)

cpy_img = image1.copy()
height, width = cpy_img.shape[:2]
cpy_img = cpy_img[:, width * 2 // 3 : width]

height, width = cpy_img.shape[:2]
image2 = cv2.getRotationMatrix2D((width, height), 30, 1.0)
image2 = cv2.warpAffine(cpy_img, image2, (width, height))

h2, w2 = image2.shape[:2]
h2, w2 = int(h2 * 0.8), int(w2 * 1.5)
image2 = cv2.resize(image2, (w2, h2))
show_image(image2)

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

matched_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches[:10],
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

show_image(matched_image, "Matched image")