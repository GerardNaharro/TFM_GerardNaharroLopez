# Color filtering debugging

import cv2
import numpy as np


# Open the image

img = cv2.imread("imagenes/portero_rojo.PNG") # green
#img = cv2.imread("imagenes/referencia2.PNG") # red

# In-Range Filtering

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#temp_mask = cv2.inRange(hsv_img, (0, 0, 156), (179, 255, 255))

red_mask = cv2.inRange(hsv, (0,77,75), (22,255,255))

gk_mask = cv2.inRange(hsv, (29,99,172) ,(36,193,254))

green_mask = cv2.inRange(hsv, (38,56,155), (91,204,255))


out = cv2.bitwise_and(img, img, mask=green_mask)
print(np.sum(out != 0))
#inv_out = cv2.bitwise_and(sample_img, sample_img, inv_mask)

cv2.imshow("Image", img)
cv2.imshow("Out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()