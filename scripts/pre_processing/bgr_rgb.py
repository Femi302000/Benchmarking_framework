import cv2
import numpy as np

# Load image
img = cv2.imread("/home/femi/Downloads/download.jpeg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Any nonzero pixel → white, zero stays black
bw = np.where(gray > 0, 255, 0).astype(np.uint8)

cv2.imshow("Black & White", bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
