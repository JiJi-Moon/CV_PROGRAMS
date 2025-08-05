# import OpenCV
import cv2
# import Numpy
import numpy as np

# Read an image in grayscale mode
img = cv2.imread("input.jfif", 0)

# Check if the image was successfully loaded

# Create Histogram Equalization
equ = cv2.equalizeHist(img)

# Stack original and equalized images side by side
res = np.hstack((img, equ))

# Save the result
cv2.imwrite('EX2_output.jpg', res)

# Show the result
cv2.imshow("Input Image vs Equalized Image", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
