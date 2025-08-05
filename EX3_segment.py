import cv2

# Load image in grayscale
gray = cv2.imread('input.jfif', cv2.IMREAD_GRAYSCALE)

# Apply Adaptive Thresholding
thresh_adapt = cv2.adaptiveThreshold(
    gray,                      # Source image (grayscale)
    255,                       # Maximum value to use with the THRESH_BINARY thresholding
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method (Gaussian or Mean)
    cv2.THRESH_BINARY,         # Thresholding type
    11,                        # Block size (must be odd)
    2                          # Constant subtracted from the mean
)

# Save or display result
cv2.imshow("Adaptive Thresholding", thresh_adapt)
cv2.waitKey(0)
cv2.destroyAllWindows()
