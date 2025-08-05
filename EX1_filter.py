import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- 1. SETUP: Load Image ---
try:
    # IMPORTANT: Replace 'dog.jpg' with the path to your image file
    input_image_path = 'input.jfif'
    img_bgr = cv2.imread(input_image_path)
    # Convert from BGR (OpenCV default) to RGB (Matplotlib default)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Convert to grayscale for filters that require it
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
except:
    print("Error: Image not found or could not be read.")
    print("Please make sure 'dog.jpg' is in the same directory or provide the full path.")
    # Create a dummy placeholder image to prevent crashing
    img_rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    img_gray = np.zeros((400, 600), dtype=np.uint8)
    cv2.putText(img_rgb, 'Image Not Found', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)


# --- 2. APPLY FILTERS ---


# Kernel size for blurring filters (must be an odd number)
kernel_size = 7


# Filter 1: Averaging (Box) Blur
# Smooths the image by taking the average of all pixels under the kernel area.
avg_blur = cv2.blur(img_rgb, (kernel_size, kernel_size))


# Filter 2: Gaussian Blur
# Similar to averaging, but uses a weighted average (center pixels get more weight).
# It produces a smoother, more natural blur.
gaussian_blur = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)


# Filter 3: Median Blur
# Replaces each pixel's value with the median of its neighbors.
# Excellent for removing "salt-and-pepper" noise.
median_blur = cv2.medianBlur(img_rgb, kernel_size)


# Filter 4: Bilateral Filter
# A powerful filter that smooths the image while preserving sharp edges.
bilateral_filter = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)


# Filter 5: Sobel Edge Detection
# Detects edges by computing the image gradient. We combine X and Y gradients.
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Filter 6: Canny Edge Detection
# A more advanced, multi-stage algorithm for detecting a wide range of edges.
canny_edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)


# Kernel for morphological operations
morph_kernel = np.ones((5, 5), np.uint8)


# Filter 7: Dilation
# Expands bright regions and the boundaries of foreground objects.
dilated_img = cv2.dilate(img_gray, morph_kernel, iterations=1)


# Filter 8: Erosion
# Shrinks bright regions and erodes the boundaries of the foreground object.
eroded_img = cv2.erode(img_gray, morph_kernel, iterations=1)


# --- 3. DISPLAY RESULTS ---


# Create a list of titles and images to display
titles = [
    'Original Image', '1. Averaging Blur', '2. Gaussian Blur', '3. Median Blur',
    '4. Bilateral Filter', '5. Sobel Edges', '6. Canny Edges', '7. Dilation', '8. Erosion'
]


images = [
    img_rgb, avg_blur, gaussian_blur, median_blur,
    bilateral_filter, sobel_edges, canny_edges, dilated_img, eroded_img
]


# Use Matplotlib to create a 3x3 grid of plots
plt.figure(figsize=(15, 10))


for i in range(9):
    plt.subplot(3, 3, i + 1)
    # Check if the image is grayscale to apply the correct color map
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')


# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("EX1_output.jpg")
plt.show()
