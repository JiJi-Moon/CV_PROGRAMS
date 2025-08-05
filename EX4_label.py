import cv2
import numpy as np

def detect_and_label_objects(image_path, color_ranges, labels):
    """
    Detects objects of specified colors in an image, labels them,
    and displays each object in binary label format.

    Args:
        image_path: Path to the image file.
        color_ranges: A flat list of HSV lower and upper bounds: [lower1, upper1, lower2, upper2, ...]
                     where each bound is a tuple of (H, S, V).
        labels: A list of labels corresponding to the color ranges.
    Returns:
        None. Displays the original image with labeled objects and binary format.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image at {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(0, len(color_ranges), 2):
        lower = np.array(color_ranges[i])
        upper = np.array(color_ranges[i + 1])
        label = labels[i // 2]

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue  # Skip small objects/noise

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            object_roi = image[y:y+h, x:x+w]
            gray_object = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
            _, binary_object = cv2.threshold(gray_object, 127, 255, cv2.THRESH_BINARY)

            binary_label = np.zeros_like(image)
            cv2.rectangle(binary_label, (x, y), (x + w, y + h), (255, 255, 255), -1)

            cv2.imshow(f"Binary label for {label}", binary_label)

    cv2.imshow("Labeled Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ===== Main script (outside the function) =====

color_ranges = [
    (0, 100, 100), (10, 255, 255),     # Red
    (25, 100, 100), (35, 255, 255),    # Yellow
    (100, 100, 100), (120, 255, 255)   # Blue
]
labels = ["Red", "Yellow", "Blue"]

# Use raw string (r"...") or escape backslashes properly
detect_and_label_objects(r"input2.jfif", color_ranges, labels)
