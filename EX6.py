import cv2
import numpy as np
import pytesseract
import os

# Update this path to where Tesseract-OCR is installed on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Haar Cascade
cascade_path = "haarcascade_russian_plate_number.xml"
if not os.path.exists(cascade_path):
    print("Cascade XML file not found!")
    exit()

cascade = cv2.CascadeClassifier(cascade_path)

# Indian state code dictionary
states = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana",
    "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
    "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
    "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
    "PY": "Puducherry", "PN": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
    "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal", "CG": "Chhattisgarh",
    "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"
}

def extract_num(img_filename):
    img = cv2.imread(img_filename)

    if img is None:
        print(" Image not found or path is incorrect.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(plates) == 0:
        print(" No license plate detected.")
        return

    for (x, y, w, h) in plates:
        h_img, w_img, _ = img.shape
        offset_x, offset_y = int(0.02 * w_img), int(0.02 * h_img)

        # Crop and preprocess
        plate_img = img[y+offset_y:y+h-offset_y, x+offset_x:x+w-offset_x]
        kernel = np.ones((1, 1), np.uint8)
        plate_img = cv2.dilate(plate_img, kernel, iterations=1)
        plate_img = cv2.erode(plate_img, kernel, iterations=1)

        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        # OCR
        text = pytesseract.image_to_string(plate_thresh, config='--psm 8')
        text = ''.join(char for char in text if char.isalnum())
        state_code = text[:2].upper()

        print(f" Detected Number: {text}")
        print(f" State: {states.get(state_code, 'Unknown')}")

        # Draw result
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Show images
        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(plate_img)
            cv2_imshow(img)
        except:
            cv2.imshow("License Plate", plate_img)
            cv2.imshow("Detected Result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save result image
        cv2.imwrite("Detected_Result.png", img)

        return text

#  Call the function with your image path
extract_num(r"D:\CV\car.jpg
## https://github.com/UB-Mannheim/tesseract/wiki
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml

