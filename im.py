import cv2
import pytesseract
import time

# Load the pre-trained license plate detection model
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Set the path to your Tesseract executable

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam. You can specify a different camera if needed.

# Configure the time interval (in seconds) for capturing pictures
capture_interval = 2 # Captures a picture every 30 seconds
last_capture_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to access the webcam.")
        break

    # Detect license plates in the frame
    plates = plate_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in plates:
        # Extract the license plate region
        plate_roi = frame[y:y + h, x:x + w]

        # Enhance the image using OpenCV techniques
        plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plate_roi = cv2.resize(plate_roi, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        plate_roi = cv2.GaussianBlur(plate_roi, (5, 5), 0)
        _, plate_roi = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use Tesseract OCR to recognize characters
        plate_text = pytesseract.image_to_string(plate_roi, config='--psm 6')  # 'psm 6' assumes a block of text

        # Draw a rectangle around the detected license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the recognized text
        cv2.putText(frame, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with license plate detection and character recognition
    cv2.imshow('License Plate Detection', frame)

    # Check if it's time to capture a picture
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        # Capture a picture and recognize characters
        captured_text = pytesseract.image_to_string(frame, config='--psm 6')
        print("Captured Text:", captured_text)

        # Update the last capture time
        last_capture_time = current_time

    # Exit the program if the 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()