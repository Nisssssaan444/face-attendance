import cv2
import numpy as np
import pickle
import os

# Initialize video capture
video = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the Haar Cascade Classifier
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces_data = []
name = input('Enter your name: ')

while True:
    # Capture frame-by-frame
    check, frame = video.read()
    
    if not check:
        print("Error: Could not read frame.")
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector.detectMultiScale(gray, 1.3, 5)
    i = 0
    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        reimg = cv2.resize(frame[y:y+h, x:x+w, :], (50, 50))
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(reimg)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255),2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Display the frame with rectangles
    cv2.imshow("Face Detection", frame)
    key=cv2.waitKey(1)
    # Break the loop if 'q' key is pressed or 60 faces have been collected
    if key == ord('q') or len(faces_data) == 100:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert faces_data to numpy array
faces_data = np.array(faces_data)
faces_data = faces_data.reshape(100, -1)

# Path to save data
data_path = 'C:/Users/Dell/OneDrive/Desktop/python'
names_file = os.path.join(data_path, 'names.pkl')
faces_file = os.path.join(data_path, 'faces_data.pkl')

# Save names data
if not os.path.exists(names_file):
    names = [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] *100)
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

# Save faces data
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)
