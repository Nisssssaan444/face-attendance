import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import csv
import time
from datetime import datetime
import win32com.client
from win32com.client import Dispatch

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

with open('names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Correct the path to the background image
imgBackground_path = "C:\\Users\\Dell\\OneDrive\\Desktop\\python\\backgrond.png"
imgBackground = cv2.imread(imgBackground_path)

# Check if the background image was loaded successfully
if imgBackground is None:
    print(f"Error: Could not load background image from path: {imgBackground_path}")
    exit()

COL_NAMES = ['NAME', 'TIME']

# Set to keep track of names that have already been recorded
recorded_names = set()

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame from video capture.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        name = str(output[0])
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display name above the bounding box
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if name not in recorded_names:
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
            cv2.putText(frame, name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
            attendance = [name, str(timestamp)]
            
            # Record the attendance
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
            
            recorded_names.add(name)
        else:
            # Indicate that attendance is already taken for this person
            cv2.putText(frame, f"{name} - Attendance Already Done", (x, y-60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)

    # Resize frame to fit in the background image
    resized_frame = cv2.resize(frame, (640, 402))
    imgBackground[162:162 + 402, 55:55 + 640] = resized_frame
    cv2.imshow("Frame", imgBackground)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
