import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load and encode known faces
sai_image = face_recognition.load_image_file("faces/sai.jpg")
sai_encoding = face_recognition.face_encodings(sai_image)[0]
teja1_image = face_recognition.load_image_file("faces/teja1.jpg")
teja1_encoding = face_recognition.face_encodings(teja1_image)[0]

# Known face encodings and names
known_face_encodings = [sai_encoding, teja1_encoding]
known_face_names = ["sai", "teja1"]

# List to track students for attendance
students = known_face_names.copy()

face_locations = []

# Prepare CSV file for attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
with open(f"{current_date}.csv", 'w+', newline="") as f:
    lnwriter = csv.writer(f)

    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Process each face found
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            name = None  # Initialize name with a default value

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Only proceed if a name was assigned
            if name and name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        # Show the video feed
        cv2.imshow("Attendance", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()