import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

prev_cx, prev_cy = None, None
prev_time = None
movement_text = ""
speed_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    direction, speed = "", 0
    cx, cy = None, None
    current_time = time.time()
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Track the first detected face
        cx, cy = x + w // 2, y + h // 2
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Movement & Speed calculations
        if prev_cx is not None and prev_cy is not None and prev_time is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy
            dt = current_time - prev_time
            if dt > 0:
                speed = int(np.sqrt(dx**2 + dy**2) / dt)
                # Determine direction
                if abs(dx) > abs(dy):
                    if dx > 10:
                        direction = "Right"
                    elif dx < -10:
                        direction = "Left"
                else:
                    if dy > 10:
                        direction = "Down"
                    elif dy < -10:
                        direction = "Up"
                if direction == "":
                    direction = "Stationary"
                movement_text = f"Dir: {direction}"
                speed_text = f"Speed: {speed} px/s"
        prev_cx, prev_cy = cx, cy
        prev_time = current_time
    else:
        movement_text = "No face detected"
        speed_text = ""
        prev_cx, prev_cy = None, None
        prev_time = None

    # Display overlays
    cv2.putText(frame, movement_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, speed_text, (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Direction Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
