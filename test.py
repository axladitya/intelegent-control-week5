from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load YOLOv8 Pose model
model = YOLO("yolov8n-pose.pt")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Function to detect hand gesture
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    threshold = 0.05
    thumb_raised = thumb_tip.y < thumb_ip.y - threshold
    index_raised = index_tip.y < index_pip.y - threshold
    middle_raised = middle_tip.y < middle_pip.y - threshold
    ring_raised = ring_tip.y < ring_pip.y - threshold
    pinky_raised = pinky_tip.y < pinky_pip.y - threshold

    raised_fingers = [thumb_raised, index_raised, middle_raised, ring_raised, pinky_raised]
    gesture_text = "".join(["1" if raised else "0" for raised in raised_fingers])

    if gesture_text == "11001":
        return "Metal"
    elif gesture_text == "11111":
        return "Jari Terbuka"
    elif gesture_text == "00000":
        return "Jari Tertutup"
    elif gesture_text == "01100":
        return "Peace"
    elif gesture_text == "10000":
        return "Jempol"
    elif gesture_text == "01000":
        return "Telunjuk"
    elif gesture_text == "00100":
        return "Tengah"
    elif gesture_text == "00010":
        return "Manis"
    elif gesture_text == "00001":
        return "Kelingking"
    else:
        return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        keypoints = result.keypoints
        hand_points = [kp for kp in keypoints if kp[0] in [9, 10]]
        for point in hand_points:
            x, y = int(point[1]), int(point[2])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(image_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, f"{gesture}", (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Fingers: {gesture}", (x_min, y_min - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Pose and Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
