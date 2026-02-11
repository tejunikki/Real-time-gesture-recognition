import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier

# ---------------- TEXT TO SPEECH ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- MEDIAPIPE HAND SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- DATASET (DEMO SIGNS) ----------------
# Finger order: [Thumb, Index, Middle, Ring, Pinky]
# 1 = finger open, 0 = finger closed

X = [
    [0,0,0,0,0],   # FIST
    [1,1,1,1,1],   # OPEN PALM
    [0,1,1,0,0],   # TWO FINGERS
    [1,0,0,0,0],   # THUMB UP
]

y = [0, 1, 2, 3]

labels = {
    0: "STOP",
    1: "HELLO",
    2: "YES",
    3: "GOOD"
}

# ---------------- TRAIN MODEL ----------------
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

# ---------------- FINGER STATE FUNCTION ----------------
def get_finger_states(landmarks):
    fingers = []

    # Thumb
    fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)

    # Other fingers
    fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)   # Index
    fingers.append(1 if landmarks[12].y < landmarks[10].y else 0) # Middle
    fingers.append(1 if landmarks[16].y < landmarks[14].y else 0) # Ring
    fingers.append(1 if landmarks[20].y < landmarks[18].y else 0) # Pinky

    return fingers

# ---------------- CAMERA START ----------------
cap = cv2.VideoCapture(0)
last_output = ""

print("Gesture Talk Running... Press Q to Exit")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            finger_states = get_finger_states(hand_landmarks.landmark)
            prediction = model.predict([finger_states])[0]
            output_text = labels[prediction]

            cv2.putText(frame, output_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 255, 0), 3)

            if output_text != last_output:
                speak(output_text)
                last_output = output_text

    cv2.imshow("Gesture Talk - AI Sign Language", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
