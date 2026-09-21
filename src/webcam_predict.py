# run pip install opencv-python mediapipe joblib numpy in the terminal

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "models/sign_language_mlp.pkl"
HAND_LANDMARKER_PATH = "notebooks/hand_landmarker.task"



# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [str(i) for i in range(63)]

print("Model loaded successfully.")
print("Classes:", model.classes_)


# ============================================================
# 3. SET UP MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=HAND_LANDMARKER_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# 4. LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(detection_result):
    """
    Extract 21 hand landmarks and normalize them
    relative to the wrist.

    Returns:
        numpy array of shape (63,)
        or None if no hand is detected.
    """

    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]

    # Wrist = landmark 0
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    landmarks = []

    for lm in hand_landmarks:
        landmarks.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])

    return np.array(landmarks, dtype=np.float32)


# ============================================================
# 5. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


print("Webcam started.")
print("Press 'q' to quit.")


# ============================================================
# 6. MAIN LOOP
# ============================================================

prediction_history = deque(maxlen=7)
stable_prediction = "No hand detected"

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    # Mirror webcam for natural interaction
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert frame into MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    detection_result = detector.detect(mp_image)

    # Extract normalized landmarks
    landmarks = extract_landmarks(detection_result)

    prediction = "No hand detected"

    # ========================================================
    # HAND DETECTED
    # ========================================================

    if landmarks is not None:

        # Model expects:
        # (number_of_samples, number_of_features)
        features = pd.DataFrame(
            [landmarks],
            columns=FEATURE_COLUMNS
        )

        # Predict letter
        current_prediction = model.predict(features)[0]

        # Store recent predictions
        prediction_history.append(current_prediction)

        # Majority vote for stable prediction
        stable_prediction = Counter(
            prediction_history
        ).most_common(1)[0][0]

        prediction = stable_prediction

        # Draw hand landmarks
        for hand in detection_result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        prediction_history.clear()
        stable_prediction = "No hand detected"
        prediction = stable_prediction

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    cv2.rectangle(
        frame,
        (20, 20),
        (350, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ASL Letter Recognition",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# 7. CLEAN UP
# ============================================================

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Webcam closed.")