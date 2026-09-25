import cv2
import csv
import numpy as np
import mediapipe as mp
from collections import deque

# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "data/sls_hand_landmarks.csv"
MODEL_PATH = "notebooks/hand_landmarker.task"

# Recognition stability settings
prediction_history = deque(maxlen=10)

# Lower = stricter matching
DISTANCE_THRESHOLD = 1.5

# Minimum number of stable frames before showing a letter
MIN_STABLE_FRAMES = 5


# -----------------------------
# Load SLS reference landmarks
# -----------------------------
references = {}

with open(CSV_PATH, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        label = row["label"]

        landmarks = []

        for i in range(21):
            x = float(row[f"x{i}"])
            y = float(row[f"y{i}"])
            z = float(row[f"z{i}"])

            landmarks.extend([x, y, z])

        references[label] = np.array(
            landmarks,
            dtype=np.float32
        )

print(f"Loaded {len(references)} SLS reference letters.")


# -----------------------------
# Normalize landmarks
# -----------------------------
def normalize_landmarks(landmarks):

    points = np.array(
        landmarks,
        dtype=np.float32
    ).reshape(21, 3)

    # Make wrist the origin
    points = points - points[0]

    # Calculate hand size
    distances_from_wrist = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(distances_from_wrist)

    # Scale normalization
    if scale > 0:
        points = points / scale

    return points.flatten()


# -----------------------------
# Normalize reference data
# -----------------------------
normalized_references = {
    label: normalize_landmarks(data)
    for label, data in references.items()
}


# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------
BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# -----------------------------
# Start webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Could not open webcam.")

    exit()


print("Webcam started.")
print("Show an SLS sign to the camera.")
print("Press Q to quit.")


# -----------------------------
# Start MediaPipe
# -----------------------------
with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # -------------------------
        # Read frame
        # -------------------------
        ret, frame = cap.read()

        if not ret:

            print("Failed to read frame.")

            break


        # Mirror image
        frame = cv2.flip(
            frame,
            1
        )


        # -------------------------
        # Convert BGR → RGB
        # -------------------------
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------
        # Create MediaPipe image
        # -------------------------
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # -------------------------
        # Detect hand
        # -------------------------
        result = landmarker.detect(
            mp_image
        )


        # Default display values
        detected_letter = "No hand"
        best_distance = None


        # -------------------------
        # Hand detected
        # -------------------------
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ---------------------
            # Extract 21 landmarks
            # ---------------------
            current_landmarks = []

            for landmark in hand:

                current_landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])


            # ---------------------
            # Normalize hand
            # ---------------------
            current_vector = (
                normalize_landmarks(
                    current_landmarks
                )
            )


            # ---------------------
            # Compare with references
            # ---------------------
            distances = {}

            for label, reference_vector in (
                normalized_references.items()
            ):

                distance = np.linalg.norm(
                    current_vector -
                    reference_vector
                )

                distances[label] = distance


            # ---------------------
            # Find closest letter
            # ---------------------
            candidate_letter = min(
                distances,
                key=distances.get
            )

            best_distance = (
                distances[candidate_letter]
            )


            # ---------------------
            # Distance threshold
            # ---------------------
            if best_distance <= DISTANCE_THRESHOLD:

                prediction_history.append(
                    candidate_letter
                )

            else:

                prediction_history.clear()


            # ---------------------
            # Stable prediction
            # ---------------------
            if len(prediction_history) >= MIN_STABLE_FRAMES:

                counts = {}

                for letter in prediction_history:

                    counts[letter] = (
                        counts.get(letter, 0) + 1
                    )


                detected_letter = max(
                    counts,
                    key=counts.get
                )

            else:

                detected_letter = "Checking..."


            # ---------------------
            # Draw landmarks
            # ---------------------
            height, width, _ = (
                frame.shape
            )

            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


        else:

            # Clear old predictions
            prediction_history.clear()


        # -----------------------------
        # Display background
        # -----------------------------
        cv2.rectangle(
            frame,
            (20, 20),
            (430, 125),
            (0, 0, 0),
            -1
        )


        # -----------------------------
        # Display detected letter
        # -----------------------------
        cv2.putText(
            frame,
            f"SLS Letter: {detected_letter}",
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2
        )


        # -----------------------------
        # Display distance
        # -----------------------------
        if best_distance is not None:

            cv2.putText(
                frame,
                f"Distance: {best_distance:.3f}",
                (35, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )


        # -----------------------------
        # Display threshold
        # -----------------------------
        cv2.putText(
            frame,
            f"Threshold: {DISTANCE_THRESHOLD:.2f}",
            (35, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


        # -----------------------------
        # Show webcam
        # -----------------------------
        cv2.imshow(
            "SLS Sign Language Reader",
            frame
        )


        # -----------------------------
        # Quit with Q
        # -----------------------------
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()

cv2.destroyAllWindows()

print("Webcam stopped.")