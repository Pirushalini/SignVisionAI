"""
SignVisionAI - Member 3: Real-Time SLS Webcam Recognition

Uses:
    Webcam
        ↓
    MediaPipe Hand Landmarker
        ↓
    Left + Right hand landmarks
        ↓
    126 features
        ↓
    Saved SVM pipeline
        ↓
    A-Z prediction
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/sls_best_model.joblib"
)

HAND_LANDMARKER_PATH = Path(
    "notebooks/hand_landmarker.task"
)

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SMOOTHING_WINDOW = 8


# ============================================================
# Validate files
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not HAND_LANDMARKER_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: "
        f"{HAND_LANDMARKER_PATH}"
    )


# ============================================================
# Load model
# ============================================================

print("Loading SLS model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")
print(
    f"Model expects "
    f"{model.n_features_in_} features."
)


if model.n_features_in_ != 126:
    raise ValueError(
        f"This webcam recognizer requires a "
        f"126-feature model, but the loaded model "
        f"expects {model.n_features_in_} features."
    )


# ============================================================
# MediaPipe setup
# ============================================================

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(
            HAND_LANDMARKER_PATH
        )
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# Hand connections
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (0, 17),

    (17, 18),
    (18, 19),
    (19, 20),
]


# ============================================================
# Feature helpers
# ============================================================

def landmarks_to_features(landmarks):
    """
    Convert one hand into 63 x/y/z features.
    """

    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    return features


def zero_hand_features():
    """
    Return 63 zeros for a missing hand.
    """

    return [0.0] * 63


def get_ordered_hand_features(result):
    """
    Return:

        left_features + right_features

    Always 126 values.

    Missing hands are zero-padded.
    """

    left_features = None
    right_features = None

    if not result.hand_landmarks:
        return (
            zero_hand_features(),
            zero_hand_features(),
        )

    for index, landmarks in enumerate(
        result.hand_landmarks
    ):

        handedness = result.handedness[index]

        if not handedness:
            continue

        hand_label = (
            handedness[0]
            .category_name
            .lower()
        )

        features = landmarks_to_features(
            landmarks
        )

        if hand_label == "left":
            left_features = features

        elif hand_label == "right":
            right_features = features


    if left_features is None:
        left_features = zero_hand_features()

    if right_features is None:
        right_features = zero_hand_features()


    return (
        left_features,
        right_features,
    )


def get_model_features(result):
    """
    Convert MediaPipe detection result into
    exactly 126 features.
    """

    left_features, right_features = (
        get_ordered_hand_features(result)
    )

    features = (
        left_features
        + right_features
    )

    return np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# ============================================================
# Drawing
# ============================================================

def draw_hand(
    frame,
    landmarks,
    handedness_label
):
    """
    Draw landmarks and connections.
    """

    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )


    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )


    # Label hand
    wrist_x, wrist_y = points[0]

    cv2.putText(
        frame,
        handedness_label,
        (
            wrist_x + 10,
            wrist_y - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


# ============================================================
# Prediction smoothing
# ============================================================

prediction_history = []


def smooth_prediction(prediction):

    prediction_history.append(
        prediction
    )

    if len(prediction_history) > SMOOTHING_WINDOW:
        prediction_history.pop(0)


    counts = {}

    for item in prediction_history:
        counts[item] = (
            counts.get(item, 0) + 1
        )


    return max(
        counts,
        key=counts.get
    )


# ============================================================
# Main
# ============================================================

def main():

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )


    print()
    print("Starting webcam...")
    print("Press Q to quit.")
    print()


    timestamp_ms = 0


    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                print(
                    "Failed to read webcam frame."
                )
                break


            # ------------------------------------------------
            # IMPORTANT:
            # Detect on the original frame.
            #
            # We flip only after detection so that
            # MediaPipe's left/right handedness remains
            # consistent with the training images.
            # ------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )


            timestamp_ms += 33


            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )


            predicted_letter = "-"
            probability = 0.0


            # ------------------------------------------------
            # Draw detected hands
            # ------------------------------------------------

            if result.hand_landmarks:

                for index, landmarks in enumerate(
                    result.hand_landmarks
                ):

                    handedness = (
                        result.handedness[index]
                    )

                    if handedness:

                        label = (
                            handedness[0]
                            .category_name
                        )

                        draw_hand(
                            frame,
                            landmarks,
                            label
                        )


                # --------------------------------------------
                # Convert to 126 model features
                # --------------------------------------------

                features = (
                    get_model_features(
                        result
                    )
                )


                if features.shape != (1, 126):

                    raise ValueError(
                        "Invalid webcam feature "
                        f"shape: {features.shape}"
                    )


                # --------------------------------------------
                # Prediction
                # --------------------------------------------

                prediction = model.predict(
                    features
                )[0]


                probabilities = (
                    model.predict_proba(
                        features
                    )[0]
                )


                probability = float(
                    np.max(probabilities)
                )


                predicted_letter = (
                    smooth_prediction(
                        prediction
                    )
                )


            else:

                prediction_history.clear()


            # =================================================
            # Flip for user-friendly display
            # =================================================

            frame = cv2.flip(
                frame,
                1
            )


            # =================================================
            # Prediction panel
            # =================================================

            cv2.rectangle(
                frame,
                (20, 20),
                (430, 155),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                f"Letter: {predicted_letter}",
                (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 255, 255),
                3
            )


            cv2.putText(
                frame,
                f"Probability: {probability:.2f}",
                (40, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                "Q = quit",
                (20, FRAME_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.imshow(
                "SignVisionAI - SLS Webcam Recognition",
                frame
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key == ord("q"):
                break


    cap.release()
    cv2.destroyAllWindows()

    print(
        "Webcam recognition stopped."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()