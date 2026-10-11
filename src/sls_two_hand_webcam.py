import cv2
import joblib
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/sls_two_hand_best_model.joblib"
HAND_MODEL_PATH = "notebooks/hand_landmarker.task"

# Keep the same feature order used during training.
FEATURE_COLUMNS = [
    f"{hand}_lm{i}_{axis}"
    for hand in ("left", "right")
    for i in range(21)
    for axis in ("x", "y", "z")
]


def extract_features(result):
    left = [0.0] * 63
    right = [0.0] * 63

    hands = result.hand_landmarks or []
    handedness = result.handedness or []

    for index, landmarks in enumerate(hands):
        features = []
        for landmark in landmarks:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        category = ""
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0].category_name.lower()

        if category == "left":
            left = features
        elif category == "right":
            right = features
        else:
            # Match the fallback approach used by dataset extraction.
            if left == [0.0] * 63:
                left = features
            elif right == [0.0] * 63:
                right = features

    return left + right


def main():
    model = joblib.load(MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("SLS Two-Hand Recognition started.")
    print("Show a sign to the camera. Press q to quit.")

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read from webcam.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    cv2.getTickCount() * 1000 / cv2.getTickFrequency()
                )

                result = detector.detect_for_video(
                    mp_image, timestamp_ms
                )

                hands = result.hand_landmarks or []

                if hands:
                    features = extract_features(result)
                    sample = pd.DataFrame(
                        [features],
                        columns=FEATURE_COLUMNS
                    )

                    prediction = str(model.predict(sample)[0])

                    confidence_text = ""
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(sample)[0]
                        confidence = float(probabilities.max()) * 100
                        confidence_text = f" | Confidence: {confidence:.1f}%"

                    display_text = f"Letter: {prediction}{confidence_text}"

                    cv2.putText(
                        frame,
                        display_text,
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 0),
                        2
                    )

                    # Draw detected hand landmarks.
                    height, width = frame.shape[:2]
                    for hand in hands:
                        for landmark in hand:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(
                                frame, (x, y), 4, (0, 255, 0), -1
                            )
                else:
                    cv2.putText(
                        frame,
                        "Show your hand sign",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )

                cv2.imshow("SignVisionAI - SLS Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
