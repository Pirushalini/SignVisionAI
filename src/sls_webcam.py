import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_LANDMARKER_PATH = "notebooks/hand_landmarker.task"


# Create MediaPipe Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path=HAND_LANDMARKER_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam started.")
print("Show your hand in front of the camera.")
print("Press 'q' to quit.")


while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    detection_result = detector.detect(mp_image)

    # Draw landmarks
    if detection_result.hand_landmarks:

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
                    5,
                    (0, 255, 0),
                    -1
                )

        status = "Hand detected"

    else:
        status = "No hand detected"


    # Display status
    cv2.rectangle(
        frame,
        (20, 20),
        (350, 90),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        status,
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SLS Hand Detection",
        frame
    )


    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()

print("Webcam closed.")
