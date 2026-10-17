"""Live SLS letter recognition using MediaPipe and distance-based matching.

KEY RULE: the live frame must be processed exactly like the reference photos.
The reference NPZ was built from UNMIRRORED back-camera photos
(is_mirrored=False, swap_unmirrored=False, square padding). So:
  * the frame used for detection is NOT flipped (flipping mirrors every x
    coordinate and swaps left/right hands -> distances of 4-6),
  * handedness is resolved with the same settings as extraction,
  * only the picture shown on screen is mirrored (for comfort).

Keys:  Q/Esc quit | F flip detection frame (diagnostic) | T thresholds on/off
       D debug print on/off
"""

from collections import Counter, deque
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent))

from distance_recognition import DistanceRecognizer
from normalization import format_hand_features


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "sls_normalized_square_pad.npz"
HAND_MODEL_PATH = ROOT / "notebooks" / "hand_landmarker.task"

CAMERA_INDEX = 0
K_NEIGHBORS = 1

# Must match extraction (extract_landmarks.py used min_confidence=0.0).
MIN_HANDEDNESS_CONFIDENCE = 0.0

# Detection frame is NOT flipped (matches the unmirrored dataset photos).
FLIP_FOR_DETECTION = False
# Mirror only the displayed picture so it feels like a mirror.
MIRROR_DISPLAY = True

# Start with NO rejection so you can see real distances for correct signs.
# After it works, set limits slightly ABOVE the distances you see for correct
# signs (e.g. 1.3x the typical value) and press T to enable them.
THRESHOLD_1H = 2.0
THRESHOLD_2H = 3.5
USE_THRESHOLDS = False

SMOOTH_WINDOW = 7      # frames used for majority vote
SMOOTH_MIN_VOTES = 4   # votes needed to show a letter

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_detected_hands(display, hand_landmarks, side, left, top, mirror_x):
    """Draw landmarks. Landmarks are normalised to the padded SQUARE image,
    so convert with `side` and remove the padding offsets."""
    h, w = display.shape[:2]
    for hand in hand_landmarks:
        pts = []
        for lm in hand:
            x = int(lm.x * side) - left
            y = int(lm.y * side) - top
            if mirror_x:
                x = w - 1 - x
            pts.append((x, y))
        for a, b in CONNECTIONS:
            cv2.line(display, pts[a], pts[b], (80, 220, 80), 2)
        for p in pts:
            cv2.circle(display, p, 4, (40, 180, 255), -1)


def pad_square(frame):
    h, w = frame.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    padded = cv2.copyMakeBorder(
        frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return padded, side, left, top


def main():
    global USE_THRESHOLDS

    for path in (DATASET_PATH, HAND_MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    recognizer = DistanceRecognizer(
        reference_path=DATASET_PATH,
        k=K_NEIGHBORS,
        use_augmented_2h=False,
    )
    print("Reference library loaded:", recognizer.sample_counts)
    print("Keys: Q/Esc quit | F flip detection | T thresholds | D debug")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    flip_det = FLIP_FOR_DETECTION
    debug = True
    last_ts = 0
    last_debug = 0.0
    history = deque(maxlen=SMOOTH_WINDOW)

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, raw = camera.read()
                if not ok:
                    print("Could not read a frame from the webcam.")
                    break

                det_frame = cv2.flip(raw, 1) if flip_det else raw
                padded, side, left, top = pad_square(det_frame)

                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts = max(int(time.monotonic_ns() // 1_000_000), last_ts + 1)
                last_ts = ts
                detection = landmarker.detect_for_video(mp_image, ts)

                display = cv2.flip(det_frame, 1) if MIRROR_DISPLAY else det_frame.copy()
                # Landmarks live in det_frame coordinates; mirror x only if the
                # display is mirrored relative to the detection frame.
                mirror_x = MIRROR_DISPLAY
                draw_detected_hands(
                    display, detection.hand_landmarks, side, left, top, mirror_x
                )

                hand_count = len(detection.hand_landmarks)
                title = "Show a sign (both hands, except C)"
                detail = f"Hands detected: {hand_count}"
                shown_letter = None

                if hand_count in (1, 2):
                    features = format_hand_features(
                        detection,
                        is_mirrored=False,          # same as dataset extraction
                        min_confidence=MIN_HANDEDNESS_CONFIDENCE,
                        strict=False,
                        swap_unmirrored=False,      # same as dataset extraction
                    )

                    if not features.is_valid:
                        history.clear()
                        title = "Uncertain: hands not resolved"
                        detail = features.error_message or features.status
                    else:
                        t1 = THRESHOLD_1H if USE_THRESHOLDS else None
                        t2 = THRESHOLD_2H if USE_THRESHOLDS else None
                        prediction = recognizer.predict(
                            features,
                            k=K_NEIGHBORS,
                            threshold=t1 if features.hand_count == 1 else t2,
                        )

                        now = time.monotonic()
                        if debug and now - last_debug >= 1.0:
                            top1 = (prediction.get("top_k") or [{}])[0]
                            print(
                                f"hands={hand_count} status={prediction['status']} "
                                f"letter={prediction['letter']} "
                                f"nearest={top1.get('letter')} "
                                f"dist={prediction['distance']} "
                                f"img={top1.get('image_id')} "
                                f"flip_det={flip_det} thresholds={USE_THRESHOLDS}"
                            )
                            last_debug = now

                        if prediction["status"] == "recognized":
                            history.append(prediction["letter"])
                        else:
                            history.append(None)

                        votes = Counter(l for l in history if l is not None)
                        if votes:
                            best, n = votes.most_common(1)[0]
                            if n >= SMOOTH_MIN_VOTES:
                                shown_letter = best

                        if shown_letter:
                            title = f"Letter: {shown_letter}"
                        elif prediction["status"] == "uncertain":
                            title = "Uncertain: sign not close enough"
                        elif prediction["status"] == "recognized":
                            title = "Hold the sign steady..."
                        else:
                            title = "Invalid detection"

                        top1 = (prediction.get("top_k") or [{}])[0]
                        if prediction.get("distance") is not None:
                            detail = (
                                f"Closest: {top1.get('letter')}  "
                                f"dist {prediction['distance']:.3f}"
                            )
                            if prediction.get("threshold") is not None:
                                detail += f" / limit {prediction['threshold']:.3f}"
                        if hand_count == 1:
                            detail += "  (1 hand = only letter C)"
                elif hand_count == 0:
                    history.clear()
                    title = "No hands detected"
                else:
                    title = "Uncertain: show at most 2 hands"

                cv2.rectangle(display, (10, 10), (display.shape[1] - 10, 95),
                              (25, 25, 25), -1)
                cv2.putText(display, title, (25, 48), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (80, 255, 120), 2, cv2.LINE_AA)
                cv2.putText(display, detail[:100], (25, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)
                status_line = (
                    f"flip_det={flip_det}  thresholds={'ON' if USE_THRESHOLDS else 'OFF'}"
                    "   Q/Esc quit | F flip | T thresholds | D debug"
                )
                cv2.putText(display, status_line, (20, display.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                            cv2.LINE_AA)

                cv2.imshow("SignVisionAI - Distance Recognition", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("f"):
                    flip_det = not flip_det
                    history.clear()
                    print("Detection flip ->", flip_det)
                elif key == ord("t"):
                    USE_THRESHOLDS = not USE_THRESHOLDS
                    history.clear()
                    print("Thresholds ->", USE_THRESHOLDS)
                elif key == ord("d"):
                    debug = not debug
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()