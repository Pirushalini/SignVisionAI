"""
SignVisionAI - SLS Landmark Extraction

Extracts up to two hands from each SLS image.

Feature representation:
    Left hand  = 21 landmarks x (x, y, z) = 63 features
    Right hand = 21 landmarks x (x, y, z) = 63 features

Total:
    126 landmark features + label

For one-handed signs, the missing hand is represented by 63 zeros.

The same left/right ordering must be used during webcam inference.
"""

import csv
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Paths
# ============================================================

DATASET_DIR = Path("data/SLS_Dataset")
MODEL_PATH = Path("notebooks/hand_landmarker.task")

OUTPUT_CSV = Path("data/sls_hand_landmarks_clean.csv")
FAILED_CSV = Path("results/sls_failed_images.csv")


# ============================================================
# Configuration
# ============================================================

NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# Feature columns
# ============================================================

HAND_FEATURES = []

for hand_prefix in ["left", "right"]:
    for i in range(21):
        HAND_FEATURES.extend([
            f"{hand_prefix}_lm{i}_x",
            f"{hand_prefix}_lm{i}_y",
            f"{hand_prefix}_lm{i}_z",
        ])


OUTPUT_COLUMNS = ["label"] + HAND_FEATURES


# ============================================================
# Helpers
# ============================================================

def landmarks_to_features(landmarks):
    """
    Convert 21 MediaPipe landmarks into 63 values.
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
    Return 63 zero values for a missing hand.
    """

    return [0.0] * 63


def get_ordered_hand_features(result):
    """
    Return hand features in a fixed order:

        left_features
        right_features

    Missing hands are represented by zeros.

    MediaPipe provides handedness information for each
    detected hand, so we do not rely on detection order.
    """

    left_features = None
    right_features = None

    if not result.hand_landmarks:
        return (
            zero_hand_features(),
            zero_hand_features(),
        )

    for index, landmarks in enumerate(result.hand_landmarks):

        # Get handedness information corresponding
        # to this detected hand.
        handedness = result.handedness[index]

        if not handedness:
            continue

        category = handedness[0]

        hand_label = category.category_name.lower()

        features = landmarks_to_features(landmarks)

        if hand_label == "left":
            left_features = features

        elif hand_label == "right":
            right_features = features

    if left_features is None:
        left_features = zero_hand_features()

    if right_features is None:
        right_features = zero_hand_features()

    return left_features, right_features


# ============================================================
# Validate paths
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"MediaPipe model not found: {MODEL_PATH}"
    )


OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# MediaPipe setup
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# Prepare CSV
# ============================================================

print("=" * 70)
print("SignVisionAI - SLS Two-Hand Landmark Extraction")
print("=" * 70)

print(f"Dataset: {DATASET_DIR}")
print(f"Output : {OUTPUT_CSV}")
print()
print("Feature format:")
print("  Left hand  : 63 features")
print("  Right hand : 63 features")
print("  Total      : 126 features")
print()


# Always create a fresh clean dataset.
# This is important because the old CSV contains 63-feature rows.
with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as output_file:

    writer = csv.writer(output_file)
    writer.writerow(OUTPUT_COLUMNS)

    with open(
        FAILED_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as failed_file:

        failed_writer = csv.writer(failed_file)

        failed_writer.writerow([
            "image_name",
            "label",
            "reason",
        ])

        processed_count = 0
        failed_count = 0
        one_hand_count = 0
        two_hand_count = 0


        # ====================================================
        # Process dataset
        # ====================================================

        for letter_dir in sorted(DATASET_DIR.iterdir()):

            if not letter_dir.is_dir():
                continue

            label = letter_dir.name.upper()

            if not ("A" <= label <= "Z"):
                continue

            print(f"Processing {label}...")

            image_files = sorted([
                path
                for path in letter_dir.iterdir()
                if path.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            ])

            for image_path in image_files:

                try:

                    image = cv2.imread(
                        str(image_path)
                    )

                    if image is None:
                        failed_writer.writerow([
                            image_path.name,
                            label,
                            "Could not read image",
                        ])

                        failed_count += 1
                        continue


                    # ----------------------------------------
                    # BGR -> RGB
                    # ----------------------------------------

                    rgb_image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB
                    )


                    # ----------------------------------------
                    # MediaPipe image
                    # ----------------------------------------

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_image
                    )


                    # ----------------------------------------
                    # Detect up to two hands
                    # ----------------------------------------

                    result = landmarker.detect(
                        mp_image
                    )


                    if not result.hand_landmarks:

                        failed_writer.writerow([
                            image_path.name,
                            label,
                            "No hand detected",
                        ])

                        failed_count += 1
                        continue


                    # ----------------------------------------
                    # Get fixed left/right representation
                    # ----------------------------------------

                    left_features, right_features = (
                        get_ordered_hand_features(result)
                    )


                    # ----------------------------------------
                    # Count one-hand/two-hand samples
                    # ----------------------------------------

                    detected_hands = len(
                        result.hand_landmarks
                    )

                    if detected_hands == 1:
                        one_hand_count += 1

                    elif detected_hands >= 2:
                        two_hand_count += 1


                    # ----------------------------------------
                    # Final row
                    # ----------------------------------------

                    row = (
                        [label]
                        + left_features
                        + right_features
                    )


                    # Safety check
                    if len(row) != 127:

                        raise ValueError(
                            f"Expected 127 columns "
                            f"(label + 126 features), "
                            f"got {len(row)}"
                        )


                    writer.writerow(row)

                    processed_count += 1


                except Exception as error:

                    failed_writer.writerow([
                        image_path.name,
                        label,
                        str(error),
                    ])

                    failed_count += 1


# ============================================================
# Cleanup
# ============================================================

landmarker.close()


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("Extraction completed.")
print("=" * 70)

print(f"Processed images : {processed_count}")
print(f"Failed images    : {failed_count}")
print(f"One-hand samples : {one_hand_count}")
print(f"Two-hand samples : {two_hand_count}")

print()
print(f"Dataset saved to:")
print(f"  {OUTPUT_CSV}")

print()
print("Expected model input:")
print("  126 features")
print("  63 left-hand features")
print("  63 right-hand features")
print("=" * 70)