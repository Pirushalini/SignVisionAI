"""
SignVisionAI - Member-wise 2-Hand Landmark Extraction

Extracts up to two hands from each SLS dataset image.

Feature order:
    Left hand  = 63 features
    Right hand = 63 features

Total:
    126 landmark features

Missing hands are zero-padded.

The member ID is inferred from the image number:
    01-20 -> Member_1
    21-40 -> Member_2
    41-60 -> Member_3
    61-80 -> Member_4
"""

import csv
import re
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

OUTPUT_CSV = Path("data/sls_hand_landmarks_memberwise.csv")
FAILED_CSV = Path("results/memberwise_failed_images.csv")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Feature definitions
# ============================================================

def make_hand_feature_columns(prefix):
    columns = []

    for i in range(21):
        columns.extend([
            f"{prefix}_lm{i}_x",
            f"{prefix}_lm{i}_y",
            f"{prefix}_lm{i}_z",
        ])

    return columns


LEFT_FEATURE_COLUMNS = make_hand_feature_columns("left")
RIGHT_FEATURE_COLUMNS = make_hand_feature_columns("right")

FEATURE_COLUMNS = (
    LEFT_FEATURE_COLUMNS
    + RIGHT_FEATURE_COLUMNS
)

OUTPUT_COLUMNS = [
    "image_name",
    "member_id",
    "label",
] + FEATURE_COLUMNS


# ============================================================
# Member mapping
# ============================================================

def get_member_id(image_name):
    """
    Determine member from image number.

    01-20 -> Member_1
    21-40 -> Member_2
    41-60 -> Member_3
    61-80 -> Member_4
    """

    match = re.search(
        r"_(\d+)\.(jpg|jpeg|png)$",
        image_name,
        re.IGNORECASE
    )

    if not match:
        return None

    number = int(match.group(1))

    if 1 <= number <= 20:
        return "Member_1"

    if 21 <= number <= 40:
        return "Member_2"

    if 41 <= number <= 60:
        return "Member_3"

    if 61 <= number <= 80:
        return "Member_4"

    return None


# ============================================================
# Feature helpers
# ============================================================

def enhance_contrast_for_detection(image_bgr):
    """Create a contrast-enhanced copy without modifying the source image."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def landmarks_to_features(landmarks):
    """
    Convert one MediaPipe hand into 63 x/y/z values.
    """

    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z,
        ])

    if len(features) != 63:
        raise ValueError(
            f"Expected 63 hand features, got {len(features)}"
        )

    return features


def zero_hand_features():
    """
    Return 63 zero values for a missing hand.
    """

    return [0.0] * 63


def get_ordered_hand_features(result):
    """
    Return:

        left_features + right_features

    Always exactly 126 values.

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


# ============================================================
# Collect images
# ============================================================

image_files = []

for letter_dir in sorted(
    DATASET_DIR.iterdir()
):

    if not letter_dir.is_dir():
        continue

    label = letter_dir.name.upper()

    if not ("A" <= label <= "Z"):
        continue

    for image_path in sorted(
        letter_dir.iterdir()
    ):

        if image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
        ]:
            image_files.append(
                (image_path, label)
            )


print("=" * 70)
print("SignVisionAI - Member-wise 2-Hand Landmark Extraction")
print("=" * 70)

print(
    f"Dataset directory : {DATASET_DIR}"
)

print(
    f"Total images       : {len(image_files)}"
)

print(
    f"Feature count      : {len(FEATURE_COLUMNS)}"
)

print(
    "Expected           : 126"
)

print()


# ============================================================
# MediaPipe setup
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

landmarker = (
    vision.HandLandmarker
    .create_from_options(options)
)


# ============================================================
# Statistics
# ============================================================

successful = 0
failed = 0

one_hand = 0
two_hands = 0
zero_hands = 0

member_counts = {
    "Member_1": 0,
    "Member_2": 0,
    "Member_3": 0,
    "Member_4": 0,
}


# ============================================================
# Fresh output files
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as output_file, open(
    FAILED_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as failed_file:

    writer = csv.writer(output_file)

    failed_writer = csv.writer(
        failed_file
    )

    writer.writerow(
        OUTPUT_COLUMNS
    )

    failed_writer.writerow([
        "image_name",
        "member_id",
        "label",
        "reason",
    ])


    # ========================================================
    # Process images
    # ========================================================

    for counter, (
        image_path,
        label
    ) in enumerate(
        image_files,
        start=1
    ):

        image_name = image_path.name

        member_id = get_member_id(
            image_name
        )

        if member_id is None:

            failed += 1

            failed_writer.writerow([
                image_name,
                "",
                label,
                "Could not determine member ID",
            ])

            continue


        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            failed += 1

            failed_writer.writerow([
                image_name,
                member_id,
                label,
                "Could not read image",
            ])

            continue


        # ----------------------------------------------------
        # MediaPipe detection with fallback retry
        # ----------------------------------------------------

        try:

            # First attempt: original image
            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            result = landmarker.detect(mp_image)
            detection_method = "original"

            # Second attempt: retry with enhanced contrast if no hands detected
            if len(result.hand_landmarks) == 0:
                enhanced_bgr = enhance_contrast_for_detection(image)
                enhanced_rgb = cv2.cvtColor(
                    enhanced_bgr,
                    cv2.COLOR_BGR2RGB
                )

                enhanced_mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=enhanced_rgb
                )

                retry_result = landmarker.detect(enhanced_mp_image)

                if len(retry_result.hand_landmarks) > 0:
                    result = retry_result
                    detection_method = "enhanced_retry"
                else:
                    detection_method = "none"

        except Exception as e:

            failed += 1

            failed_writer.writerow([
                image_name,
                member_id,
                label,
                f"MediaPipe error: {e}",
            ])

            continue


        # ----------------------------------------------------
        # Count detected hands
        # ----------------------------------------------------

        detected_hands = len(
            result.hand_landmarks
        )


        if detected_hands == 0:

            zero_hands += 1

        elif detected_hands == 1:

            one_hand += 1

        else:

            two_hands += 1


        # ----------------------------------------------------
        # Get ordered 126 features
        # ----------------------------------------------------

        try:

            left_features, right_features = (
                get_ordered_hand_features(
                    result
                )
            )

        except Exception as e:

            failed += 1

            failed_writer.writerow([
                image_name,
                member_id,
                label,
                f"Feature extraction error: {e}",
            ])

            continue


        features = (
            left_features
            + right_features
        )


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if len(features) != 126:

            failed += 1

            failed_writer.writerow([
                image_name,
                member_id,
                label,
                (
                    "Invalid feature count: "
                    f"{len(features)}"
                ),
            ])

            continue


        # ----------------------------------------------------
        # Write row
        # ----------------------------------------------------

        writer.writerow([
            image_name,
            member_id,
            label,
        ] + features)

        successful += 1

        member_counts[member_id] += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            counter % 100 == 0
            or counter == len(image_files)
        ):

            print(
                f"[{counter}/{len(image_files)}] "
                f"Processed: {successful} | "
                f"Failed: {failed}"
            )


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

print(
    f"Successful samples : {successful}"
)

print(
    f"Failed samples     : {failed}"
)

print(
    f"Zero hands         : {zero_hands}"
)

print(
    f"One hand           : {one_hand}"
)

print(
    f"Two hands          : {two_hands}"
)

print()
print("Samples by member:")

for member, count in member_counts.items():

    print(
        f"  {member}: {count}"
    )

print()
print(
    f"Saved dataset: {OUTPUT_CSV}"
)

print(
    f"Saved failures: {FAILED_CSV}"
)

print()
print("=" * 70)