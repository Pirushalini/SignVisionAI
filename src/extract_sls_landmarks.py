import os
import csv
import urllib.request
import tempfile
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 1. MediaPipe model
MODEL_PATH = "notebooks/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading MediaPipe Hand Landmarker...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# 2. Paths
DATASET_DIR = Path("data/SLS_Dataset")
OUTPUT_CSV = Path("data/sls_two_hand_landmarks.csv")

# Never overwrite an existing output file
if OUTPUT_CSV.exists():
    raise FileExistsError(
        f"{OUTPUT_CSV} already exists. "
        "Rename or back it up before running extraction again."
    )

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")


# 3. CSV header: label + 126 coordinates
header = ["label"]

for hand in ("left", "right"):
    for i in range(21):
        header.extend([
            f"{hand}_lm{i}_x",
            f"{hand}_lm{i}_y",
            f"{hand}_lm{i}_z",
        ])


# 4. Detection counters
total_images = 0
one_hand_count = 0
two_hand_count = 0
no_hand_count = 0
error_count = 0
unknown_handedness_count = 0

image_extensions = {".jpg", ".jpeg", ".png"}

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
)

# Temporary file prevents leaving a partial final CSV on failure
temp_path = None

try:
    with vision.HandLandmarker.create_from_options(options) as detector:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            suffix=".csv",
            dir=str(OUTPUT_CSV.parent),
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.writer(temp_file)
            writer.writerow(header)

            for folder in sorted(DATASET_DIR.iterdir()):
                if not folder.is_dir():
                    continue

                label = folder.name.upper()

                if len(label) != 1 or not ("A" <= label <= "Z"):
                    print(f"Skipping unexpected folder: {folder.name}")
                    continue

                print(f"Processing {label}...")

                for image_path in sorted(folder.iterdir()):
                    if image_path.suffix.lower() not in image_extensions:
                        continue

                    total_images += 1

                    try:
                        mp_image = mp.Image.create_from_file(
                            str(image_path)
                        )
                        result = detector.detect(mp_image)

                        detected_hands = result.hand_landmarks or []
                        handedness = result.handedness or []

                        if len(detected_hands) == 0:
                            no_hand_count += 1
                            continue

                        if len(detected_hands) == 1:
                            one_hand_count += 1
                        else:
                            two_hand_count += 1

                        left_features = [0.0] * 63
                        right_features = [0.0] * 63
                        unknown_features = []

                        for index, landmarks in enumerate(detected_hands):
                            features = []
                            for lm in landmarks:
                                features.extend([lm.x, lm.y, lm.z])

                            category = ""
                            if index < len(handedness) and handedness[index]:
                                category = (
                                    handedness[index][0].category_name.lower()
                                )

                            if category == "left":
                                left_features = features
                            elif category == "right":
                                right_features = features
                            else:
                                unknown_handedness_count += 1
                                unknown_features.append(features)

                        # Do not silently discard a detected hand if its
                        # handedness label is unavailable.
                        for features in unknown_features:
                            if left_features == [0.0] * 63:
                                left_features = features
                            elif right_features == [0.0] * 63:
                                right_features = features

                        writer.writerow(
                            [label] + left_features + right_features
                        )

                    except Exception as exc:
                        error_count += 1
                        print(f"Error processing {image_path}: {exc}")

    # Publish the completed CSV only after extraction finishes
    os.replace(temp_path, OUTPUT_CSV)
    temp_path = None

finally:
    if temp_path is not None and temp_path.exists():
        temp_path.unlink()


print("\n========== EXTRACTION SUMMARY ==========")
print(f"Images examined:          {total_images}")
print(f"One hand detected:        {one_hand_count}")
print(f"Two hands detected:       {two_hand_count}")
print(f"No hands detected:        {no_hand_count}")
print(f"Image processing errors:  {error_count}")
print(f"Unknown handedness labels: {unknown_handedness_count}")
print(f"CSV saved to:             {OUTPUT_CSV}")
print("========================================")
