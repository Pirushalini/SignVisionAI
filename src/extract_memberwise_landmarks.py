import os
import re
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Paths
# =========================
DATASET_DIR = Path("data/SLS_Dataset")
MODEL_PATH = Path("notebooks/hand_landmarker.task")

OUTPUT_CSV = Path("data/sls_hand_landmarks_memberwise.csv")
FAILED_CSV = Path("results/memberwise_failed_images.csv")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)


# =========================
# CSV columns
# =========================
feature_columns = []

for i in range(21):
    feature_columns.extend([
        f"lm{i}_x",
        f"lm{i}_y",
        f"lm{i}_z",
    ])

output_columns = [
    "image_name",
    "member_id",
    "label",
] + feature_columns


# =========================
# Member mapping
# =========================
def get_member_id(image_name):
    match = re.search(r"_(\d+)\.(jpg|jpeg|png)$", image_name, re.IGNORECASE)

    if not match:
        return None

    number = int(match.group(1))

    if 1 <= number <= 20:
        return "Member_1"
    elif 21 <= number <= 40:
        return "Member_2"
    elif 41 <= number <= 60:
        return "Member_3"
    elif 61 <= number <= 80:
        return "Member_4"

    return None


# =========================
# Get already processed files
# =========================
processed_images = set()

if OUTPUT_CSV.exists():
    try:
        with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_name"):
                    processed_images.add(row["image_name"])
    except Exception as e:
        print(f"Warning: Could not read existing output CSV: {e}")

print(f"Already processed: {len(processed_images)} images")


# =========================
# CSV initialization
# =========================
if not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(output_columns)


if not FAILED_CSV.exists() or FAILED_CSV.stat().st_size == 0:
    with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "member_id",
            "label",
            "reason",
        ])


# =========================
# Collect images
# =========================
image_files = []

for letter_dir in sorted(DATASET_DIR.iterdir()):
    if not letter_dir.is_dir():
        continue

    label = letter_dir.name.upper()

    if not ("A" <= label <= "Z"):
        continue

    for image_path in sorted(letter_dir.iterdir()):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_files.append((image_path, label))


total = len(image_files)

print(f"Total images found: {total}")
print("Starting MediaPipe landmark extraction...")


# =========================
# MediaPipe setup
# =========================
base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.HandLandmarker.create_from_options(options)


# =========================
# Processing
# =========================
successful = len(processed_images)
failed = 0

# Load failed image names so resume does not repeat them
failed_images = set()

if FAILED_CSV.exists():
    try:
        with open(FAILED_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image_name"):
                    failed_images.add(row["image_name"])
    except Exception:
        pass


with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as output_file, \
     open(FAILED_CSV, "a", newline="", encoding="utf-8") as failed_file:

    output_writer = csv.writer(output_file)
    failed_writer = csv.writer(failed_file)

    try:
        for index, (image_path, label) in enumerate(image_files, start=1):

            image_name = image_path.name

            # Skip already completed images
            if image_name in processed_images:
                continue

            member_id = get_member_id(image_name)

            if member_id is None:
                failed_writer.writerow([
                    image_name,
                    "",
                    label,
                    "Could not determine member ID",
                ])
                failed_file.flush()
                failed += 1
                continue

            try:
                image = cv2.imread(str(image_path))

                if image is None:
                    failed_writer.writerow([
                        image_name,
                        member_id,
                        label,
                        "Could not read image",
                    ])
                    failed_file.flush()
                    failed += 1
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image,
                )

                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    failed_writer.writerow([
                        image_name,
                        member_id,
                        label,
                        "No hand detected",
                    ])
                    failed_file.flush()
                    failed += 1
                    continue

                landmarks = result.hand_landmarks[0]

                row = [
                    image_name,
                    member_id,
                    label,
                ]

                for lm in landmarks:
                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z,
                    ])

                output_writer.writerow(row)
                output_file.flush()

                processed_images.add(image_name)
                successful += 1

            except Exception as e:
                failed_writer.writerow([
                    image_name,
                    member_id,
                    label,
                    str(e),
                ])
                failed_file.flush()
                failed += 1

            # Progress
            if index % 50 == 0 or index == total:
                print(
                    f"Progress: {index}/{total} | "
                    f"Successful: {successful} | "
                    f"Failed: {failed}"
                )

    except KeyboardInterrupt:
        print("\nExtraction interrupted by user.")
        print("Already processed results are safely saved.")
        print("Run the same command again to resume.")

    finally:
        landmarker.close()


# =========================
# Final summary
# =========================
print("\n========================================")
print("Member-wise landmark extraction complete")
print("========================================")
print(f"Successful samples: {successful}")
print(f"Failed samples: {failed}")
print(f"Output CSV: {OUTPUT_CSV}")
print(f"Failed images: {FAILED_CSV}")

# Member counts
member_counts = {
    "Member_1": 0,
    "Member_2": 0,
    "Member_3": 0,
    "Member_4": 0,
}

if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            member = row.get("member_id")
            if member in member_counts:
                member_counts[member] += 1

print("\nMember-wise successful samples:")
for member, count in member_counts.items():
    print(f"{member}: {count}")
