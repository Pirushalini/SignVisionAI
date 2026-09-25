import cv2
import csv
import glob
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DATASET_DIR = "data/SLS_Dataset"
OUTPUT_FILE = "data/sls_hand_landmarks.csv"
MODEL_PATH = "notebooks/hand_landmarker.task"


# Create MediaPipe Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# Prepare output directory
os.makedirs("data", exist_ok=True)

rows = []

image_files = sorted(
    glob.glob(os.path.join(DATASET_DIR, "*.jpeg"))
)

print(f"Found {len(image_files)} images.")


for image_path in image_files:

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = detector.detect(mp_image)

    filename = os.path.basename(image_path)
    label = os.path.splitext(filename)[0].upper()

    if not result.hand_landmarks:
        print(f"No hand detected: {filename}")
        continue

    landmarks = result.hand_landmarks[0]

    # Store x, y, z for all 21 landmarks
    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    rows.append([label] + features)

    print(f"Processed: {filename} → {label}")


detector.close()


# Create CSV
header = ["label"]

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(header)
    writer.writerows(rows)


print()
print(f"Saved: {OUTPUT_FILE}")
print(f"Successful images: {len(rows)}")
print(f"Expected images: 26")
