import os
import csv
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Download Model
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading MediaPipe Hand Landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded successfully!")

# 2. Setup Detector for BOTH HANDS (num_hands=2)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Define Paths
DATASET_DIR = os.path.join("data", "SLS_Dataset")
OUTPUT_CSV = os.path.join("data", "sls_hand_landmarks_clean.csv")

# Create Header for 2 Hands (126 features + label)
header = ['label']
# Hand 1 features
for i in range(21):
    header.extend([f'h1_lm{i}_x', f'h1_lm{i}_y', f'h1_lm{i}_z'])
# Hand 2 features
for i in range(21):
    header.extend([f'h2_lm{i}_x', f'h2_lm{i}_y', f'h2_lm{i}_z'])

print("🚀 Starting Data Preprocessing for BOTH hands...")

processed_count = 0
failed_count = 0

os.makedirs("data", exist_ok=True)

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Directory '{DATASET_DIR}' not found!")
    else:
        for letter_folder in sorted(os.listdir(DATASET_DIR)):
            folder_path = os.path.join(DATASET_DIR, letter_folder)

            if os.path.isdir(folder_path):
                print(f"📸 Processing Folder: {letter_folder}...")

                for img_name in sorted(os.listdir(folder_path)):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(folder_path, img_name)

                        try:
                            mp_image = mp.Image.create_from_file(img_path)
                            detection_result = detector.detect(mp_image)

                            if detection_result.hand_landmarks:
                                landmarks_list = detection_result.hand_landmarks
                                row = [letter_folder]

                                # 1st Hand Coordinates
                                for lm in landmarks_list[0]:
                                    row.extend([lm.x, lm.y, lm.z])

                                # 2nd Hand Coordinates (If 2nd hand is present, add it. Otherwise fill zeros)
                                if len(landmarks_list) > 1:
                                    for lm in landmarks_list[1]:
                                        row.extend([lm.x, lm.y, lm.z])
                                else:
                                    # Fill 63 zeros for the missing second hand
                                    row.extend([0.0] * 63)

                                writer.writerow(row)
                                processed_count += 1
                            else:
                                failed_count += 1
                        except Exception as e:
                            failed_count += 1

print("\n" + "="*50)
print("✅ Extraction Finished Successfully for Both Hands!")
print(f"📊 Total Rows Saved: {processed_count}")
print(f"⚠️ Skipped Images: {failed_count}")
print(f"📁 Output Saved To: {OUTPUT_CSV}")
print("="*50)