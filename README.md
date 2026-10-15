# SignVisionAI

## Sri Lankan Sign Language Recognition System

SignVisionAI is a machine learning project focused on recognizing Sri Lankan Sign Language (SLS) hand signs using computer vision.

### Project Structure

- `data/SLS_Dataset/` — SLS image dataset organized into letter folders A–Z.
- `src/` — Source code for data processing, model training, and webcam recognition.
- `models/` — Trained machine learning models.
- `notebooks/` — Experiments and exploratory analysis.
- `results/` — Evaluation reports and visualizations.
- `requirements.txt` — Python dependencies.

### Project Goal

To develop a computer vision-based system that recognizes Sri Lankan Sign Language hand signs and displays the predicted letter.

### Status

- **Part 1:** Dataset Audit & MediaPipe Landmark Failure Analysis completed.
- **Part 2:** Landmark Extraction + Normalization completed (796 verified samples extracted using aspect-ratio-preserving square padding).
- **Part 3:** Distance-Based Recognition (1-NN / KNN) — ready for development.

---

## Technical Usage (Part 2: Landmark Normalization)

Part 2 implements a robust, leak-free landmark extraction and normalization pipeline for Sri Lankan Sign Language (SLS).

### Feature Representation
- **Single-Hand Signs (Letter C):** Exactly **63 float32 features** (21 landmarks $\times$ 3 coordinates $(x, y, z)$).
- **Two-Hand Signs (All other 25 letters):** Exactly **126 float32 features** ordered strictly as **Left Hand ($0 \dots 62$) followed by Right Hand ($63 \dots 125$)**, invariant to MediaPipe detection order.
- **Relative Wrist Displacement (Optional):** Preserves inter-hand 3D offset ($\Delta \mathbf{W} = \frac{\mathbf{L}_{0, R} - \mathbf{L}_{0, L}}{\bar{s}}$) as a separate **3 float32 features** (`shape: (3,)`), allowing Part 3 to benchmark baseline (126) vs augmented (129) representations.
- **Normalization:** Each hand is independently wrist-centered ($L_0 = (0, 0, 0)$) and scaled by palm size ($\|L_9 - L_0\|_2$).
- **Zero-Padding Policy:** Missing hands are never zero-padded.

### 1. Loading the Extracted Feature Library (for Part 3)
The pre-extracted normalized features from the 2,080-image SLS dataset are saved in `data/sls_normalized_square_pad.npz`:

```python
import numpy as np

# Load feature dataset
data = np.load("data/sls_normalized_square_pad.npz")

# 1-Hand signs (Letter C: 80 samples)
features_1h = data["features_1h"]          # Shape: (80, 63), dtype: float32
labels_1h = data["labels_1h"]              # Shape: (80,), all 'C'
image_ids_1h = data["image_ids_1h"]        # Source filenames for traceability

# 2-Hand signs (25 classes: A, B, D..Z: 716 samples)
features_2h = data["features_2h"]          # Shape: (716, 126), dtype: float32
labels_2h = data["labels_2h"]              # Shape: (716,)
image_ids_2h = data["image_ids_2h"]
rel_wrist_2h = data["relative_wrist_displacement_2h"]  # Shape: (716, 3)

# Optional: Construct augmented 129-feature representation
features_2h_augmented = np.concatenate([features_2h, rel_wrist_2h], axis=1)  # (716, 129)
```

### 2. Normalizing New Images or Live Webcam Frames
To process live webcam frames or test images with the exact same geometry:

```python
import cv2
import mediapipe as mp
from src.preprocessing import pad_to_square
from src.normalization import format_hand_features

# 1. Preprocess frame with aspect-ratio-preserving square padding
# (e.g. 1280x720 webcam -> 1280x1280 letterbox -> 1024x1024)
frame_padded = pad_to_square(frame, target_size=1024)

# 2. Run MediaPipe HandLandmarker detector on the padded frame
# ...

# 3. Format normalized features (strictly Left hand 0..62, Right hand 63..125)
result = format_hand_features(
    detected_hands=detection_result.hand_landmarks,
    handedness=detection_result.handedness,
    is_mirrored=True,  # True for mirrored selfie webcam, False for 3rd-person dataset photos
)

if result.is_valid:
    features = result.features  # (63,) for 1-hand, (126,) for 2-hand
    rel_wrist = result.relative_wrist_displacement  # (3,) for 2-hand
```

### 3. Running Unit Tests
```bash
python -m unittest discover tests
```
39 automated tests cover translation/scale invariance, landmark ordering, handedness resolution, aspect-ratio preservation, and coordinate consistency.

### 4. MediaPipe Model Setup (`hand_landmarker.task`)
To run landmark extraction or live detection, download the official Google MediaPipe Hand Landmarker model bundle (float16):

- **File Name:** `hand_landmarker.task` (~7.8 MB)
- **Destination:** `models/hand_landmarker.task`
- **Official Google Storage URL:**
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

**Download via PowerShell (Windows):**
```powershell
New-Item -ItemType Directory -Force -Path models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models/hand_landmarker.task"
```

**Download via curl (Linux / macOS):**
```bash
mkdir -p models
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```
