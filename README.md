# SignVisionAI

SignVisionAI is a computer-vision and machine-learning project for recognizing
alphabetical sign-language gestures from hand images and live webcam video.
The project uses MediaPipe Hand Landmarker to convert images or video frames
into hand landmark coordinates, then uses scikit-learn classifiers to predict
one of the 26 letters from **A** through **Z**.

The repository contains two related recognition pipelines:

- **One-hand pipeline** - uses 63 features for one hand (21 landmarks x
  `x`, `y`, and `z` coordinates).
- **Two-hand pipeline** - uses 126 features for two hands, with separate
  left-hand and right-hand feature groups.

The two-hand pipeline also includes a real-time OpenCV webcam application.

## Features

- MediaPipe hand landmark extraction from the included image dataset.
- Support for one-hand and two-hand landmark CSV datasets.
- Model comparison across:
  - Support Vector Machine (SVM)
  - Random Forest
  - Multi-Layer Perceptron (MLP)
- Stratified train/test splitting with a reproducible random seed.
- Five-fold stratified cross-validation and optional hyperparameter tuning.
- Accuracy, precision, recall, F1-score, classification reports, and confusion
  matrices.
- Export of the complete fitted scikit-learn pipeline with Joblib.
- Evaluation scripts for trained models.
- Person-independent evaluation and landmark normalization experiments.
- Live two-hand gesture recognition from a webcam.
- A Jupyter notebook for interactive model development.

## Project structure

```text
SignVisionAI/
├── data/
│   ├── SLS_Dataset/                         # Input images grouped by A-Z
│   ├── sls_hand_landmarks_clean.csv         # One-hand training data
│   ├── sls_hand_landmarks_memberwise.csv    # Member-aware one-hand data
│   └── sls_two_hand_landmarks.csv           # Two-hand training data
├── models/
│   ├── sls_best_model.joblib                # Best one-hand pipeline
│   ├── sls_model_metadata.json
│   ├── sls_improved_model.joblib
│   ├── sls_normalized_svm.joblib
│   ├── sls_two_hand_best_model.joblib       # Best two-hand pipeline
│   └── sls_two_hand_model_metadata.json
├── notebooks/
│   ├── 02_ml_model_development.ipynb
│   └── hand_landmarker.task                  # MediaPipe model asset
├── results/                                  # One-hand/evaluation results
├── results_two_hand/                         # Two-hand training results
├── src/
│   ├── extract_sls_landmarks.py              # Build two-hand landmarks
│   ├── extract_memberwise_landmarks.py       # Build member-aware landmarks
│   ├── train_sls_models.py                   # Train one-hand models
│   ├── train_sls_two_hand_models.py          # Train two-hand models
│   ├── evaluate_sls_model.py                 # Evaluate one-hand model
│   ├── sls_two_hand_webcam.py                # Live webcam inference
│   └── member3_*.py                          # Normalization/evaluation studies
└── README.md
```

## Requirements

- Python 3.10 or newer
- A working webcam for live inference
- The MediaPipe Hand Landmarker model asset
- Python packages:
  - `mediapipe`
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `joblib`
  - `jupyter` (optional, for notebooks)

The repository currently uses local model and dataset files rather than a
packaged Python distribution. Commands below should be run from the repository
root so that the relative paths used by the scripts resolve correctly.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Pirushalini/SignVisionAI.git
cd SignVisionAI
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install mediapipe opencv-python numpy pandas scikit-learn matplotlib joblib jupyter
```

If MediaPipe does not provide a wheel for the Python version in use, create
the environment with a Python version supported by the MediaPipe release you
are installing.

## Dataset layout

The landmark extraction script expects the source images in folders named for
their labels:

```text
data/SLS_Dataset/
├── A/
│   ├── A_001.jpg
│   └── ...
├── B/
├── ...
└── Z/
```

Only folders whose names are a single letter from `A` to `Z` are processed.
Supported image extensions are `.jpg`, `.jpeg`, and `.png`.

The checked-in dataset contains 26 letter directories and approximately 2,080
images. If you replace the dataset, keep the same directory and label
conventions.

## MediaPipe model asset

`src/extract_sls_landmarks.py` downloads the Hand Landmarker asset
automatically to `notebooks/hand_landmarker.task` if it is missing. The
webcam script expects the same file at that path.

You can also download the asset manually from the official MediaPipe model
hosting location:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Quick start: live webcam recognition

The webcam demo uses the exported two-hand model and MediaPipe's video mode:

```bash
python src/sls_two_hand_webcam.py
```

The application opens the default webcam, draws detected hand landmarks, and
shows the predicted letter and confidence. Press **`q`** in the OpenCV window
to quit.

The webcam script expects:

```text
models/sls_two_hand_best_model.joblib
notebooks/hand_landmarker.task
```

To use a different model or asset location, update `MODEL_PATH` and
`HAND_MODEL_PATH` in `src/sls_two_hand_webcam.py`.

## Build the landmark datasets

### Two-hand landmarks

Run this from the repository root:

```bash
python src/extract_sls_landmarks.py
```

The script:

1. Loads images from `data/SLS_Dataset`.
2. Detects up to two hands with MediaPipe.
3. Stores 21 landmarks and three coordinates per hand.
4. Uses zero-filled features when a corresponding hand is not detected.
5. Writes the completed output atomically to:
   `data/sls_two_hand_landmarks.csv`.

The script intentionally refuses to overwrite an existing output CSV. Move or
rename the existing file before regenerating it.

The two-hand CSV contains a `label` column followed by these feature groups:

```text
left_lm0_x, left_lm0_y, left_lm0_z, ... left_lm20_z,
right_lm0_x, right_lm0_y, right_lm0_z, ... right_lm20_z
```

### Member-aware one-hand landmarks

For person-independent evaluation, run:

```bash
python src/extract_memberwise_landmarks.py
```

This produces `data/sls_hand_landmarks_memberwise.csv` with image name,
member ID, label, and 63 one-hand landmark features.

## Train the one-hand models

The one-hand trainer searches for a dataset in `data/` by default:

```bash
python src/train_sls_models.py
```

To use a specific dataset:

```bash
python src/train_sls_models.py \
  --data-path data/sls_hand_landmarks_clean.csv
```

To reduce runtime by skipping `GridSearchCV`:

```bash
python src/train_sls_models.py --no-tuning
```

To change the held-out test ratio:

```bash
python src/train_sls_models.py --test-size 0.25
```

The one-hand trainer:

- Filters labels to single-letter `A`-`Z` classes.
- Performs an 80/20 stratified train/test split by default.
- Uses five-fold stratified cross-validation during tuning.
- Compares SVM, Random Forest, and MLP pipelines.
- Selects the best model by test macro F1, then test accuracy.
- Saves the fitted pipeline to `models/sls_best_model.joblib`.
- Saves metadata to `models/sls_model_metadata.json`.
- Writes reports and plots under `results/`.

The checked-in one-hand metadata records an SVM model with 63 features and a
test accuracy of approximately 95.81%. Re-training can produce different
results if the data or dependency versions change.

## Train the two-hand models

First generate `data/sls_two_hand_landmarks.csv` if it is not already present,
then run:

```bash
python src/train_sls_two_hand_models.py
```

Useful options:

```bash
python src/train_sls_two_hand_models.py --no-tuning
python src/train_sls_two_hand_models.py --test-size 0.25
python src/train_sls_two_hand_models.py \
  --data-path data/sls_two_hand_landmarks.csv
```

The two-hand trainer uses 126 features and writes:

```text
models/sls_two_hand_best_model.joblib
models/sls_two_hand_model_metadata.json
results_two_hand/sls_model_comparison.csv
results_two_hand/sls_classification_reports.txt
results_two_hand/sls_confusion_matrices.png
results_two_hand/sls_best_model_confusion_matrix.png
```

The checked-in two-hand metadata records a Random Forest model with 126
features and a test accuracy of approximately 94.97%.

## Evaluate the one-hand model

Evaluate the default one-hand model on the default clean CSV:

```bash
python src/evaluate_sls_model.py
```

Use a custom model and evaluation dataset:

```bash
python src/evaluate_sls_model.py \
  --model-path models/sls_best_model.joblib \
  --data-path data/sls_hand_landmarks_clean.csv
```

The evaluator loads the serialized pipeline, filters the evaluation labels to
`A`-`Z`, prints overall accuracy, and prints a complete classification report.

## Additional evaluation experiments

The `member3_*.py` scripts contain focused experiments for robustness and
generalization:

```bash
python src/member3_evaluation.py
python src/member3_improvement.py
python src/member3_normalized_improvement.py
python src/member3_person_independent_evaluation.py
python src/member3_rotation_normalization_evaluation.py
```

These scripts use the existing files in `data/` and `models/`, and write
comparison tables, per-class metrics, predictions, reports, and confusion
matrices into `results/`. In particular, the person-independent evaluation
uses leave-one-member-out folds for the configured members:
`Member_1`, `Member_2`, `Member_3`, and `Member_4`.

## Use the model from Python

The serialized model is a scikit-learn pipeline and can be loaded with Joblib:

```python
import joblib
import numpy as np

model = joblib.load("models/sls_best_model.joblib")

# One sample: 21 landmarks x (x, y, z) = 63 values.
landmarks = np.zeros((1, 63), dtype=np.float32)
prediction = model.predict(landmarks)

print(prediction[0])
```

For a two-hand model, provide 126 values in the same order used by
`src/sls_two_hand_webcam.py`:

```python
model = joblib.load("models/sls_two_hand_best_model.joblib")
prediction = model.predict(np.zeros((1, 126), dtype=np.float32))
```

The input feature order is part of the model contract. Do not reorder
landmarks or coordinates between extraction and inference.

## Jupyter notebook

Launch Jupyter from the repository root:

```bash
jupyter notebook
```

Open `notebooks/02_ml_model_development.ipynb`. The notebook demonstrates the
one-hand model development workflow, including data loading, model training,
evaluation, and Joblib serialization.

## Reproducibility

The training scripts use:

- `random_state=42`
- An 80/20 stratified train/test split by default
- Five-fold stratified cross-validation
- Standardization inside the SVM and MLP pipelines to avoid data leakage

For comparable results, use the same dataset, Python version, and compatible
package versions. Existing `.joblib` files should only be loaded from trusted
sources because Joblib uses Python object serialization.

## Troubleshooting

### `Dataset file not found`

Run commands from the repository root and confirm that the expected CSV exists:

```bash
ls data/*.csv
```

For a custom location, pass `--data-path` to the relevant training script.

### `hand_landmarker.task` is missing

Run `python src/extract_sls_landmarks.py` to trigger the automatic download, or
download the asset manually into `notebooks/hand_landmarker.task`.

### The webcam cannot be opened

Check that:

- A camera is connected and not being used by another application.
- The terminal or IDE has camera permission.
- The default camera index (`0`) is correct.

The camera index is configured in `cv2.VideoCapture(0)` in
`src/sls_two_hand_webcam.py`.

### The extractor says the output already exists

`src/extract_sls_landmarks.py` does not overwrite an existing
`data/sls_two_hand_landmarks.csv`. Back up or remove that specific file before
running extraction again.

### Training is slow

Grid search can be computationally expensive. Use:

```bash
python src/train_sls_models.py --no-tuning
python src/train_sls_two_hand_models.py --no-tuning
```

You can also train with a smaller dataset while developing and reserve the
full dataset for the final run.

## Limitations

- Predictions are limited to the 26 letter classes represented in the
  training data.
- The webcam application is an inference demo, not a continuous sign-language
  translation system.
- Recognition quality depends on lighting, camera angle, hand visibility,
  occlusion, background, and similarity to the training data.
- The model does not infer words, sentences, motion-based signs, facial
  expressions, or grammatical context.
- A model trained on one landmark ordering must be used with the same ordering
  during inference.

## Contributing

1. Create a branch for your change.
2. Keep generated datasets, models, and result files clearly identified.
3. Run the relevant extraction, training, or evaluation command.
4. Include the command and environment details when reporting new metrics.
5. Keep changes focused and document new scripts or artifacts in this README.

## License

No license file is currently included in the repository. Add an explicit
license before distributing the project or its dataset and model artifacts.