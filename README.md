# SignVisionAI

SignVisionAI is a sign-language recognition project that uses **MediaPipe hand landmarks** and **scikit-learn machine learning models** to recognize ASL classes from hand-landmark features.

The current repository contains the feature-extraction notebook, prepared landmark dataset, machine-learning training scripts, model comparison results, evaluation results, cross-validation results, and the final trained MLP model.

## Project Structure

```text
SignVisionAI/
│
├── data/
│   └── asl_hand_landmarks.csv
│
├── models/
│   └── sign_language_mlp.pkl
│
├── notebooks/
│   └── 01_feature_extraction (1).ipynb
│
├── results/
│   ├── cross_validation_results.csv
│   ├── mlp_confusion_matrix.png
│   └── model_comparison.csv
│
└── src/
    ├── train_model.py
    ├── compare_models.py
    ├── cross_validate.py
    ├── evaluate_model.py
    └── final_train.py
```

## Requirements

* Python 3.10 or newer
* Jupyter Notebook (for feature extraction)
* MediaPipe
* Pandas
* Scikit-learn
* Joblib
* Matplotlib

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install pandas scikit-learn joblib matplotlib jupyter mediapipe
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset

The main training dataset is:

```text
data/asl_hand_landmarks.csv
```

The dataset contains:

* **10,615 samples**
* **63 hand-landmark features**
* **28 classes**
* No missing values
* No duplicate rows

The 63 features represent **21 MediaPipe hand landmarks**, with three coordinates for each landmark:

```text
x, y, z
```

The dataset contains the following classes:

```text
A-Z
del
space
```

The training scripts expect the feature columns followed by a target column named:

```text
label
```

## Feature Extraction

The feature-extraction notebook uses **MediaPipe Hand Landmarker** to detect a hand and extract 21 hand landmarks.

The landmarks are converted into 63 numerical features and stored in the CSV dataset.

To regenerate the feature dataset, open the notebook:

```bash
jupyter notebook "notebooks/01_feature_extraction (1).ipynb"
```

The generated feature CSV should be saved as:

```text
data/asl_hand_landmarks.csv
```

The original ASL image dataset is kept separately and is not included in this repository.

## Machine Learning Workflow

The project follows this workflow:

```text
ASL Images
    ↓
MediaPipe Hand Landmark Extraction
    ↓
63 Landmark Features
    ↓
asl_hand_landmarks.csv
    ↓
Model Training
    ↓
Model Comparison
    ↓
Evaluation
    ↓
5-Fold Cross-Validation
    ↓
Final MLP Model
    ↓
sign_language_mlp.pkl
```

## Model Comparison

Five machine-learning classifiers were compared using an **80/20 stratified train-test split**:

* Random Forest
* SVM
* KNN
* Logistic Regression
* MLP

Run:

```bash
python src/compare_models.py
```

The results are saved to:

```text
results/model_comparison.csv
```

### Comparison Results

| Model               |   Accuracy |
| ------------------- | ---------: |
| Random Forest       |     97.13% |
| SVM                 |     95.90% |
| KNN                 |     94.77% |
| Logistic Regression |     97.60% |
| MLP                 | **97.69%** |

The MLP achieved **97.69% accuracy** on the held-out test set.

## Five-Fold Cross-Validation

To evaluate model performance across multiple train-test splits, stratified **5-fold cross-validation** was performed for:

* Random Forest
* Logistic Regression
* MLP

Run:

```bash
python src/cross_validate.py
```

Results are saved to:

```text
results/cross_validation_results.csv
```

The MLP achieved:

```text
Mean Accuracy: 97.62%
Accuracy Std:  ±0.41%
Mean Macro F1: 97.33%
```

## MLP Evaluation

The MLP model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Run:

```bash
python src/evaluate_model.py
```

The confusion matrix is saved to:

```text
results/mlp_confusion_matrix.png
```

## Final Model

The final model is an MLP pipeline containing:

```text
StandardScaler
      ↓
MLPClassifier
      ↓
Hidden Layers: (128, 64)
```

The model is trained using the complete available feature dataset.

Run:

```bash
python src/final_train.py
```

The final trained pipeline is saved as:

```text
models/sign_language_mlp.pkl
```

## Loading the Final Model

The saved model can be loaded using Joblib:

```python
import joblib

model = joblib.load("models/sign_language_mlp.pkl")

prediction = model.predict([features_63])

print(prediction[0])
```

`features_63` must contain the same **63 hand-landmark features in the same order** as the training dataset.

## Baseline Random Forest

The Random Forest baseline can be trained using:

```bash
python src/train_model.py
```

The script creates:

```text
models/sign_language_rf.pkl
```

The Random Forest model was used as a baseline for model comparison. The generated `.pkl` file is **not included in the repository because of its large file size**.

## Output Files

The project produces the following outputs:

```text
results/
├── model_comparison.csv
├── cross_validation_results.csv
└── mlp_confusion_matrix.png
```

Final model:

```text
models/sign_language_mlp.pkl
```

## Current Status

Completed:

* [x] ASL landmark feature dataset preparation
* [x] MediaPipe landmark feature extraction
* [x] Random Forest training
* [x] SVM training
* [x] KNN training
* [x] Logistic Regression training
* [x] MLP training
* [x] Model comparison
* [x] Classification evaluation
* [x] Confusion matrix
* [x] Five-fold cross-validation
* [x] Final MLP model training
* [x] Model saving for integration

Not yet implemented:

* [ ] Live webcam inference application
* [ ] Real-time prediction interface
* [ ] Integration with the final application

## Important Notes

* Run all Python commands from the **repository root**.
* The training scripts expect `data/asl_hand_landmarks.csv`.
* The final MLP model expects 63 hand-landmark features in the same order used during training.
* The original ASL image dataset is maintained separately and is not included in the repository.
* The Random Forest `.pkl` model is not stored in the repository because of its large file size.
* Models saved with Joblib should only be loaded from trusted sources and with compatible Python and scikit-learn versions.

## Team Handover

The ML training and evaluation pipeline is ready for integration.

The final model is available at:

```text
models/sign_language_mlp.pkl
```

The integration/realtime component can use this model with the same 63-feature MediaPipe landmark format used during training.
