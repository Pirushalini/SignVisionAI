# SignVisionAI

**SignVisionAI** is a computer vision and machine learning project for recognizing **American Sign Language (ASL) alphabet signs** from hand images and webcam input.

The system uses **MediaPipe Hand Landmarker** to extract hand landmarks and machine learning models to classify the extracted hand features into ASL alphabet classes.

---

## Project Objective

The main objective of SignVisionAI is to develop a system that can recognize static ASL hand signs and predict the corresponding alphabet letter.

The current project focuses on **static ASL alphabet recognition** as the first stage toward a broader sign language communication system.

---

## System Overview

The overall pipeline is:

```text
ASL Image / Webcam
        ↓
MediaPipe Hand Landmarker
        ↓
21 Hand Landmarks
        ↓
63 Numerical Features
        ↓
Wrist-relative Normalization
        ↓
Machine Learning Model
        ↓
ASL Letter Prediction
```

---

## Key Features

* ASL alphabet recognition
* Hand detection using MediaPipe
* Extraction of 21 hand landmarks
* 63 numerical hand-coordinate features
* Wrist-relative landmark normalization
* Multiple machine learning models for comparison
* MLP-based final classification model
* Model evaluation using accuracy and F1-score
* Stratified 5-fold cross-validation
* Saved trained model using Joblib
* Initial real-time webcam prediction
* Prediction smoothing using recent-frame majority voting

---

## Dataset

The project uses an ASL alphabet image dataset.

After processing the images with MediaPipe, the extracted hand landmarks are stored in a CSV file.

### Dataset Statistics

| Property           |          Value |
| ------------------ | -------------: |
| Total samples      |         10,615 |
| Features           |             63 |
| Classes            |             28 |
| Alphabet classes   |            A–Z |
| Additional classes | `del`, `space` |

The extracted dataset is available as:

```text
data/asl_hand_landmarks.csv
```

### Feature Representation

MediaPipe detects **21 hand landmarks**.

Each landmark contains:

* X coordinate
* Y coordinate
* Z coordinate

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

The coordinates are normalized relative to the wrist to reduce the effect of the hand's position in the image.

---

## Machine Learning Models

Several classification algorithms were evaluated using the extracted landmark features:

1. Random Forest
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Logistic Regression
5. Multi-Layer Perceptron (MLP)

### Test Results

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Random Forest       |        97.13% |
| SVM                 |        95.90% |
| KNN                 |        94.77% |
| Logistic Regression |        97.60% |
| **MLP**             |    **97.69%** |

Based on the experiments, **MLP achieved the highest test accuracy** among the evaluated models.

---

## Final Model

The final classification pipeline uses:

```text
StandardScaler
      ↓
MLPClassifier
      ↓
Hidden Layer: 128 neurons
      ↓
Hidden Layer: 64 neurons
      ↓
28-Class Prediction
```

The trained model is saved as:

```text
models/sign_language_mlp.pkl
```

Joblib is used to save and load the trained model.

---

## Cross-Validation

To evaluate the consistency of the model across different data splits, **Stratified 5-Fold Cross-Validation** was performed.

### MLP Cross-Validation Results

* Mean Accuracy: **97.62%**
* Standard Deviation: **0.41%**
* Mean Macro F1: **97.33%**

The cross-validation results provide an additional evaluation beyond the single held-out test set.

> **Note:** The 97.69% result represents the offline held-out test accuracy on the extracted landmark dataset. It should not be interpreted as the final real-world webcam accuracy. Real-world performance requires separate testing under webcam conditions.

---

## Real-Time Webcam Prediction

The project includes an initial webcam prediction component using:

* OpenCV
* MediaPipe
* NumPy
* Joblib
* Trained MLP model

The webcam pipeline is:

```text
Webcam Frame
     ↓
OpenCV
     ↓
MediaPipe Hand Detection
     ↓
21 Hand Landmarks
     ↓
Landmark Normalization
     ↓
63 Features
     ↓
Trained MLP
     ↓
Predicted ASL Letter
```

To reduce frame-to-frame prediction fluctuations, the system maintains recent predictions and uses **majority voting** to produce a more stable displayed prediction.

The webcam implementation is currently being validated as part of the end-to-end real-time system.

---

## Project Structure

```text
SignVisionAI/
│
├── data/
│   └── asl_hand_landmarks.csv
│
├── models/
│   ├── sign_language_rf.pkl
│   └── sign_language_mlp.pkl
│
├── notebooks/
│   └── feature_extraction.ipynb
│
├── src/
│   ├── train_model.py
│   ├── compare_models.py
│   ├── cross_validate.py
│   ├── final_train.py
│   ├── evaluate_model.py
│   └── webcam_predict.py
│
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming Language

* Python

### Computer Vision

* OpenCV
* MediaPipe Hand Landmarker

### Machine Learning

* Scikit-learn
* Random Forest
* SVM
* KNN
* Logistic Regression
* MLP

### Data Processing

* NumPy
* Pandas

### Model Persistence

* Joblib

---

## Environment

The project was developed and tested using Python 3.13.

Main packages include:

```text
Python 3.13
OpenCV
MediaPipe
NumPy
Pandas
Scikit-learn
Joblib
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Pirushalini/SignVisionAI.git
cd SignVisionAI
```

Create a virtual environment:

```bash
python3.13 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the Random Forest model:

```bash
python src/train_model.py
```

To compare the machine learning models:

```bash
python src/compare_models.py
```

To perform cross-validation:

```bash
python src/cross_validate.py
```

To train the final MLP model:

```bash
python src/final_train.py
```

---

## Model Evaluation

The project includes evaluation scripts for checking model performance using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification reports
* Confusion matrix
* Cross-validation

Example:

```bash
python src/evaluate_model.py
```

---

## Webcam Prediction

The initial webcam prediction script can be executed using:

```bash
python src/webcam_predict.py
```

The webcam captures frames, detects the hand using MediaPipe, extracts and normalizes the landmarks, and passes the resulting features to the trained MLP model.

---

## Current Project Status

### Completed

* [x] ASL dataset investigation
* [x] Hand landmark extraction
* [x] 21-landmark representation
* [x] 63-feature generation
* [x] Wrist-relative normalization
* [x] Landmark CSV dataset creation
* [x] Multiple model training
* [x] Model comparison
* [x] Model evaluation
* [x] Confusion matrix analysis
* [x] Stratified 5-fold cross-validation
* [x] Final MLP model training
* [x] Trained model saving
* [x] Initial webcam prediction implementation

### In Progress

* [ ] End-to-end real-time webcam validation
* [ ] Real-world testing with different users and environments
* [ ] Performance analysis under different lighting and backgrounds
* [ ] Further improvement of difficult/ambiguous classes

---

## Future Scope

The current system focuses on **static ASL alphabet recognition**.

Future development can extend the system toward:

```text
ASL Alphabet
     ↓
Static Word Recognition
     ↓
Dynamic Sign Recognition
     ↓
Sentence-Level Communication
```

For dynamic sign recognition, temporal machine learning models such as **LSTM or GRU** can be explored because they can process sequences of hand movements over time.

Other possible improvements include:

* Larger and more diverse datasets
* Multi-user evaluation
* Improved real-time interface
* Confidence-based predictions
* Better handling of similar hand signs
* Word and phrase recognition
* Dynamic gesture recognition
* Text and speech output

---

## Limitations

The current version has several limitations:

* The main focus is static ASL alphabet recognition.
* The offline test dataset may not fully represent real-world webcam conditions.
* Lighting, background, camera quality, and hand orientation can affect detection.
* Some visually similar signs can be difficult to classify.
* Real-world webcam accuracy requires further evaluation.
* Full word and sentence-level sign language communication is not yet implemented.

---

## Team Project

**Project:** SignVisionAI
**Domain:** Computer Vision & Machine Learning
**Application:** American Sign Language Recognition

The project is being developed as a team-based project with separate responsibilities covering:

* Existing hand detection / system development
* Dataset and feature extraction
* Machine learning model training and evaluation

---

## Conclusion

SignVisionAI establishes a machine-learning pipeline for recognizing static ASL alphabet signs using hand landmarks rather than raw image pixels.

By combining **MediaPipe Hand Landmarker**, landmark normalization, multiple machine-learning classifiers, model evaluation, cross-validation, and webcam inference, the project provides a foundation for developing a more advanced sign language communication system in future stages.
