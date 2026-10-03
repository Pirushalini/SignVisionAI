"""
=============================================================================
SignVisionAI - Member 2: Model Evaluation & Inference Verification Script
=============================================================================
Author: Member 2 (ML Model Development)
Description:
    Loads the trained and exported Pipeline (from models/sls_best_model.joblib)
    and validates its performance on test data, or performs single/batch
    predictions from 63 landmark coordinates (lm0_x ... lm20_z).
=============================================================================
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DEFAULT_MODEL_PATH = os.path.join("models", "sls_best_model.joblib")
DEFAULT_METADATA_PATH = os.path.join("models", "sls_model_metadata.json")
DEFAULT_DATA_PATH = os.path.join("data", "sls_hand_landmarks_clean.csv")


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Loads the serialized scikit-learn pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please run src/train_sls_models.py first.")
    
    print(f"Loading model pipeline from: {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def evaluate_on_dataset(model, data_path=DEFAULT_DATA_PATH):
    """Evaluates the loaded model against a clean landmark CSV dataset."""
    if not os.path.exists(data_path):
        # Check fallback
        fallback = os.path.join("data", "asl_hand_landmarks (1).csv")
        if os.path.exists(fallback):
            data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found at: {data_path}")

    print(f"Loading evaluation data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Identify label column
    if "label" in df.columns:
        target_col = "label"
    elif "Label" in df.columns:
        target_col = "Label"
    else:
        target_col = df.columns[-1] if df.dtypes.iloc[-1] == object else df.columns[0]

    df[target_col] = df[target_col].astype(str).str.strip().str.upper()
    valid_mask = df[target_col].str.match(r'^[A-Z]$')
    df = df[valid_mask].copy()

    X = df.drop(columns=[target_col]).astype(np.float32)
    y = df[target_col]

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print("\n" + "=" * 60)
    print(f" Overall Accuracy on Evaluated Data: {acc * 100:.2f}% ({len(df)} samples)")
    print("=" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred, zero_division=0))


def predict_landmarks(model, landmarks_array):
    """
    Predicts sign class for given 63 landmark coordinates.
    landmarks_array: list or numpy array of shape (63,) or (N, 63).
    """
    landmarks_array = np.array(landmarks_array, dtype=np.float32)
    if landmarks_array.ndim == 1:
        landmarks_array = landmarks_array.reshape(1, -1)
    
    predictions = model.predict(landmarks_array)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(landmarks_array)
        
    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description="Evaluate or infer using trained SLS sign model.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to joblib model")
    parser.add_argument("--data-path", type=str, default=None, help="Path to test CSV")
    args = parser.parse_args()

    model = load_model(args.model_path)
    
    if os.path.exists(DEFAULT_METADATA_PATH):
        with open(DEFAULT_METADATA_PATH, "r") as f:
            meta = json.load(f)
            print(f"Model Name: {meta.get('model_name')}")
            print(f"Classes   : {meta.get('classes')}")
            print(f"Pipeline  : {meta.get('pipeline_steps')}")
            print(f"Test Acc  : {meta.get('test_metrics', {}).get('accuracy', 0)*100:.2f}%\n")

    eval_data = args.data_path if args.data_path else DEFAULT_DATA_PATH
    evaluate_on_dataset(model, eval_data)


if __name__ == "__main__":
    main()
