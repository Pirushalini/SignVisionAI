"""
=============================================================================
SignVisionAI - Member 2: ML Model Development for SLS Hand Landmark Recognition
=============================================================================
Author: Member 2 (ML Model Development)
Description:
    This script trains, tunes, compares, and evaluates Machine Learning models
    (Support Vector Machine, Random Forest, and Multi-Layer Perceptron) for
    recognizing A-Z sign language classes using MediaPipe hand landmark coordinates.

Pipeline Features:
    1. Dataset Loading & Validation (sls_hand_landmarks_clean.csv).
    2. Stratified Train/Test Split (reproducible with random_state=42).
    3. Scikit-learn Pipelines (with StandardScaler for SVM and MLP to prevent leakage).
    4. 5-Fold Stratified Cross-Validation & Hyperparameter Tuning on training data only.
    5. Comprehensive Model Evaluation (Accuracy, Precision, Recall, Macro/Weighted F1,
       Classification Report, Confusion Matrices).
    6. Model Selection & Export of the complete fitted Pipeline using Joblib.
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =============================================================================
# Configuration & Constants
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DEFAULT_DATA_PATHS = [
    os.path.join("data", "sls_hand_landmarks_clean.csv"),
    "sls_hand_landmarks_clean.csv",
    os.path.join("data", "asl_hand_landmarks (1).csv"),
    os.path.join("data", "asl_hand_landmarks.csv"),
    os.path.join("data", "sls_hand_landmarks.csv")
]

MODEL_SAVE_PATH = os.path.join("models", "sls_best_model.joblib")
METADATA_SAVE_PATH = os.path.join("models", "sls_model_metadata.json")
RESULTS_DIR = "results"
MODELS_DIR = "models"


# =============================================================================
# Step 1: Load and Prepare Dataset
# =============================================================================
def locate_dataset(provided_path=None):
    """
    Finds the dataset file from the provided path or default search paths.
    """
    if provided_path and os.path.exists(provided_path):
        return provided_path
    
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(
        f"Dataset file not found! Looked in: {DEFAULT_DATA_PATHS}. "
        "Please ensure 'sls_hand_landmarks_clean.csv' is placed in 'data/' or provide the path via --data-path."
    )


def load_and_validate_dataset(data_path):
    """
    Loads sls_hand_landmarks_clean.csv, validates landmark coordinates & target labels,
    and handles missing/invalid values without modifying the original file.
    
    Expected Landmark Format:
        Features: lm0_x, lm0_y, lm0_z, ... , lm20_x, lm20_y, lm20_z (63 columns)
        Target: label (A-Z classes)
    """
    print(f"\n[1/7] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f" -> Raw dataset shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    
    # Identify target column
    if "label" in df.columns:
        target_col = "label"
    elif "Label" in df.columns:
        target_col = "Label"
    elif df.columns[0] in [chr(c) for c in range(ord('A'), ord('Z')+1)]:
        target_col = df.columns[0]
    else:
        # Check if the first or last column is non-numeric (target label)
        if df.dtypes.iloc[0] == object:
            target_col = df.columns[0]
        else:
            target_col = df.columns[-1]

    # Clean and standardize label column
    df[target_col] = df[target_col].astype(str).str.strip().str.upper()
    
    # Filter out non-alphabetical labels if dataset contains noise (e.g. del, space, nothing)
    # Keeping A-Z sign classes
    valid_mask = df[target_col].str.match(r'^[A-Z]$')
    if not valid_mask.all():
        num_dropped = (~valid_mask).sum()
        print(f" -> Filtering out {num_dropped} non-A-Z samples for clean A-Z classification.")
        df = df[valid_mask].copy()

    # Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Verify feature columns
    # Expected 63 landmark coordinates (21 landmarks x 3: x, y, z)
    print(f" -> Number of landmark feature dimensions: {X.shape[1]}")
    print(f" -> Number of unique sign classes: {y.nunique()}")
    
    # Check and handle missing/invalid values
    missing_features = X.isnull().sum().sum()
    missing_labels = y.isnull().sum()
    if missing_features > 0 or missing_labels > 0:
        print(f" -> Warning: Found {missing_features} missing feature values and {missing_labels} missing labels.")
        print(" -> Dropping rows with missing values for training integrity...")
        valid_rows = X.notnull().all(axis=1) & y.notnull()
        X = X[valid_rows]
        y = y[valid_rows]
    else:
        print(" -> Dataset integrity check: No missing values found.")

    # Convert all feature columns to float
    X = X.astype(np.float32)

    # Display class distribution summary
    class_counts = y.value_counts().sort_index()
    print("\nClass Distribution Summary (A-Z):")
    for cls, count in class_counts.items():
        print(f"   Class '{cls}': {count} samples", end=" | " if (ord(cls) - ord('A') + 1) % 4 != 0 else "\n")
    print(f"\nTotal Clean Samples for Training & Evaluation: {len(X)}")

    return X, y


# =============================================================================
# Step 2: Stratified Train/Test Split
# =============================================================================
def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Splits the dataset into training and held-out testing sets using stratification.
    Stratification ensures balanced class representation in both splits.
    """
    print(f"\n[2/7] Splitting dataset (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%, Stratified)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f" -> Training set samples : {X_train.shape[0]} ({X_train.shape[0]/len(X):.1%})")
    print(f" -> Testing set samples  : {X_test.shape[0]} ({X_test.shape[0]/len(X):.1%})")
    print(f" -> Number of features   : {X_train.shape[1]}")
    print(f" -> Classes represented  : {y_train.nunique()} in train, {y_test.nunique()} in test")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Step 3: Model Pipelines & Hyperparameter Search Spaces
# =============================================================================
def build_model_definitions():
    """
    Defines baseline pipelines and hyperparameter tuning grids for:
      1. Support Vector Machine (SVM)
      2. Random Forest (RF)
      3. Multi-Layer Perceptron (MLP)
      
    StandardScaler is embedded inside Pipelines to prevent data leakage.
    """
    # 1. Support Vector Classifier Pipeline
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=RANDOM_STATE, probability=True))
    ])
    svm_param_grid = {
        'classifier__C': [0.1, 1.0, 10.0, 50.0],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__kernel': ['rbf', 'linear']
    }

    # 2. Random Forest Classifier Pipeline
    rf_pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    rf_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 15, 25],
        'classifier__min_samples_split': [2, 5],
        'classifier__criterion': ['gini', 'entropy']
    }

    # 3. Multi-Layer Perceptron (Neural Network) Pipeline
    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_STATE
        ))
    ])
    mlp_param_grid = {
        'classifier__hidden_layer_sizes': [(128, 64), (128, 64, 32), (64, 32)],
        'classifier__activation': ['relu', 'tanh'],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate_init': [0.001, 0.01]
    }

    models_config = {
        "Support Vector Machine (SVM)": {
            "pipeline": svm_pipeline,
            "param_grid": svm_param_grid
        },
        "Random Forest": {
            "pipeline": rf_pipeline,
            "param_grid": rf_param_grid
        },
        "Multi-Layer Perceptron (MLP)": {
            "pipeline": mlp_pipeline,
            "param_grid": mlp_param_grid
        }
    }
    
    return models_config


# =============================================================================
# Step 4 & 5: Hyperparameter Tuning & Cross-Validation (Training Set Only)
# =============================================================================
def tune_and_evaluate_models(models_config, X_train, y_train, X_test, y_test, run_tuning=True):
    """
    Performs 5-Fold Stratified Cross-Validation and Hyperparameter Tuning using
    GridSearchCV strictly on the training set. Then evaluates best estimators on
    the held-out test set.
    """
    print("\n[3/7] Training & Hyperparameter Tuning via 5-Fold Stratified CV on Training Data...")
    print("=" * 80)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    trained_models = {}
    comparison_records = []
    test_predictions = {}
    
    for name, config in models_config.items():
        print(f"\n>>> Model: {name}")
        print("-" * 50)
        
        pipeline = config["pipeline"]
        param_grid = config["param_grid"] if run_tuning else {}
        
        if run_tuning:
            print(f" -> Running GridSearchCV with {CV_FOLDS}-fold CV across parameter grid...")
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            print(f" -> Best 5-Fold CV Macro F1: {best_cv_score:.4f}")
            print(f" -> Optimal Hyperparameters:")
            for p, val in best_params.items():
                print(f"      {p}: {val}")
        else:
            print(" -> Fitting baseline pipeline on training data...")
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = "Default"
            best_cv_score = np.nan
        
        # Evaluate on Held-out Test Set
        y_pred = best_model.predict(X_test)
        test_predictions[name] = y_pred
        trained_models[name] = {
            "model": best_model,
            "best_params": best_params,
            "cv_score": best_cv_score
        }
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        print(f"\n [Held-out Test Performance - {name}]")
        print(f"   Accuracy        : {acc * 100:.2f}%")
        print(f"   Macro Precision : {prec_macro:.4f}")
        print(f"   Macro Recall    : {rec_macro:.4f}")
        print(f"   Macro F1-Score  : {f1_macro:.4f}")
        print(f"   Weighted F1     : {f1_weighted:.4f}")
        
        comparison_records.append({
            "Model": name,
            "CV Macro F1": best_cv_score,
            "Test Accuracy": acc,
            "Test Macro Precision": prec_macro,
            "Test Macro Recall": rec_macro,
            "Test Macro F1": f1_macro,
            "Test Weighted F1": f1_weighted
        })
    
    comparison_df = pd.DataFrame(comparison_records)
    return trained_models, comparison_df, test_predictions


# =============================================================================
# Step 6: Detailed Evaluation & Visualization
# =============================================================================
def generate_evaluation_reports(trained_models, comparison_df, test_predictions, y_test, results_dir=RESULTS_DIR):
    """
    Generates and saves:
      1. Model comparison table (CSV).
      2. Detailed classification reports.
      3. Confusion matrices for all models and for the best model.
    """
    os.makedirs(results_dir, exist_ok=True)
    print("\n[4/7] Generating Comparative Evaluation Reports & Visualizations...")
    print("=" * 80)
    
    # 1. Print and Save Comparison Table
    print("\n================================================================================")
    print("                     MODEL PERFORMANCE COMPARISON TABLE                         ")
    print("================================================================================")
    formatted_table = comparison_df.copy()
    formatted_table["Test Accuracy"] = formatted_table["Test Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    formatted_table["CV Macro F1"] = formatted_table["CV Macro F1"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    formatted_table["Test Macro Precision"] = formatted_table["Test Macro Precision"].apply(lambda x: f"{x:.4f}")
    formatted_table["Test Macro Recall"] = formatted_table["Test Macro Recall"].apply(lambda x: f"{x:.4f}")
    formatted_table["Test Macro F1"] = formatted_table["Test Macro F1"].apply(lambda x: f"{x:.4f}")
    formatted_table["Test Weighted F1"] = formatted_table["Test Weighted F1"].apply(lambda x: f"{x:.4f}")
    print(formatted_table.to_string(index=False))
    print("================================================================================")
    
    csv_path = os.path.join(results_dir, "sls_model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f" -> Saved comparison table to: {csv_path}")

    # 2. Print & Save Detailed Classification Reports
    report_file_path = os.path.join(results_dir, "sls_classification_reports.txt")
    with open(report_file_path, "w") as f:
        for name, y_pred in test_predictions.items():
            report = classification_report(y_test, y_pred, zero_division=0)
            print(f"\n--- Classification Report: {name} ---")
            print(report)
            f.write(f"================================================================\n")
            f.write(f" Classification Report: {name}\n")
            f.write(f"================================================================\n")
            f.write(report + "\n\n")
    print(f" -> Saved classification reports to: {report_file_path}")

    # 3. Plot Combined Confusion Matrices
    classes = sorted(y_test.unique())
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle("Sign Language SLS Hand Landmark Recognition - Model Comparison", fontsize=16, fontweight='bold')

    for ax, (name, y_pred) in zip(axes, test_predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical")
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        ax.set_title(f"{name}\nAcc: {acc*100:.2f}% | F1: {f1:.4f}", fontsize=12)
        ax.set_xlabel("Predicted Sign Class", fontsize=10)
        ax.set_ylabel("True Sign Class", fontsize=10)

    plt.tight_layout()
    cm_all_path = os.path.join(results_dir, "sls_confusion_matrices.png")
    plt.savefig(cm_all_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved combined confusion matrices to: {cm_all_path}")


# =============================================================================
# Step 7: Best Model Selection & Artifact Export
# =============================================================================
def select_and_save_best_model(trained_models, comparison_df, X_train, y_train, X_test, y_test, models_dir=MODELS_DIR, results_dir=RESULTS_DIR):
    """
    Selects the winning model based on Cross-Validation & Test Macro F1 scores,
    plots its individual high-resolution confusion matrix, and serializes the
    complete fitted Pipeline using joblib.
    """
    print("\n[5/7] Selecting Best Performing Model...")
    print("=" * 80)
    
    # Sort by Test Macro F1 then Test Accuracy
    best_row = comparison_df.sort_values(by=["Test Macro F1", "Test Accuracy"], ascending=False).iloc[0]
    best_name = best_row["Model"]
    best_info = trained_models[best_name]
    best_pipeline = best_info["model"]
    
    print(f" >>> Winning Model: {best_name}")
    print(f"     CV Macro F1  : {best_row['CV Macro F1']:.4f}")
    print(f"     Test Accuracy: {best_row['Test Accuracy']*100:.2f}%")
    print(f"     Test Macro F1: {best_row['Test Macro F1']:.4f}")
    
    # Plot dedicated high-res confusion matrix for best model
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    classes = sorted(y_test.unique())
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation="vertical")
    ax.set_title(f"Best Model Confusion Matrix: {best_name}\nAccuracy: {best_row['Test Accuracy']*100:.2f}% | Macro F1: {best_row['Test Macro F1']:.4f}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Ground Truth Class", fontsize=12)
    plt.tight_layout()
    best_cm_path = os.path.join(results_dir, "sls_best_model_confusion_matrix.png")
    plt.savefig(best_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved best model confusion matrix to: {best_cm_path}")

    # Save Pipeline using joblib
    print("\n[6/7] Exporting Trained Model Pipeline...")
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f" -> Saved complete fitted pipeline to: {MODEL_SAVE_PATH}")
    
    # Save Model Metadata & Configuration
    metadata = {
        "model_name": best_name,
        "classes": [str(c) for c in classes],
        "num_classes": len(classes),
        "feature_count": int(X_train.shape[1]),
        "best_hyperparameters": str(best_info["best_params"]),
        "test_metrics": {
            "accuracy": float(best_row["Test Accuracy"]),
            "macro_precision": float(best_row["Test Macro Precision"]),
            "macro_recall": float(best_row["Test Macro Recall"]),
            "macro_f1": float(best_row["Test Macro F1"]),
            "weighted_f1": float(best_row["Test Weighted F1"]),
            "cv_macro_f1": float(best_row["CV Macro F1"]) if pd.notnull(best_row["CV Macro F1"]) else None
        },
        "pipeline_steps": [name for name, _ in best_pipeline.steps]
    }
    
    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f" -> Saved model metadata to: {METADATA_SAVE_PATH}")

    # Verify model reloading
    print("\n[7/7] Verifying Saved Model Pipeline Reload & Inference...")
    loaded_pipeline = joblib.load(MODEL_SAVE_PATH)
    sample_input = X_test.iloc[0:1]
    sample_true_label = y_test.iloc[0]
    sample_pred_label = loaded_pipeline.predict(sample_input)[0]
    
    print(f" -> Verification Sample Prediction: True='{sample_true_label}', Predicted='{sample_pred_label}'")
    assert hasattr(loaded_pipeline, "predict"), "Loaded pipeline missing predict method!"
    print(" -> Verification Successful! The pipeline is ready for deployment and real-time inference.")
    
    return best_name, best_pipeline


# =============================================================================
# Main CLI Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Machine Learning models for SLS Sign Recognition (Member 2)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to sls_hand_landmarks_clean.csv (default: auto-detected in data/)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Held-out test split ratio (default: 0.20)"
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip GridSearchCV and use default baseline models (faster)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  SignVisionAI - Member 2: Machine Learning Model Development  ")
    print("=" * 80)
    
    # 1. Locate and Load Dataset
    data_path = locate_dataset(args.data_path)
    X, y = load_and_validate_dataset(data_path)
    
    # 2. Stratified Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size, random_state=RANDOM_STATE)
    
    # 3. Build Model Definitions
    models_config = build_model_definitions()
    
    # 4 & 5. Train, Tune & Evaluate
    trained_models, comparison_df, test_predictions = tune_and_evaluate_models(
        models_config,
        X_train,
        y_train,
        X_test,
        y_test,
        run_tuning=not args.no_tuning
    )
    
    # 6. Generate Reports & Visualizations
    generate_evaluation_reports(trained_models, comparison_df, test_predictions, y_test)
    
    # 7. Select & Save Best Model
    select_and_save_best_model(trained_models, comparison_df, X_train, y_train, X_test, y_test)
    
    print("\n" + "=" * 80)
    print(" ML Model Development Completed Successfully! ")
    print("=" * 80)


if __name__ == "__main__":
    main()
