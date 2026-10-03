import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATA_PATH = "data/sls_hand_landmarks_clean.csv"
BASELINE_MODEL_PATH = "models/sls_improved_model.joblib"

RESULTS_DIR = "results"
MODELS_DIR = "models"

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "member3_normalization_comparison.csv"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "member3_normalization_report.txt"
)

PER_CLASS_PATH = os.path.join(
    RESULTS_DIR,
    "member3_normalization_per_class.csv"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "sls_normalized_svm.joblib"
)


def normalize_landmarks(X):
    """
    Normalize each hand independently.

    Steps:
    1. Reshape 63 features into 21 landmarks x 3 coordinates.
    2. Move wrist (landmark 0) to the origin.
    3. Scale the hand using the maximum distance from the wrist.
    """

    X_array = np.asarray(
        X,
        dtype=np.float32
    )

    normalized_samples = []

    for sample in X_array:

        points = sample.reshape(21, 3)

        # Wrist becomes origin
        points = points - points[0]

        # Hand-size normalization
        distances = np.linalg.norm(
            points,
            axis=1
        )

        scale = np.max(distances)

        if scale > 0:
            points = points / scale

        normalized_samples.append(
            points.flatten()
        )

    return np.asarray(
        normalized_samples,
        dtype=np.float32
    )


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("Member 3 - Landmark Normalization Improvement")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["label"]
    )

    y = df["label"]

    print("\nDataset:")
    print(f"Samples  : {len(df)}")
    print(f"Features : {X.shape[1]}")
    print(f"Classes  : {y.nunique()}")

    # --------------------------------------------------
    # Same split used throughout the project
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nSplit:")
    print(f"Train : {len(X_train)}")
    print(f"Test  : {len(X_test)}")

    # --------------------------------------------------
    # Evaluate current improved SVM
    # --------------------------------------------------

    print("\n[1] Current class-balanced SVM")

    current_model = joblib.load(
        BASELINE_MODEL_PATH
    )

    current_pred = current_model.predict(
        X_test
    )

    current_metrics = metrics(
        y_test,
        current_pred
    )

    print(
        f"Accuracy : "
        f"{current_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1 : "
        f"{current_metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------
    # Normalize train/test landmarks
    # --------------------------------------------------

    print("\n[2] Normalizing landmarks...")

    X_train_norm = normalize_landmarks(
        X_train
    )

    X_test_norm = normalize_landmarks(
        X_test
    )

    print(
        f"Normalized feature shape: "
        f"{X_train_norm.shape}"
    )

    # --------------------------------------------------
    # Class-balanced normalized SVM
    # --------------------------------------------------

    print(
        "\n[3] Training normalized + class-balanced SVM..."
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ])

    param_grid = {
        "classifier__C": [
            10,
            50,
            100
        ],
        "classifier__gamma": [
            "scale",
            0.01,
            0.1
        ]
    }

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(
        X_train_norm,
        y_train
    )

    normalized_model = search.best_estimator_

    print("\nBest parameters:")
    print(
        search.best_params_
    )

    print(
        f"Best CV Macro F1: "
        f"{search.best_score_:.4f}"
    )

    # --------------------------------------------------
    # Evaluate normalized model
    # --------------------------------------------------

    print(
        "\n[4] Evaluating normalized model..."
    )

    normalized_pred = normalized_model.predict(
        X_test_norm
    )

    normalized_metrics = metrics(
        y_test,
        normalized_pred
    )

    print(
        f"Accuracy : "
        f"{normalized_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision : "
        f"{normalized_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall : "
        f"{normalized_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{normalized_metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{normalized_metrics['weighted_f1']:.4f}"
    )

    # --------------------------------------------------
    # Overall comparison
    # --------------------------------------------------

    comparison = pd.DataFrame([
        {
            "Model": "Class-Balanced SVM",
            **current_metrics
        },
        {
            "Model": "Normalized + Class-Balanced SVM",
            **normalized_metrics
        }
    ])

    comparison.to_csv(
        COMPARISON_PATH,
        index=False
    )

    # --------------------------------------------------
    # Per-class comparison
    # --------------------------------------------------

    current_report = classification_report(
        y_test,
        current_pred,
        output_dict=True,
        zero_division=0
    )

    normalized_report = classification_report(
        y_test,
        normalized_pred,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for label in sorted(y.unique()):

        old_f1 = current_report[
            label
        ]["f1-score"]

        new_f1 = normalized_report[
            label
        ]["f1-score"]

        rows.append({
            "label": label,
            "current_f1": old_f1,
            "normalized_f1": new_f1,
            "f1_change": new_f1 - old_f1
        })

    per_class = pd.DataFrame(
        rows
    ).sort_values(
        "f1_change",
        ascending=False
    )

    per_class.to_csv(
        PER_CLASS_PATH,
        index=False
    )

    print(
        "\n[5] Per-class F1 changes"
    )

    print(
        per_class.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------

    with open(
        REPORT_PATH,
        "w"
    ) as file:

        file.write(
            "Member 3 - Landmark Normalization Improvement\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            "Current Class-Balanced SVM\n"
        )

        for key, value in current_metrics.items():
            file.write(
                f"{key}: {value:.6f}\n"
            )

        file.write(
            "\nNormalized + Class-Balanced SVM\n"
        )

        for key, value in normalized_metrics.items():
            file.write(
                f"{key}: {value:.6f}\n"
            )

        file.write(
            "\nBest Parameters\n"
        )

        file.write(
            str(
                search.best_params_
            )
        )

        file.write(
            "\n\nClassification Report\n"
        )

        file.write(
            classification_report(
                y_test,
                normalized_pred,
                zero_division=0
            )
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    joblib.dump(
        normalized_model,
        MODEL_PATH
    )

    print(
        "\nSaved model:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nNormalization improvement completed."
    )


if __name__ == "__main__":
    main()
