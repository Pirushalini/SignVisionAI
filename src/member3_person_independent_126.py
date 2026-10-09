"""
SignVisionAI - Member 3
Person-Independent Evaluation - 126 Features

Tests whether the 126-feature SVM generalizes to
people whose images were NOT used during training.

For each member:

    Train = other 3 members
    Test  = held-out member

Feature order:

    left hand  = 63
    right hand = 63
    total      = 126
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path(
    "data/sls_hand_landmarks_memberwise.csv"
)

RESULTS_DIR = Path("results")

RANDOM_STATE = 42

EXPECTED_FEATURE_COUNT = 126


# ============================================================
# Load dataset
# ============================================================

print("=" * 70)
print("Member 3 - 126 Feature Person-Independent Evaluation")
print("=" * 70)

print()
print(
    f"Loading dataset: {DATA_PATH}"
)

df = pd.read_csv(
    DATA_PATH
)


# ============================================================
# Validate columns
# ============================================================

metadata_columns = [
    "image_name",
    "member_id",
    "label",
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print()
print(
    f"Dataset shape : {df.shape}"
)

print(
    f"Feature count : {len(feature_columns)}"
)

print(
    f"Classes       : {df['label'].nunique()}"
)

print(
    f"Members       : "
    f"{sorted(df['member_id'].unique())}"
)


if len(feature_columns) != EXPECTED_FEATURE_COUNT:

    raise ValueError(
        "Expected 126 features, but found "
        f"{len(feature_columns)}."
    )


# ============================================================
# Basic validation
# ============================================================

if df[feature_columns].isnull().any().any():

    raise ValueError(
        "Missing values detected in feature columns."
    )


print()
print("Feature validation passed.")


# ============================================================
# Results
# ============================================================

fold_results = []

all_predictions = []


# ============================================================
# Person-independent evaluation
# ============================================================

members = sorted(
    df["member_id"].unique()
)


for test_member in members:

    print()
    print("=" * 70)

    print(
        f"TEST MEMBER: {test_member}"
    )

    print("=" * 70)


    train_df = df[
        df["member_id"] != test_member
    ].copy()

    test_df = df[
        df["member_id"] == test_member
    ].copy()


    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        "label"
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "label"
    ]


    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        f"Training members  : "
        f"{sorted(train_df['member_id'].unique())}"
    )

    print(
        f"Classes in test   : "
        f"{y_test.nunique()}/26"
    )


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    print()
    print(
        "Training 126-feature SVM..."
    )


    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                C=50.0,
                gamma=0.01,
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )
        ),
    ])


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print()
    print(
        f"Accuracy       : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{weighted_f1:.4f}"
    )


    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )


    # --------------------------------------------------------
    # Store fold results
    # --------------------------------------------------------

    fold_results.append({
        "test_member": test_member,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })


    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    for image_name, true_label, predicted_label in zip(
        test_df["image_name"],
        y_test,
        predictions
    ):

        all_predictions.append({
            "test_member": test_member,
            "image_name": image_name,
            "true_label": true_label,
            "predicted_label": predicted_label,
        })


    # --------------------------------------------------------
    # Save per-member classification report
    # --------------------------------------------------------

    report_path = RESULTS_DIR / (
        f"member3_126_{test_member.lower()}_report.txt"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )


# ============================================================
# Overall results
# ============================================================

results_df = pd.DataFrame(
    fold_results
)

predictions_df = pd.DataFrame(
    all_predictions
)


print()
print("=" * 70)
print("OVERALL 126-FEATURE PERSON-INDEPENDENT RESULTS")
print("=" * 70)

print()

print(
    f"Average Accuracy       : "
    f"{results_df['accuracy'].mean():.4f} "
    f"({results_df['accuracy'].mean() * 100:.2f}%)"
)

print(
    f"Average Macro Precision: "
    f"{results_df['macro_precision'].mean():.4f}"
)

print(
    f"Average Macro Recall   : "
    f"{results_df['macro_recall'].mean():.4f}"
)

print(
    f"Average Macro F1       : "
    f"{results_df['macro_f1'].mean():.4f}"
)

print(
    f"Average Weighted F1    : "
    f"{results_df['weighted_f1'].mean():.4f}"
)


# ============================================================
# Save fold results
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_path = (
    RESULTS_DIR
    / "member3_person_independent_126_results.csv"
)

predictions_path = (
    RESULTS_DIR
    / "member3_person_independent_126_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# Overall confusion matrix
# ============================================================

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cm = confusion_matrix(
    predictions_df["true_label"],
    predictions_df["predicted_label"],
    labels=labels
)


cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_path = (
    RESULTS_DIR
    / "member3_person_independent_126_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ============================================================
# Finish
# ============================================================

print()
print("Files saved:")

print(
    f"  {results_path}"
)

print(
    f"  {predictions_path}"
)

print(
    f"  {cm_path}"
)

print()
print("=" * 70)
print("Evaluation completed.")
print("=" * 70)