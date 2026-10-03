from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Paths
# ============================================================
DATA_PATH = Path("data/sls_hand_landmarks_memberwise.csv")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULTS_PATH = RESULTS_DIR / "member3_person_independent_fold_results.csv"
PER_CLASS_PATH = RESULTS_DIR / "member3_person_independent_per_class.csv"
PREDICTIONS_PATH = RESULTS_DIR / "member3_person_independent_predictions.csv"
REPORT_PATH = RESULTS_DIR / "member3_person_independent_report.txt"
CONFUSION_PATH = RESULTS_DIR / "member3_person_independent_confusion_matrix.png"


# ============================================================
# Configuration
# ============================================================
MEMBERS = [
    "Member_1",
    "Member_2",
    "Member_3",
    "Member_4",
]

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Selected earlier from the normalized + class-balanced SVM search
SVM_C = 100
SVM_GAMMA = 0.01


# ============================================================
# Landmark normalization
# ============================================================
def normalize_landmarks(X):
    """
    Normalize each hand independently.

    Steps:
    1. Reshape 63 features into 21 landmarks x 3 coordinates.
    2. Move wrist (landmark 0) to the origin.
    3. Scale the hand using the maximum distance from the wrist.
    """

    X_array = np.asarray(X, dtype=np.float32)

    normalized_samples = []

    for sample in X_array:
        points = sample.reshape(21, 3)

        # Move wrist to origin
        points = points - points[0]

        # Hand-size normalization
        distances = np.linalg.norm(points, axis=1)
        scale = np.max(distances)

        if scale > 0:
            points = points / scale

        normalized_samples.append(points.flatten())

    return np.asarray(normalized_samples, dtype=np.float32)


# ============================================================
# Model
# ============================================================
def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=SVM_C,
                gamma=SVM_GAMMA,
                class_weight="balanced",
            ),
        ),
    ])


# ============================================================
# Main
# ============================================================
def main():

    print("=" * 70)
    print("Member 3 - Person-Independent SLS Evaluation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")

    required_columns = {
        "image_name",
        "member_id",
        "label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("lm")
    ]

    feature_columns = sorted(
        feature_columns,
        key=lambda name: (
            int(name[2:name.index("_")]),
            name[name.index("_") + 1:]
        ),
    )

    if len(feature_columns) != 63:
        raise ValueError(
            f"Expected 63 landmark features, found {len(feature_columns)}"
        )

    print(f"Feature count: {len(feature_columns)}")
    print(f"Classes: {sorted(df['label'].unique())}")

    # --------------------------------------------------------
    # Prepare arrays
    # --------------------------------------------------------
    X = df[feature_columns].values
    y = df["label"].values
    member_ids = df["member_id"].values
    image_names = df["image_name"].values

    # --------------------------------------------------------
    # Four-member leave-one-member-out evaluation
    # --------------------------------------------------------
    fold_results = []
    per_class_results = []
    all_predictions = []

    for held_out_member in MEMBERS:

        print("\n" + "=" * 70)
        print(f"TEST MEMBER: {held_out_member}")
        print("=" * 70)

        train_mask = member_ids != held_out_member
        test_mask = member_ids == held_out_member

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        test_images = image_names[test_mask]

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples : {len(X_test)}")

        train_members = sorted(
            np.unique(member_ids[train_mask])
        )

        print(f"Training members: {train_members}")

        # Check class availability
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))

        missing_in_train = sorted(
            set(CLASSES) - train_classes
        )

        if missing_in_train:
            print(
                f"WARNING: Classes missing from training: "
                f"{missing_in_train}"
            )

        print(
            f"Classes present in test: "
            f"{len(test_classes)}/26"
        )

        # ----------------------------------------------------
        # Normalize independently
        # ----------------------------------------------------
        X_train_norm = normalize_landmarks(X_train)
        X_test_norm = normalize_landmarks(X_test)

        # ----------------------------------------------------
        # Fresh model for this fold
        # ----------------------------------------------------
        model = create_model()

        print(
            f"\nTraining normalized + class-balanced SVM "
            f"(C={SVM_C}, gamma={SVM_GAMMA})..."
        )

        model.fit(X_train_norm, y_train)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------
        y_pred = model.predict(X_test_norm)

        # ----------------------------------------------------
        # Metrics using only classes actually present in test
        # ----------------------------------------------------
        present_classes = sorted(
            np.unique(y_test)
        )

        fold_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        fold_macro_precision = precision_score(
            y_test,
            y_pred,
            labels=present_classes,
            average="macro",
            zero_division=0,
        )

        fold_macro_recall = recall_score(
            y_test,
            y_pred,
            labels=present_classes,
            average="macro",
            zero_division=0,
        )

        fold_macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=present_classes,
            average="macro",
            zero_division=0,
        )

        fold_weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        print(
            f"\nAccuracy       : {fold_accuracy * 100:.2f}%"
        )
        print(
            f"Macro Precision: {fold_macro_precision:.4f}"
        )
        print(
            f"Macro Recall   : {fold_macro_recall:.4f}"
        )
        print(
            f"Macro F1       : {fold_macro_f1:.4f}"
        )
        print(
            f"Weighted F1    : {fold_weighted_f1:.4f}"
        )

        fold_results.append({
            "held_out_member": held_out_member,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_classes": len(test_classes),
            "accuracy": fold_accuracy,
            "macro_precision": fold_macro_precision,
            "macro_recall": fold_macro_recall,
            "macro_f1": fold_macro_f1,
            "weighted_f1": fold_weighted_f1,
        })

        # ----------------------------------------------------
        # Per-class results
        # ----------------------------------------------------
        report = classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        test_support = pd.Series(y_test).value_counts()

        for label in CLASSES:
            per_class_results.append({
                "held_out_member": held_out_member,
                "label": label,
                "support": int(test_support.get(label, 0)),
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
            })

        # ----------------------------------------------------
        # Save fold predictions
        # ----------------------------------------------------
        for image_name, true_label, predicted_label in zip(
            test_images,
            y_test,
            y_pred
        ):
            all_predictions.append({
                "held_out_member": held_out_member,
                "image_name": image_name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
            })

    # ========================================================
    # Aggregate results
    # ========================================================
    fold_df = pd.DataFrame(fold_results)
    per_class_df = pd.DataFrame(per_class_results)
    predictions_df = pd.DataFrame(all_predictions)

    fold_df.to_csv(
        FOLD_RESULTS_PATH,
        index=False
    )

    per_class_df.to_csv(
        PER_CLASS_PATH,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Overall prediction metrics
    # --------------------------------------------------------
    y_true_all = predictions_df["true_label"]
    y_pred_all = predictions_df["predicted_label"]

    overall_accuracy = accuracy_score(
        y_true_all,
        y_pred_all
    )

    overall_macro_precision = precision_score(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    overall_macro_recall = recall_score(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    overall_macro_f1 = f1_score(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    overall_weighted_f1 = f1_score(
        y_true_all,
        y_pred_all,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("OVERALL PERSON-INDEPENDENT RESULTS")
    print("=" * 70)

    print(
        f"Accuracy       : {overall_accuracy * 100:.2f}%"
    )
    print(
        f"Macro Precision: {overall_macro_precision:.4f}"
    )
    print(
        f"Macro Recall   : {overall_macro_recall:.4f}"
    )
    print(
        f"Macro F1       : {overall_macro_f1:.4f}"
    )
    print(
        f"Weighted F1    : {overall_weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Fold average
    # --------------------------------------------------------
    print("\nFour-fold average:")
    print(
        f"Accuracy       : "
        f"{fold_df['accuracy'].mean() * 100:.2f}%"
    )
    print(
        f"Macro Precision: "
        f"{fold_df['macro_precision'].mean():.4f}"
    )
    print(
        f"Macro Recall   : "
        f"{fold_df['macro_recall'].mean():.4f}"
    )
    print(
        f"Macro F1       : "
        f"{fold_df['macro_f1'].mean():.4f}"
    )
    print(
        f"Weighted F1    : "
        f"{fold_df['weighted_f1'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------
    full_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
        target_names=CLASSES,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Misclassification pairs
    # --------------------------------------------------------
    errors = predictions_df[
        predictions_df["correct"] == False
    ]

    if len(errors) > 0:
        misclassification_pairs = (
            errors
            .groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    else:
        misclassification_pairs = pd.DataFrame(
            columns=[
                "true_label",
                "predicted_label",
                "count",
            ]
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------
    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=CLASSES,
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    image = ax.imshow(cm)

    ax.set_title(
        "Person-Independent SLS Confusion Matrix"
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)

    plt.colorbar(image, ax=ax)

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(
        CONFUSION_PATH,
        dpi=200,
    )
    plt.close()

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Member 3 - Person-Independent SLS Evaluation\n"
        )
        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            "Dataset\n"
        )
        file.write(
            f"Total successful landmark samples: {len(df)}\n"
        )
        file.write(
            f"Members: {', '.join(MEMBERS)}\n"
        )
        file.write(
            "Evaluation: Leave-one-member-out\n\n"
        )

        file.write(
            "Normalization\n"
        )
        file.write(
            "1. Wrist landmark moved to origin\n"
        )
        file.write(
            "2. Maximum wrist distance used for hand-size normalization\n"
        )
        file.write(
            "3. StandardScaler fitted only on training members\n\n"
        )

        file.write(
            "SVM Configuration\n"
        )
        file.write(
            "Kernel: RBF\n"
        )
        file.write(
            f"C: {SVM_C}\n"
        )
        file.write(
            f"Gamma: {SVM_GAMMA}\n"
        )
        file.write(
            "Class weight: balanced\n\n"
        )

        file.write(
            "Fold Results\n"
        )
        file.write(
            fold_df.to_string(index=False)
        )
        file.write(
            "\n\n"
        )

        file.write(
            "Four-Fold Average\n"
        )
        file.write(
            f"Accuracy: {fold_df['accuracy'].mean():.6f}\n"
        )
        file.write(
            f"Macro Precision: {fold_df['macro_precision'].mean():.6f}\n"
        )
        file.write(
            f"Macro Recall: {fold_df['macro_recall'].mean():.6f}\n"
        )
        file.write(
            f"Macro F1: {fold_df['macro_f1'].mean():.6f}\n"
        )
        file.write(
            f"Weighted F1: {fold_df['weighted_f1'].mean():.6f}\n\n"
        )

        file.write(
            "Overall Person-Independent Results\n"
        )
        file.write(
            f"Accuracy: {overall_accuracy:.6f}\n"
        )
        file.write(
            f"Macro Precision: {overall_macro_precision:.6f}\n"
        )
        file.write(
            f"Macro Recall: {overall_macro_recall:.6f}\n"
        )
        file.write(
            f"Macro F1: {overall_macro_f1:.6f}\n"
        )
        file.write(
            f"Weighted F1: {overall_weighted_f1:.6f}\n\n"
        )

        file.write(
            "Classification Report\n"
        )
        file.write(
            full_report
        )

        file.write(
            "\nMisclassification Pairs\n"
        )

        if len(misclassification_pairs) > 0:
            file.write(
                misclassification_pairs.to_string(index=False)
            )
        else:
            file.write(
                "No misclassifications."
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------
    print("\nFiles saved:")
    print(FOLD_RESULTS_PATH)
    print(PER_CLASS_PATH)
    print(PREDICTIONS_PATH)
    print(REPORT_PATH)
    print(CONFUSION_PATH)

    print("\n" + "=" * 70)
    print("Person-independent evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
