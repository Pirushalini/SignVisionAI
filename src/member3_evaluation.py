import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

DATA_PATH = "data/sls_hand_landmarks_clean.csv"
MODEL_PATH = "models/sls_best_model.joblib"
RESULTS_DIR = "results"

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "member3_evaluation_report.txt"
)

CLASS_METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "member3_per_class_metrics.csv"
)

ERRORS_PATH = os.path.join(
    RESULTS_DIR,
    "member3_misclassifications.csv"
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "member3_confusion_matrix.png"
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("Member 3 - SLS Model Evaluation and Error Analysis")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    print("\n[1] Loading clean SLS landmark dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Total samples : {len(df)}")
    print(f"Features      : {X.shape[1]}")
    print(f"Classes       : {y.nunique()}")

    # --------------------------------------------------
    # Recreate same train/test split used by Member 2
    # --------------------------------------------------
    print("\n[2] Recreating stratified test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # --------------------------------------------------
    # Load best trained model
    # --------------------------------------------------
    print("\n[3] Loading trained SVM pipeline...")

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    print("\n[4] Generating predictions...")

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n[5] Overall Evaluation")
    print("-" * 50)
    print(f"Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision   : {precision:.4f}")
    print(f"Macro Recall      : {recall:.4f}")
    print(f"Macro F1          : {f1:.4f}")

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n[6] Classification Report")
    print(report)

    # --------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    class_rows = []

    for label in sorted(y.unique()):
        metrics = report_dict[label]

        class_rows.append({
            "label": label,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1-score"],
            "support": int(metrics["support"])
        })

    class_df = pd.DataFrame(class_rows)

    class_df = class_df.sort_values(
        by="f1_score",
        ascending=True
    )

    class_df.to_csv(
        CLASS_METRICS_PATH,
        index=False
    )

    print(
        f"Saved per-class metrics to: {CLASS_METRICS_PATH}"
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    labels = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation="vertical"
    )

    ax.set_title(
        "Member 3 - SLS A-Z SVM Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved confusion matrix to: {CONFUSION_PATH}"
    )

    # --------------------------------------------------
    # Misclassification analysis
    # --------------------------------------------------
    error_rows = []

    for true_label, predicted_label in zip(
        y_test,
        y_pred
    ):
        if true_label != predicted_label:
            error_rows.append({
                "true_label": true_label,
                "predicted_label": predicted_label
            })

    errors_df = pd.DataFrame(error_rows)

    if not errors_df.empty:

        pair_counts = (
            errors_df
            .groupby(
                ["true_label", "predicted_label"]
            )
            .size()
            .reset_index(name="count")
            .sort_values(
                "count",
                ascending=False
            )
        )

    else:

        pair_counts = pd.DataFrame(
            columns=[
                "true_label",
                "predicted_label",
                "count"
            ]
        )

    pair_counts.to_csv(
        ERRORS_PATH,
        index=False
    )

    print(
        f"Saved misclassification analysis to: {ERRORS_PATH}"
    )

    # --------------------------------------------------
    # Save text report
    # --------------------------------------------------
    with open(
        REPORT_PATH,
        "w"
    ) as file:

        file.write(
            "Member 3 - SLS Model Evaluation Report\n"
        )
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Dataset samples: {len(df)}\n"
        )
        file.write(
            f"Test samples: {len(X_test)}\n"
        )
        file.write(
            f"Features: {X.shape[1]}\n"
        )
        file.write(
            f"Classes: {y.nunique()}\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )
        file.write(
            f"Macro Precision: {precision:.6f}\n"
        )
        file.write(
            f"Macro Recall: {recall:.6f}\n"
        )
        file.write(
            f"Macro F1: {f1:.6f}\n\n"
        )

        file.write(
            "Classification Report\n"
        )
        file.write("-" * 60 + "\n")
        file.write(report)

        file.write(
            "\n\nMisclassification Pairs\n"
        )
        file.write("-" * 60 + "\n")

        if pair_counts.empty:
            file.write("No misclassifications found.\n")
        else:
            file.write(
                pair_counts.to_string(index=False)
            )

    print(
        f"Saved evaluation report to: {REPORT_PATH}"
    )

    # --------------------------------------------------
    # Print difficult classes
    # --------------------------------------------------
    print("\n[7] Lowest F1-score classes")
    print("-" * 50)

    print(
        class_df.head(10).to_string(
            index=False
        )
    )

    print("\n[8] Top misclassification pairs")
    print("-" * 50)

    if pair_counts.empty:
        print("No errors found.")
    else:
        print(
            pair_counts.head(15).to_string(
                index=False
            )
        )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
