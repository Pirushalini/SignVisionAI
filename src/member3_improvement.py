import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

DATA_PATH = "data/sls_hand_landmarks_clean.csv"
BASELINE_MODEL_PATH = "models/sls_best_model.joblib"

RESULTS_DIR = "results"
MODELS_DIR = "models"

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "member3_improvement_comparison.csv"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "member3_improvement_report.txt"
)

PER_CLASS_PATH = os.path.join(
    RESULTS_DIR,
    "member3_baseline_vs_improved.csv"
)

CONFUSION_PATH = os.path.join(
    RESULTS_DIR,
    "member3_improved_confusion_matrix.png"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "sls_improved_model.joblib"
)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("Member 3 - SLS Model Improvement")
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    print("\n[1] Loading clean dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Samples  : {len(df)}")
    print(f"Features : {X.shape[1]}")
    print(f"Classes  : {y.nunique()}")

    # --------------------------------------------------
    # Same split as Member 2
    # --------------------------------------------------
    print("\n[2] Creating the same stratified 80/20 split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train : {len(X_train)}")
    print(f"Test  : {len(X_test)}")

    # --------------------------------------------------
    # Baseline model
    # --------------------------------------------------
    print("\n[3] Evaluating baseline SVM...")

    baseline_model = joblib.load(BASELINE_MODEL_PATH)

    baseline_pred = baseline_model.predict(X_test)

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_pred
    )

    print(
        f"Baseline Accuracy : "
        f"{baseline_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Baseline Macro F1 : "
        f"{baseline_metrics['macro_f1']:.4f}"
    )

    # --------------------------------------------------
    # Balanced SVM
    # --------------------------------------------------
    print("\n[4] Training class-balanced SVM...")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    balanced_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="rbf",
                random_state=RANDOM_STATE
            )
        )
    ])

    param_grid = {
        "classifier__C": [10, 50, 100],
        "classifier__gamma": ["scale", 0.01, 0.1],
        "classifier__class_weight": ["balanced"]
    }

    grid_search = GridSearchCV(
        balanced_pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    improved_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"Best CV Macro F1: "
        f"{grid_search.best_score_:.4f}"
    )

    # --------------------------------------------------
    # Evaluate improved model
    # --------------------------------------------------
    print("\n[5] Evaluating improved model...")

    improved_pred = improved_model.predict(X_test)

    improved_metrics = calculate_metrics(
        y_test,
        improved_pred
    )

    print(
        f"Improved Accuracy : "
        f"{improved_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Improved Macro Precision : "
        f"{improved_metrics['macro_precision']:.4f}"
    )

    print(
        f"Improved Macro Recall : "
        f"{improved_metrics['macro_recall']:.4f}"
    )

    print(
        f"Improved Macro F1 : "
        f"{improved_metrics['macro_f1']:.4f}"
    )

    print(
        f"Improved Weighted F1 : "
        f"{improved_metrics['weighted_f1']:.4f}"
    )

    # --------------------------------------------------
    # Compare baseline vs improved
    # --------------------------------------------------
    comparison_df = pd.DataFrame([
        {
            "Model": "Baseline SVM",
            **baseline_metrics
        },
        {
            "Model": "Class-Balanced SVM",
            **improved_metrics
        }
    ])

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    # --------------------------------------------------
    # Per-class comparison
    # --------------------------------------------------
    baseline_report = classification_report(
        y_test,
        baseline_pred,
        output_dict=True,
        zero_division=0
    )

    improved_report = classification_report(
        y_test,
        improved_pred,
        output_dict=True,
        zero_division=0
    )

    labels = sorted(y.unique())

    rows = []

    for label in labels:

        rows.append({
            "label": label,
            "baseline_precision": baseline_report[label]["precision"],
            "baseline_recall": baseline_report[label]["recall"],
            "baseline_f1": baseline_report[label]["f1-score"],
            "improved_precision": improved_report[label]["precision"],
            "improved_recall": improved_report[label]["recall"],
            "improved_f1": improved_report[label]["f1-score"],
            "f1_change":
                improved_report[label]["f1-score"]
                - baseline_report[label]["f1-score"]
        })

    per_class_df = pd.DataFrame(rows)

    per_class_df = per_class_df.sort_values(
        by="f1_change",
        ascending=True
    )

    per_class_df.to_csv(
        PER_CLASS_PATH,
        index=False
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    cm = confusion_matrix(
        y_test,
        improved_pred,
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
        "Improved SLS A-Z Classification - "
        "Class-Balanced SVM"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Save improved model
    # --------------------------------------------------
    joblib.dump(
        improved_model,
        MODEL_PATH
    )

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------
    baseline_report_text = classification_report(
        y_test,
        baseline_pred,
        zero_division=0
    )

    improved_report_text = classification_report(
        y_test,
        improved_pred,
        zero_division=0
    )

    with open(
        REPORT_PATH,
        "w"
    ) as file:

        file.write(
            "Member 3 - Model Improvement Report\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(
            "Baseline SVM Metrics\n"
        )
        file.write("-" * 60 + "\n")

        for key, value in baseline_metrics.items():
            file.write(
                f"{key}: {value:.6f}\n"
            )

        file.write(
            "\nImproved Class-Balanced SVM Metrics\n"
        )
        file.write("-" * 60 + "\n")

        for key, value in improved_metrics.items():
            file.write(
                f"{key}: {value:.6f}\n"
            )

        file.write(
            "\nBest Hyperparameters\n"
        )
        file.write("-" * 60 + "\n")

        file.write(
            str(grid_search.best_params_)
        )

        file.write(
            "\n\nBaseline Classification Report\n"
        )
        file.write("-" * 60 + "\n")
        file.write(
            baseline_report_text
        )

        file.write(
            "\n\nImproved Classification Report\n"
        )
        file.write("-" * 60 + "\n")
        file.write(
            improved_report_text
        )

    # --------------------------------------------------
    # Print most improved / reduced classes
    # --------------------------------------------------
    print("\n[6] Classes with largest F1 changes")
    print("-" * 60)

    print(
        per_class_df[
            [
                "label",
                "baseline_f1",
                "improved_f1",
                "f1_change"
            ]
        ].head(10).to_string(
            index=False
        )
    )

    print("\n[7] Improvement files saved:")
    print(COMPARISON_PATH)
    print(PER_CLASS_PATH)
    print(CONFUSION_PATH)
    print(REPORT_PATH)
    print(MODEL_PATH)

    print("\nImprovement analysis completed.")


if __name__ == "__main__":
    main()
