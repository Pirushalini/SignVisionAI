from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DATA_PATH = Path("data/sls_hand_landmarks_memberwise.csv")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULTS_PATH = (
    RESULTS_DIR / "member3_rotation_normalization_fold_results.csv"
)

REPORT_PATH = (
    RESULTS_DIR / "member3_rotation_normalization_report.txt"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "member3_rotation_normalization_predictions.csv"
)

MEMBERS = [
    "Member_1",
    "Member_2",
    "Member_3",
    "Member_4",
]

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Same configuration selected previously
SVM_C = 100
SVM_GAMMA = 0.01


def normalize_and_align_landmarks(X):
    """
    Hand-relative 3D normalization.

    1. Wrist (landmark 0) becomes the origin.
    2. Palm direction defines the first axis.
    3. Index and pinky MCP landmarks define the palm plane.
    4. All landmarks are transformed into this hand-relative coordinate system.
    5. Hand size is normalized using maximum distance from the wrist.
    """

    X = np.asarray(X, dtype=np.float32)
    output = []

    for sample in X:
        points = sample.reshape(21, 3).copy()

        wrist = points[0]

        # Move wrist to origin
        points = points - wrist

        # Key palm landmarks
        index_mcp = points[5]
        middle_mcp = points[9]
        ring_mcp = points[13]
        pinky_mcp = points[17]

        # Palm center direction
        palm_center = (
            index_mcp
            + middle_mcp
            + ring_mcp
            + pinky_mcp
        ) / 4.0

        palm_norm = np.linalg.norm(palm_center)

        if palm_norm < 1e-8:
            output.append(points.flatten())
            continue

        x_axis = palm_center / palm_norm

        # Palm plane normal
        palm_normal = np.cross(
            index_mcp,
            pinky_mcp
        )

        normal_norm = np.linalg.norm(palm_normal)

        if normal_norm < 1e-8:
            output.append(points.flatten())
            continue

        z_axis = palm_normal / normal_norm

        # Complete right-handed coordinate system
        y_axis = np.cross(
            z_axis,
            x_axis
        )

        y_norm = np.linalg.norm(y_axis)

        if y_norm < 1e-8:
            output.append(points.flatten())
            continue

        y_axis = y_axis / y_norm

        # Re-orthogonalize z
        z_axis = np.cross(
            x_axis,
            y_axis
        )

        z_norm = np.linalg.norm(z_axis)

        if z_norm < 1e-8:
            output.append(points.flatten())
            continue

        z_axis = z_axis / z_norm

        # Transform every landmark into local hand coordinates
        aligned = np.column_stack([
            points @ x_axis,
            points @ y_axis,
            points @ z_axis,
        ])

        # Hand-size normalization
        distances = np.linalg.norm(
            aligned,
            axis=1
        )

        scale = np.max(distances)

        if scale > 1e-8:
            aligned = aligned / scale

        output.append(
            aligned.flatten()
        )

    return np.asarray(
        output,
        dtype=np.float32
    )


def create_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
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


def calculate_metrics(y_true, y_pred):
    present_classes = sorted(np.unique(y_true))

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=present_classes,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=present_classes,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=present_classes,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def main():

    print("=" * 70)
    print("Member 3 - Rotation-Normalized Person-Independent Evaluation")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    landmark_columns = [
        f"lm{i}_{axis}"
        for i in range(21)
        for axis in ["x", "y", "z"]
    ]

    missing = [
        col for col in landmark_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing landmark columns: {missing}"
        )

    X = df[landmark_columns].values
    y = df["label"].values
    member_ids = df["member_id"].values
    image_names = df["image_name"].values

    print(f"\nDataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    print("Feature count: 63")

    all_predictions = []
    fold_results = []

    for held_out in MEMBERS:

        print("\n" + "=" * 70)
        print(f"TEST MEMBER: {held_out}")
        print("=" * 70)

        train_mask = member_ids != held_out
        test_mask = member_ids == held_out

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        test_images = image_names[test_mask]

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples : {len(X_test)}")

        print("\nApplying hand-relative rotation normalization...")

        X_train_transformed = normalize_and_align_landmarks(
            X_train
        )

        X_test_transformed = normalize_and_align_landmarks(
            X_test
        )

        model = create_model()

        print(
            f"Training SVM: "
            f"C={SVM_C}, gamma={SVM_GAMMA}, "
            f"class_weight=balanced"
        )

        model.fit(
            X_train_transformed,
            y_train
        )

        y_pred = model.predict(
            X_test_transformed
        )

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        print(
            f"\nAccuracy       : "
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Macro Precision: "
            f"{metrics['macro_precision']:.4f}"
        )

        print(
            f"Macro Recall   : "
            f"{metrics['macro_recall']:.4f}"
        )

        print(
            f"Macro F1       : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1    : "
            f"{metrics['weighted_f1']:.4f}"
        )

        fold_results.append({
            "held_out_member": held_out,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            **metrics,
        })

        for image_name, true_label, predicted_label in zip(
            test_images,
            y_test,
            y_pred
        ):
            all_predictions.append({
                "held_out_member": held_out,
                "image_name": image_name,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
            })

    fold_df = pd.DataFrame(fold_results)
    prediction_df = pd.DataFrame(all_predictions)

    fold_df.to_csv(
        FOLD_RESULTS_PATH,
        index=False
    )

    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Overall out-of-fold metrics
    # --------------------------------------------------------

    y_true = prediction_df["true_label"]
    y_pred = prediction_df["predicted_label"]

    overall = calculate_metrics(
        y_true,
        y_pred
    )

    print("\n" + "=" * 70)
    print("OVERALL ROTATION-NORMALIZED RESULTS")
    print("=" * 70)

    print(
        f"Accuracy       : "
        f"{overall['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{overall['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{overall['macro_recall']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{overall['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{overall['weighted_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Fold averages
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
    # Compare with previous person-independent result
    # --------------------------------------------------------

    previous_accuracy = 0.3895
    previous_macro_f1 = 0.3864

    accuracy_change = (
        overall["accuracy"] - previous_accuracy
    )

    macro_f1_change = (
        overall["macro_f1"] - previous_macro_f1
    )

    print("\nComparison with previous person-independent result:")
    print(
        f"Previous Accuracy : {previous_accuracy * 100:.2f}%"
    )
    print(
        f"New Accuracy      : {overall['accuracy'] * 100:.2f}%"
    )
    print(
        f"Accuracy Change   : {accuracy_change * 100:+.2f} percentage points"
    )

    print(
        f"\nPrevious Macro F1 : {previous_macro_f1:.4f}"
    )
    print(
        f"New Macro F1      : {overall['macro_f1']:.4f}"
    )
    print(
        f"Macro F1 Change   : {macro_f1_change:+.4f}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Member 3 - Rotation-Normalized "
            "Person-Independent Evaluation\n"
        )
        file.write("=" * 65 + "\n\n")

        file.write(
            "Method\n"
        )
        file.write(
            "Wrist translated to origin.\n"
        )
        file.write(
            "Palm-relative 3D coordinate system created using "
            "palm landmarks.\n"
        )
        file.write(
            "Hand size normalized using maximum wrist distance.\n"
        )
        file.write(
            "StandardScaler fitted within each training fold.\n"
        )
        file.write(
            "RBF SVM with class_weight='balanced'.\n"
        )
        file.write(
            f"C={SVM_C}, gamma={SVM_GAMMA}.\n\n"
        )

        file.write(
            "Fold Results\n"
        )
        file.write(
            fold_df.to_string(index=False)
        )
        file.write("\n\n")

        file.write(
            "Overall Results\n"
        )

        for key, value in overall.items():
            file.write(
                f"{key}: {value:.6f}\n"
            )

        file.write("\n")

        file.write(
            "Four-Fold Average\n"
        )

        for key in [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
        ]:
            file.write(
                f"{key}: "
                f"{fold_df[key].mean():.6f}\n"
            )

        file.write("\n")

        file.write(
            "Comparison With Previous Person-Independent Result\n"
        )
        file.write(
            f"Previous Accuracy: {previous_accuracy:.6f}\n"
        )
        file.write(
            f"New Accuracy: {overall['accuracy']:.6f}\n"
        )
        file.write(
            f"Accuracy Change: {accuracy_change:.6f}\n"
        )
        file.write(
            f"Previous Macro F1: {previous_macro_f1:.6f}\n"
        )
        file.write(
            f"New Macro F1: {overall['macro_f1']:.6f}\n"
        )
        file.write(
            f"Macro F1 Change: {macro_f1_change:.6f}\n"
        )

    print("\nSaved:")
    print(FOLD_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(REPORT_PATH)

    print("\n" + "=" * 70)
    print("Rotation-normalized evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
