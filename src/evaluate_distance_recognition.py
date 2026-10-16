"""
SignVisionAI - Distance-Based Recognition Evaluation Script
Part 3: Distance-Based Recognition (1-NN & KNN)

This script performs rigorous evaluation of the distance-based recognition module
using the normalized SLS dataset (data/sls_normalized_square_pad.npz).

Evaluation Methodology:
1. Leakage Mitigation & Splitting:
   - Identifies exact duplicates in the feature dataset to prevent identical row leakage.
   - Performs stratified train/val/test splitting (60% Reference / 20% Validation / 20% Test)
     using a fixed random seed (seed=42) for strict reproducibility.
   - Prevents test samples from ever being compared against themselves.
   - Documents dataset limitations (e.g., Leave-One-Signer-Out cannot be used because letters
     H and T were only successfully detected for Member_4 due to MediaPipe extraction failures).

2. Model Benchmarking:
   - Evaluates Euclidean 1-NN, KNN (k=3), and KNN (k=5).
   - Compares the baseline 126-feature representation against the augmented 129-feature
     representation (which incorporates normalized relative wrist displacement).

3. Validation-Calibrated Uncertainty Rejection:
   - Calibrates empirical distance thresholds on the validation set (at 95th and 99th percentiles).
   - Evaluates the calibrated thresholds on the untouched held-out test set.
   - Reports rejection rates and accuracy among accepted predictions.

4. Artifacts Generated:
   - results/distance_recognition_evaluation.json
   - results/distance_recognition_summary.csv
   - results/per_class_accuracy_knn3.csv
   - results/confusion_matrix_knn3.csv
"""

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distance_recognition import (
    DEFAULT_DATASET_NPZ,
    ONE_HAND_FEATURE_COUNT,
    TWO_HAND_FEATURE_COUNT,
    AUGMENTED_TWO_HAND_FEATURE_COUNT,
    DistanceRecognizer,
    calibrate_distance_thresholds,
)
from src.extract_landmarks import get_member_id

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default random seed for reproducible train/val/test splits
RANDOM_SEED = 42



def stratified_split_indices(
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    atol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform deterministic stratified split of indices into train, validation, and test sets.

    When `features` is provided, exact duplicate feature vectors are automatically clustered
    and atomically assigned to the SAME split. This completely prevents duplicate leakage
    across Reference, Validation, and Test sets while maintaining close class stratification.

    Args:
        labels: 1D array of class labels.
        features: Optional 2D array of feature vectors (shape (N, D)). If provided,
                  identical feature vectors are guaranteed to stay in the same split.
        train_ratio: Proportion for reference / train set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for held-out test set.
        random_seed: Seed for reproducibility.
        atol: Absolute tolerance for identifying identical feature vectors.

    Returns:
        Tuple of (train_indices, val_indices, test_indices) as 1D integer arrays.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = np.random.default_rng(random_seed)
    n_samples = len(labels)

    # 1. Cluster exact duplicates if features are provided
    if features is not None:
        visited = set()
        clusters: List[List[int]] = []
        for i in range(n_samples):
            if i in visited:
                continue
            c = [i]
            visited.add(i)
            for j in range(i + 1, n_samples):
                if j not in visited and np.allclose(features[i], features[j], atol=atol):
                    c.append(j)
                    visited.add(j)
            clusters.append(c)
    else:
        clusters = [[i] for i in range(n_samples)]

    # Group clusters by class label
    class_clusters: Dict[Any, List[List[int]]] = {}
    for c in clusters:
        lbl = labels[c[0]]
        class_clusters.setdefault(lbl, []).append(c)

    train_idx_list: List[int] = []
    val_idx_list: List[int] = []
    test_idx_list: List[int] = []

    for lbl in sorted(class_clusters.keys()):
        cls_c = list(class_clusters[lbl])
        rng.shuffle(cls_c)

        total_cls_samples = sum(len(c) for c in cls_c)
        target_train = max(1, int(round(total_cls_samples * train_ratio)))
        target_val = max(1, int(round(total_cls_samples * val_ratio)))

        c_train: List[List[int]] = []
        c_val: List[List[int]] = []
        c_test: List[List[int]] = []
        train_cnt = 0
        val_cnt = 0

        for c in cls_c:
            c_len = len(c)
            # Greedily assign cluster atomically
            if train_cnt + c_len <= target_train or (train_cnt == 0):
                c_train.append(c)
                train_cnt += c_len
            elif val_cnt + c_len <= target_val or (val_cnt == 0 and len(cls_c) >= 3):
                c_val.append(c)
                val_cnt += c_len
            else:
                c_test.append(c)

        # Ensure test partition has at least one cluster if class has >= 3 samples
        if len(c_test) == 0 and len(cls_c) >= 3:
            if len(c_val) > 1:
                c_test.append(c_val.pop())
            elif len(c_train) > 1:
                c_test.append(c_train.pop())

        for c in c_train:
            train_idx_list.extend(c)
        for c in c_val:
            val_idx_list.extend(c)
        for c in c_test:
            test_idx_list.extend(c)

    train_indices = np.array(sorted(train_idx_list), dtype=int)
    val_indices = np.array(sorted(val_idx_list), dtype=int)
    test_indices = np.array(sorted(test_idx_list), dtype=int)

    # Sanity checks
    all_indices = set(train_indices).union(val_indices).union(test_indices)
    assert len(all_indices) == n_samples, "Indices mismatch"
    assert len(set(train_indices).intersection(val_indices)) == 0, "Leakage between train and val"
    assert len(set(train_indices).intersection(test_indices)) == 0, "Leakage between train and test"
    assert len(set(val_indices).intersection(test_indices)) == 0, "Leakage between val and test"

    return train_indices, val_indices, test_indices


def evaluate_recognizer_on_test_set(
    recognizer: DistanceRecognizer,
    test_features_1h: np.ndarray,
    test_labels_1h: np.ndarray,
    test_features_2h: np.ndarray,
    test_labels_2h: np.ndarray,
    k: int,
    threshold_1h: Optional[float] = None,
    threshold_2h: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run evaluation of the recognizer across both 1-hand and 2-hand test samples.

    Returns comprehensive metrics: overall accuracy, accepted accuracy, rejection rate,
    per-class accuracy, and confusion matrix.
    """
    all_true_labels: List[str] = []
    all_pred_labels: List[Optional[str]] = []
    all_statuses: List[str] = []
    all_distances: List[float] = []

    # 1-Hand predictions (Letter C)
    for feat, true_lbl in zip(test_features_1h, test_labels_1h):
        res = recognizer.predict(
            feat, hand_count=1, k=k, threshold=threshold_1h
        )
        all_true_labels.append(str(true_lbl))
        all_pred_labels.append(res["letter"])
        all_statuses.append(res["status"])
        all_distances.append(res["distance"])

    # 2-Hand predictions (Letters A, B, D..Z)
    for feat, true_lbl in zip(test_features_2h, test_labels_2h):
        res = recognizer.predict(
            feat, hand_count=2, k=k, threshold=threshold_2h
        )
        all_true_labels.append(str(true_lbl))
        all_pred_labels.append(res["letter"])
        all_statuses.append(res["status"])
        all_distances.append(res["distance"])

    total_queries = len(all_true_labels)
    accepted_mask = [s == "recognized" for s in all_statuses]
    uncertain_mask = [s == "uncertain" for s in all_statuses]

    n_accepted = sum(accepted_mask)
    n_uncertain = sum(uncertain_mask)
    rejection_rate = (n_uncertain / total_queries) * 100.0 if total_queries > 0 else 0.0

    # Correct accepted predictions
    n_correct_accepted = sum(
        1 for true, pred, acc in zip(all_true_labels, all_pred_labels, accepted_mask)
        if acc and (true == pred)
    )

    # Accuracy among accepted queries
    accepted_accuracy = (n_correct_accepted / n_accepted) * 100.0 if n_accepted > 0 else 0.0

    # Overall accuracy (considering rejected as non-correct)
    overall_accuracy = (n_correct_accepted / total_queries) * 100.0 if total_queries > 0 else 0.0

    # Per-class metrics
    classes = sorted(list(set(all_true_labels)))
    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in classes:
        cls_indices = [i for i, y in enumerate(all_true_labels) if y == cls]
        cls_total = len(cls_indices)
        cls_accepted = sum(1 for i in cls_indices if accepted_mask[i])
        cls_correct = sum(
            1 for i in cls_indices if accepted_mask[i] and all_pred_labels[i] == cls
        )
        cls_acc = (cls_correct / cls_total) * 100.0 if cls_total > 0 else 0.0
        cls_accepted_acc = (cls_correct / cls_accepted) * 100.0 if cls_accepted > 0 else 0.0

        per_class[cls] = {
            "total": cls_total,
            "accepted": cls_accepted,
            "uncertain": cls_total - cls_accepted,
            "correct": cls_correct,
            "accuracy_overall_pct": round(cls_acc, 2),
            "accuracy_accepted_pct": round(cls_accepted_acc, 2),
        }

    # Confusion matrix across all classes
    # Rows: True classes, Columns: Predicted classes (+ 'UNCERTAIN')
    matrix_cols = classes + ["UNCERTAIN"]
    conf_matrix: Dict[str, Dict[str, int]] = {c: {col: 0 for col in matrix_cols} for c in classes}

    for true_lbl, pred_lbl, status in zip(all_true_labels, all_pred_labels, all_statuses):
        if status == "uncertain" or pred_lbl is None:
            conf_matrix[true_lbl]["UNCERTAIN"] += 1
        else:
            if pred_lbl in conf_matrix[true_lbl]:
                conf_matrix[true_lbl][pred_lbl] += 1
            else:
                conf_matrix[true_lbl]["UNCERTAIN"] += 1

    return {
        "k": k,
        "threshold_1h": threshold_1h,
        "threshold_2h": threshold_2h,
        "total_queries": total_queries,
        "n_accepted": n_accepted,
        "n_uncertain": n_uncertain,
        "rejection_rate_pct": round(rejection_rate, 2),
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "accepted_accuracy_pct": round(accepted_accuracy, 2),
        "correct_accepted_count": n_correct_accepted,
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
        "mean_nearest_distance": round(float(np.mean(all_distances)), 5),
    }


def main(seed: int = RANDOM_SEED):
    print("=" * 80)
    print("SignVisionAI - Part 3: Distance-Based Recognition Evaluation")
    print("=" * 80)

    dataset_path = DEFAULT_DATASET_NPZ
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"\n1. Loading normalized dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    features_1h = data["features_1h"]
    labels_1h = data["labels_1h"]
    image_ids_1h = data["image_ids_1h"]

    features_2h = data["features_2h"]
    labels_2h = data["labels_2h"]
    image_ids_2h = data["image_ids_2h"]
    rel_wrist_2h = data["relative_wrist_displacement_2h"]

    features_2h_aug = np.concatenate([features_2h, rel_wrist_2h], axis=1).astype(np.float32)

    print(f"   - 1-hand samples (Letter C): {len(labels_1h)} samples, shape: {features_1h.shape}")
    print(f"   - 2-hand samples (25 letters): {len(labels_2h)} samples, shape: {features_2h.shape}")
    print(f"   - 2-hand augmented features shape: {features_2h_aug.shape}")

    # 2. Check for exact duplicate rows in the raw feature vectors
    print("\n2. Checking for exact duplicate feature vectors and leakage audit:")
    dup_2h_count = len(features_2h) - len(np.unique(features_2h, axis=0))
    print(f"   - 1-hand exact duplicates: {len(features_1h) - len(np.unique(features_1h, axis=0))}")
    print(f"   - 2-hand exact duplicates: {dup_2h_count}")

    # Identify exact duplicate pairs
    dup_pairs = []
    for i in range(len(features_2h)):
        for j in range(i + 1, len(features_2h)):
            if np.allclose(features_2h[i], features_2h[j], atol=1e-7):
                dup_pairs.append((i, j, image_ids_2h[i], image_ids_2h[j], labels_2h[i]))
    for idx_a, idx_b, img_a, img_b, lbl in dup_pairs:
        print(f"     * Found duplicate pair: {img_a} == {img_b} (Class '{lbl}')")

    # 3. Stratified Splitting
    print("\n3. Performing Stratified Splitting (60% Reference, 20% Validation, 20% Test):")
    idx_train_1h, idx_val_1h, idx_test_1h = stratified_split_indices(
        labels_1h, features=features_1h, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=seed
    )
    idx_train_2h, idx_val_2h, idx_test_2h = stratified_split_indices(
        labels_2h, features=features_2h, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=seed
    )

    print(f"   - 1-Hand: Reference={len(idx_train_1h)}, Validation={len(idx_val_1h)}, Test={len(idx_test_1h)}")
    print(f"   - 2-Hand: Reference={len(idx_train_2h)}, Validation={len(idx_val_2h)}, Test={len(idx_test_2h)}")
    print(f"   - Total : Reference={len(idx_train_1h)+len(idx_train_2h)}, Validation={len(idx_val_1h)+len(idx_val_2h)}, Test={len(idx_test_1h)+len(idx_test_2h)}")

    # Audit duplicate pairs against split locations across ALL sets
    print("   - Duplicate Leakage Audit across ALL 3 splits (Reference, Validation, Test):")
    leakage_found = False
    for idx_a, idx_b, img_a, img_b, lbl in dup_pairs:
        loc_a = "Reference" if idx_a in idx_train_2h else ("Validation" if idx_a in idx_val_2h else "Test")
        loc_b = "Reference" if idx_b in idx_train_2h else ("Validation" if idx_b in idx_val_2h else "Test")
        is_same = (loc_a == loc_b)
        print(f"     * Pair ({img_a}, {img_b}): {loc_a} vs {loc_b} -> SAME SPLIT: {is_same}")
        if not is_same:
            leakage_found = True

    assert not leakage_found, "CRITICAL: Duplicate sample leaked across Reference, Validation, or Test sets!"
    print("     * Verified: ZERO duplicate leakage across ALL three sets (Ref, Val, and Test).")

    # Near-Duplicate Risk Audit (non-zero distance < 0.15)
    print("\n   - Near-Duplicate Risk Audit (pairwise distance < 0.15):")
    near_dup_pairs = []
    d2 = np.linalg.norm(features_2h[:, None, :] - features_2h[None, :, :], axis=-1)
    np.fill_diagonal(d2, np.inf)
    for i in range(len(features_2h)):
        for j in range(i + 1, len(features_2h)):
            if 0.0 < d2[i, j] < 0.15:
                near_dup_pairs.append((image_ids_2h[i], image_ids_2h[j], labels_2h[i], d2[i, j]))
    near_dup_pairs.sort(key=lambda x: x[3])
    print(f"     * Total near-duplicate pairs (0.0 < dist < 0.15): {len(near_dup_pairs)}")
    for a, b, lbl, dst in near_dup_pairs:
        print(f"       - {a} and {b} (Class '{lbl}'): dist = {dst:.5f}")

    # Slice datasets
    train_f1h, val_f1h, test_f1h = features_1h[idx_train_1h], features_1h[idx_val_1h], features_1h[idx_test_1h]
    train_l1h, val_l1h, test_l1h = labels_1h[idx_train_1h], labels_1h[idx_val_1h], labels_1h[idx_test_1h]

    train_f2h, val_f2h, test_f2h = features_2h[idx_train_2h], features_2h[idx_val_2h], features_2h[idx_test_2h]
    train_l2h, val_l2h, test_l2h = labels_2h[idx_train_2h], labels_2h[idx_val_2h], labels_2h[idx_test_2h]

    train_f2h_aug = features_2h_aug[idx_train_2h]
    val_f2h_aug = features_2h_aug[idx_val_2h]
    test_f2h_aug = features_2h_aug[idx_test_2h]

    # 4. Calibrate Uncertainty Thresholds on Validation Set
    print("\n4. Calibrating Distance Thresholds on Validation Set:")
    calib_95 = calibrate_distance_thresholds(
        val_features_1h=val_f1h,
        ref_features_1h=train_f1h,
        val_features_2h=val_f2h,
        ref_features_2h=train_f2h,
        percentile=95.0,
        k=1,
    )
    calib_99 = calibrate_distance_thresholds(
        val_features_1h=val_f1h,
        ref_features_1h=train_f1h,
        val_features_2h=val_f2h,
        ref_features_2h=train_f2h,
        percentile=99.0,
        k=1,
    )

    t95_1h, t95_2h = calib_95["threshold_1h"], calib_95["threshold_2h"]
    t99_1h, t99_2h = calib_99["threshold_1h"], calib_99["threshold_2h"]

    print(f"   - 95th Percentile Thresholds: 1-hand (C) = {t95_1h:.4f}, 2-hand = {t95_2h:.4f}")
    print(f"   - 99th Percentile Thresholds: 1-hand (C) = {t99_1h:.4f}, 2-hand = {t99_2h:.4f}")

    # Build recognizers with reference libraries
    rec_baseline = DistanceRecognizer(k=3, use_augmented_2h=False)
    rec_baseline.set_references_1h(train_f1h, train_l1h)
    rec_baseline.set_references_2h(train_f2h, train_l2h)

    rec_augmented = DistanceRecognizer(k=3, use_augmented_2h=True)
    rec_augmented.set_references_1h(train_f1h, train_l1h)
    rec_augmented.set_references_2h(train_f2h_aug, train_l2h)

    # 5. Benchmarking Experiments on Held-Out Test Set
    print("\n5. Benchmarking on Untouched Held-Out Test Set (160 samples):")
    benchmarks = []

    experiments = [
        # (Name, Recognizer, k, feat2_test, t_1h, t_2h, desc)
        ("1-NN (k=1, Baseline 126, No Rejection)", rec_baseline, 1, test_f2h, None, None, "1nn_baseline_raw"),
        ("KNN (k=3, Baseline 126, No Rejection)", rec_baseline, 3, test_f2h, None, None, "knn3_baseline_raw"),
        ("KNN (k=5, Baseline 126, No Rejection)", rec_baseline, 5, test_f2h, None, None, "knn5_baseline_raw"),
        ("KNN (k=3, Baseline 126, 95% Threshold)", rec_baseline, 3, test_f2h, t95_1h, t95_2h, "knn3_baseline_thresh95"),
        ("KNN (k=3, Baseline 126, 99% Threshold)", rec_baseline, 3, test_f2h, t99_1h, t99_2h, "knn3_baseline_thresh99"),
        ("1-NN (k=1, Augmented 129, No Rejection)", rec_augmented, 1, test_f2h_aug, None, None, "1nn_aug_raw"),
        ("KNN (k=3, Augmented 129, No Rejection)", rec_augmented, 3, test_f2h_aug, None, None, "knn3_aug_raw"),
        ("KNN (k=5, Augmented 129, No Rejection)", rec_augmented, 5, test_f2h_aug, None, None, "knn5_aug_raw"),
    ]

    all_results = {}

    summary_rows = []
    for exp_title, recognizer, k_val, f2_test, t1, t2, exp_id in experiments:
        res = evaluate_recognizer_on_test_set(
            recognizer=recognizer,
            test_features_1h=test_f1h,
            test_labels_1h=test_l1h,
            test_features_2h=f2_test,
            test_labels_2h=test_l2h,
            k=k_val,
            threshold_1h=t1,
            threshold_2h=t2,
        )
        all_results[exp_id] = res

        summary_rows.append({
            "Experiment": exp_title,
            "k": k_val,
            "Representation": "Augmented (129)" if "Augmented" in exp_title else "Baseline (126)",
            "Threshold": "None" if t1 is None else ("95%" if t1 == t95_1h else "99%"),
            "Total Queries": res["total_queries"],
            "Accepted": res["n_accepted"],
            "Uncertain": res["n_uncertain"],
            "Rejection Rate (%)": res["rejection_rate_pct"],
            "Overall Accuracy (%)": res["overall_accuracy_pct"],
            "Accepted Accuracy (%)": res["accepted_accuracy_pct"],
            "Mean NN Distance": res["mean_nearest_distance"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nBenchmark Summary Table:")
    print(summary_df.to_string(index=False))

    # Save summary CSV
    summary_csv_path = RESULTS_DIR / "distance_recognition_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary CSV to: {summary_csv_path}")

    # 6. Save Detailed Per-Class Metrics and Confusion Matrix for Best Model (KNN k=3 Baseline)
    best_res = all_results["knn3_baseline_raw"]
    per_class_df = pd.DataFrame.from_dict(best_res["per_class"], orient="index")
    per_class_csv_path = RESULTS_DIR / "per_class_accuracy_knn3.csv"
    per_class_df.to_csv(per_class_csv_path)
    print(f"Saved per-class accuracy to: {per_class_csv_path}")

    conf_matrix_df = pd.DataFrame.from_dict(best_res["confusion_matrix"], orient="index")
    conf_matrix_csv_path = RESULTS_DIR / "confusion_matrix_knn3.csv"
    conf_matrix_df.to_csv(conf_matrix_csv_path)
    print(f"Saved confusion matrix to: {conf_matrix_csv_path}")

    # Also save 1-NN and 5-NN confusion matrices
    res_1nn = all_results["1nn_baseline_raw"]
    pd.DataFrame.from_dict(res_1nn["confusion_matrix"], orient="index").to_csv(
        RESULTS_DIR / "confusion_matrix_1nn.csv"
    )
    res_5nn = all_results["knn5_baseline_raw"]
    pd.DataFrame.from_dict(res_5nn["confusion_matrix"], orient="index").to_csv(
        RESULTS_DIR / "confusion_matrix_knn5.csv"
    )

    # 7. Save complete JSON evaluation record
    json_output_path = RESULTS_DIR / "distance_recognition_evaluation.json"
    full_output = {
        "metadata": {
            "dataset": str(DEFAULT_DATASET_NPZ),
            "split": "Stratified 60% Reference / 20% Validation / 20% Test",
            "seed": seed,
            "counts": {
                "reference_total": len(idx_train_1h) + len(idx_train_2h),
                "validation_total": len(idx_val_1h) + len(idx_val_2h),
                "test_total": len(idx_test_1h) + len(idx_test_2h),
                "reference_1h": len(idx_train_1h),
                "validation_1h": len(idx_val_1h),
                "test_1h": len(idx_test_1h),
                "reference_2h": len(idx_train_2h),
                "validation_2h": len(idx_val_2h),
                "test_2h": len(idx_test_2h),
            },
            "calibrated_thresholds": {
                "percentile_95": {"1h": t95_1h, "2h": t95_2h},
                "percentile_99": {"1h": t99_1h, "2h": t99_2h},
            },
        },
        "experiments": all_results,
    }

    with open(json_output_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved complete evaluation record to: {json_output_path}")

    # 8. Print Per-Class Accuracy Highlights
    print("\nPer-Class Recognition Highlights (KNN k=3, Baseline):")
    print(f"{'Letter':<8}{'Total':<8}{'Correct':<10}{'Accuracy (%)':<15}")
    print("-" * 40)
    for cls, row in best_res["per_class"].items():
        print(f"{cls:<8}{row['total']:<8}{row['correct']:<10}{row['accuracy_overall_pct']:<15.2f}")

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
