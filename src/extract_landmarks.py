"""
SignVisionAI - SLS Dataset Landmark Extraction Pipeline
Part 2: Landmark Extraction + Normalization

This script processes SLS dataset images using MediaPipe Tasks HandLandmarker, applies
wrist-centering and palm-scale normalization, separates one-hand and two-hand signs,
and exports structured reference features for distance-based recognition (Part 3).

Features:
- Safe MediaPipe Tasks HandLandmarker initialization.
- Separation of one-hand signs (Letter C: 63 features) and two-hand signs (all other letters: 126 features).
- Preservation of relative wrist displacement for two-hand signs as a separate 3D feature.
- Explicit image orientation policy (un-mirrored third-person photos).
- Structured NPZ export with embedded JSON metadata.
- Comprehensive audit summary (CSV) tracking per-letter detection success and failure reasons.
- Clean per-letter summary table with total, success, failure, and success percentage.
- Preservation of source image identifiers for traceability without including member ID in feature vectors.
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import (
    DEFAULT_EPSILON,
    HandFeatureResult,
    HandednessError,
    NUM_LANDMARKS,
    ONE_HAND_FEATURE_COUNT,
    RELATIVE_WRIST_FEATURE_COUNT,
    TWO_HAND_FEATURE_COUNT,
    format_hand_features,
)
from src.preprocessing import DEFAULT_TARGET_SIZE, pad_to_square

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default paths
DEFAULT_DATASET_DIR = Path("data/SLS_Dataset")
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")
DEFAULT_OUTPUT_NPZ = Path("data/sls_normalized_landmarks.npz")
DEFAULT_SUMMARY_CSV = Path("results/full_extraction_summary.csv")


def get_member_id(image_name: str) -> str:
    """
    Infer contributor Member ID from dataset image filename convention:
    - 01 to 20 -> Member_1
    - 21 to 40 -> Member_2
    - 41 to 60 -> Member_3
    - 61 to 80 -> Member_4
    """
    match = re.search(r"_(\d+)\.(jpg|jpeg|png|bmp|webp)$", image_name, re.IGNORECASE)
    if not match:
        return "Unknown"
    num = int(match.group(1))
    if 1 <= num <= 20:
        return "Member_1"
    elif 21 <= num <= 40:
        return "Member_2"
    elif 41 <= num <= 60:
        return "Member_3"
    elif 61 <= num <= 80:
        return "Member_4"
    return "Unknown"


def load_mediapipe_landmarker(model_path: Path, min_conf: float = 0.3):
    """
    Safely initialize MediaPipe Tasks HandLandmarker.

    Raises:
        FileNotFoundError: If the .task model file does not exist on disk.
    """
    if not model_path.exists():
        alt_path = Path("notebooks/hand_landmarker.task")
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"MediaPipe model file not found at: {model_path.resolve()}\n"
                "Please place the official 'hand_landmarker.task' file in the 'models/' folder."
            )

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def process_single_image(
    image_path: Path,
    detector: Any,
    expected_hands: int,
    is_mirrored: bool = False,
    min_confidence: float = 0.0,
    swap_unmirrored: bool = False,
    preprocessing: str = "none",
    target_size: int = DEFAULT_TARGET_SIZE,
    upscale: bool = False,
) -> Tuple[Optional[HandFeatureResult], int, str, str]:
    """
    Load an image, apply optional preprocessing, run MediaPipe HandLandmarker,
    and format normalized features.

    Preprocessing modes:
    - 'none' (baseline): Load image directly via mp.Image.create_from_file (unmodified original).
    - 'square_pad': Aspect-ratio-preserving symmetric square padding + resize to target_size.

    Returns:
        Tuple[Optional[HandFeatureResult], detected_hands_count, handedness_info, status_reason]:
        - Result object if extraction succeeded and matched expected hand count; otherwise None.
        - Number of detected hands (int).
        - String describing detected handedness labels.
        - Descriptive failure or success reason string.
    """
    import mediapipe as mp

    if not image_path.exists():
        return None, 0, "", "FILE_NOT_FOUND"

    prep_norm = (preprocessing or "none").strip().lower()

    if prep_norm in ("none", "original", "baseline"):
        try:
            mp_image = mp.Image.create_from_file(str(image_path))
        except Exception as exc:
            return None, 0, "", f"IMAGE_READ_ERROR: {exc}"
    elif prep_norm in ("square_pad", "square_padding", "square"):
        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            if data.size == 0:
                return None, 0, "", "EMPTY_IMAGE_FILE"
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                return None, 0, "", "IMAGE_DECODE_ERROR"
            padded = pad_to_square(bgr, target_size=target_size, upscale=upscale)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        except Exception as exc:
            return None, 0, "", f"PREPROCESSING_ERROR: {exc}"
    else:
        return None, 0, "", f"UNKNOWN_PREPROCESSING_METHOD: {preprocessing}"

    try:
        detection_result = detector.detect(mp_image)
    except Exception as exc:
        return None, 0, "", f"DETECTOR_ERROR: {exc}"

    detected_hands = detection_result.hand_landmarks or []
    handedness = detection_result.handedness or []
    n_detected = len(detected_hands)

    # Extract raw labels for audit
    raw_labels = []
    for h in handedness:
        if h and len(h) > 0:
            c = h[0]
            raw_labels.append(f"{c.category_name}:{c.score:.2f}")
    handedness_str = "|".join(raw_labels)

    if n_detected == 0:
        return None, 0, "", "NO_HAND_DETECTED"

    if n_detected != expected_hands:
        return None, n_detected, handedness_str, f"HAND_COUNT_MISMATCH (got {n_detected}, expected {expected_hands})"

    try:
        feature_result = format_hand_features(
            detected_hands=detected_hands,
            handedness=handedness,
            is_mirrored=is_mirrored,
            min_confidence=min_confidence,
            strict=False,
            swap_unmirrored=swap_unmirrored,
        )
    except Exception as exc:
        return None, n_detected, handedness_str, f"NORMALIZATION_ERROR: {exc}"

    if not feature_result.is_valid:
        return None, n_detected, handedness_str, f"RESOLUTION_FAILED: {feature_result.status} ({feature_result.error_message})"

    resolved_hd_str = "|".join(feature_result.resolved_handedness)
    return feature_result, n_detected, resolved_hd_str, "OK"


def extract_dataset_landmarks(
    dataset_dir: Path,
    model_path: Path,
    output_npz_path: Optional[Path] = None,
    summary_csv_path: Optional[Path] = None,
    letters: Optional[Sequence[str]] = None,
    single_hand_letters: Sequence[str] = ("C",),
    max_per_letter: Optional[int] = None,
    is_mirrored: bool = False,
    min_conf: float = 0.3,
    swap_unmirrored: bool = False,
    preprocessing: str = "none",
    target_size: int = DEFAULT_TARGET_SIZE,
    upscale: bool = False,
    sample_csv_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run extraction pipeline across dataset folders (or a specified evaluation sample)
    and return/save structured features.
    """
    dataset_dir = Path(dataset_dir)
    model_path = Path(model_path)
    single_hand_set = {c.strip().upper() for c in single_hand_letters}

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir.resolve()}")

    landmarker = load_mediapipe_landmarker(model_path, min_conf=min_conf)

    # Storage for 1-hand samples
    features_1h_list: List[np.ndarray] = []
    labels_1h_list: List[str] = []
    image_ids_1h_list: List[str] = []

    # Storage for 2-hand samples
    features_2h_list: List[np.ndarray] = []
    rel_wrist_2h_list: List[np.ndarray] = []
    labels_2h_list: List[str] = []
    image_ids_2h_list: List[str] = []

    # Audit records for CSV
    audit_records: List[Dict[str, Any]] = []
    per_letter_stats: List[Dict[str, Any]] = []

    # Organize target images either from sample CSV or directory walk
    letter_image_map: Dict[str, List[Tuple[str, int, Path]]] = {}

    if sample_csv_path is not None:
        sample_csv_path = Path(sample_csv_path)
        if not sample_csv_path.exists():
            raise FileNotFoundError(f"Sample CSV file not found: {sample_csv_path.resolve()}")
        import pandas as pd
        sample_df = pd.read_csv(sample_csv_path)
        for _, row in sample_df.iterrows():
            l = str(row["letter"]).strip().upper()
            fn = str(row["filename"]).strip()
            eh = int(row.get("expected_hands", 1 if l in single_hand_set else 2))
            ip = dataset_dir / l / fn
            letter_image_map.setdefault(l, []).append((fn, eh, ip))
    else:
        folder_list = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        if letters:
            target_letters = {l.strip().upper() for l in letters}
            folder_list = [d for d in folder_list if d.name.upper() in target_letters]

        for folder in folder_list:
            l = folder.name.upper()
            eh = 1 if l in single_hand_set else 2
            image_files = sorted([
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            ])
            if max_per_letter is not None and max_per_letter > 0:
                image_files = image_files[:max_per_letter]
            letter_image_map[l] = [(f.name, eh, f) for f in image_files]

    total_images_scanned = 0
    total_successful = 0
    total_failed = 0

    try:
        for letter, items in letter_image_map.items():
            expected_hands = 1 if letter in single_hand_set else 2
            letter_total = len(items)
            letter_success = 0
            letter_failed = 0

            for filename, exp_h, img_path in items:
                total_images_scanned += 1
                member_id = get_member_id(filename)

                res, det_count, handedness_str, reason = process_single_image(
                    image_path=img_path,
                    detector=landmarker,
                    expected_hands=exp_h,
                    is_mirrored=is_mirrored,
                    min_confidence=0.0,
                    swap_unmirrored=swap_unmirrored,
                    preprocessing=preprocessing,
                    target_size=target_size,
                    upscale=upscale,
                )

                audit_records.append({
                    "letter": letter,
                    "filename": filename,
                    "member_id": member_id,
                    "expected_hands": exp_h,
                    "detected_hands": det_count,
                    "status": "SUCCESS" if reason == "OK" else "FAILED",
                    "reason": reason,
                    "handedness": handedness_str,
                })

                if reason == "OK" and res is not None and res.features is not None:
                    letter_success += 1
                    total_successful += 1

                    if exp_h == 1:
                        features_1h_list.append(res.features)
                        labels_1h_list.append(letter)
                        image_ids_1h_list.append(filename)
                    else:
                        features_2h_list.append(res.features)
                        rel_disp = res.relative_wrist_displacement
                        if rel_disp is None:
                            rel_disp = np.zeros(RELATIVE_WRIST_FEATURE_COUNT, dtype=np.float32)
                        rel_wrist_2h_list.append(rel_disp)
                        labels_2h_list.append(letter)
                        image_ids_2h_list.append(filename)
                else:
                    letter_failed += 1
                    total_failed += 1

            success_pct = (letter_success / letter_total * 100) if letter_total > 0 else 0.0
            per_letter_stats.append({
                "letter": letter,
                "expected_hands": expected_hands,
                "total": letter_total,
                "success": letter_success,
                "failed": letter_failed,
                "success_pct": round(success_pct, 1),
            })

            if verbose:
                print(
                    f"[{letter}] Total: {letter_total:2d} | "
                    f"Success: {letter_success:2d} | "
                    f"Failed: {letter_failed:2d} | "
                    f"Success Rate: {success_pct:5.1f}%"
                )

    finally:
        landmarker.close()

    # Convert to NumPy arrays
    features_1h = np.array(features_1h_list, dtype=np.float32) if features_1h_list else np.empty((0, ONE_HAND_FEATURE_COUNT), dtype=np.float32)
    labels_1h = np.array(labels_1h_list, dtype=str)
    image_ids_1h = np.array(image_ids_1h_list, dtype=str)

    features_2h = np.array(features_2h_list, dtype=np.float32) if features_2h_list else np.empty((0, TWO_HAND_FEATURE_COUNT), dtype=np.float32)
    rel_wrist_2h = np.array(rel_wrist_2h_list, dtype=np.float32) if rel_wrist_2h_list else np.empty((0, RELATIVE_WRIST_FEATURE_COUNT), dtype=np.float32)
    labels_2h = np.array(labels_2h_list, dtype=str)
    image_ids_2h = np.array(image_ids_2h_list, dtype=str)

    # Metadata dictionary
    overall_success_pct = (total_successful / total_images_scanned * 100) if total_images_scanned > 0 else 0.0
    metadata = {
        "dataset_name": "SignVisionAI Sri Lankan Sign Language (SLS)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mediapipe_model": str(model_path),
        "min_detection_confidence": min_conf,
        "is_mirrored": is_mirrored,
        "swap_unmirrored": swap_unmirrored,
        "preprocessing": {
            "method": preprocessing,
            "target_size": target_size if preprocessing != "none" else None,
            "upscale": upscale,
            "description": (
                "Aspect-ratio-preserving square padding + resize to target_size"
                if preprocessing == "square_pad"
                else "Original image unmodified (baseline)"
            ),
        },
        "normalization": "wrist_centered + palm_size_scaled",
        "reference_points": {
            "wrist_origin_index": 0,
            "middle_mcp_scale_index": 9,
        },
        "feature_dimensions": {
            "one_hand_baseline": ONE_HAND_FEATURE_COUNT,
            "two_hand_baseline": TWO_HAND_FEATURE_COUNT,
            "relative_wrist_displacement": RELATIVE_WRIST_FEATURE_COUNT,
            "two_hand_augmented": TWO_HAND_FEATURE_COUNT + RELATIVE_WRIST_FEATURE_COUNT,
        },
        "counts": {
            "total_scanned": total_images_scanned,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_pct": round(overall_success_pct, 2),
            "one_hand_samples": len(features_1h),
            "two_hand_samples": len(features_2h),
        },
        "per_letter_summary": per_letter_stats,
    }

    results_dict = {
        "features_1h": features_1h,
        "labels_1h": labels_1h,
        "image_ids_1h": image_ids_1h,
        "features_2h": features_2h,
        "relative_wrist_displacement_2h": rel_wrist_2h,
        "labels_2h": labels_2h,
        "image_ids_2h": image_ids_2h,
        "metadata_json": json.dumps(metadata, indent=2),
    }

    # Save NPZ if requested
    if output_npz_path:
        output_npz_path = Path(output_npz_path)
        output_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(output_npz_path),
            **results_dict,
        )
        if verbose:
            print(f"\n[Saved NPZ Output] -> {output_npz_path.resolve()}")

    # Save summary CSV if requested
    if summary_csv_path and audit_records:
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_records[0].keys()))
            writer.writeheader()
            writer.writerows(audit_records)
        if verbose:
            print(f"[Saved Audit CSV] -> {summary_csv_path.resolve()}")

    return results_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignVisionAI - SLS Landmark Extraction and Normalization Pipeline"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET_DIR, help="Path to SLS_Dataset directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to hand_landmarker.task")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_NPZ, help="Path to save normalized .npz")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV, help="Path to save audit CSV")
    parser.add_argument("--letters", type=str, default=None, help="Comma-separated letters to process (e.g. C,A,B)")
    parser.add_argument("--single-hand-letters", type=str, default="C", help="Letters expecting 1 hand (default: C)")
    parser.add_argument("--max-per-letter", type=int, default=None, help="Max images per letter (useful for pilot)")
    parser.add_argument("--pilot", action="store_true", help="Run quick pilot test on C, A, B with 5 images each")
    parser.add_argument("--min-conf", type=float, default=0.3, help="Min detection confidence (default: 0.3)")
    parser.add_argument("--mirrored", action="store_true", help="Set if dataset images are mirrored (default: False)")
    parser.add_argument("--swap-unmirrored", action="store_true", help="Swap Left/Right on unmirrored images")
    parser.add_argument(
        "--preprocessing",
        type=str,
        choices=["none", "square_pad"],
        default="none",
        help="Preprocessing strategy: 'none' (baseline) or 'square_pad' (aspect-preserving padding + resize)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Target square dimension for padding (default: {DEFAULT_TARGET_SIZE})",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Whether to upscale small images (< target-size) during square padding",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=None,
        help="Optional path to sample CSV (e.g. results/evaluation_sample_96.csv) to process only specified images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    letters = None
    max_per_letter = args.max_per_letter

    if args.pilot:
        print("=== RUNNING PILOT EXTRACTION TEST ===")
        letters = ["C", "A", "B"]
        if max_per_letter is None:
            max_per_letter = 5
        output_npz = Path("data/sls_pilot_normalized.npz")
        summary_csv = Path("results/pilot_extraction_summary.csv")
    else:
        if args.letters:
            letters = [l.strip().upper() for l in args.letters.split(",") if l.strip()]
        output_npz = args.output
        summary_csv = args.summary_csv

    single_hand_list = [l.strip().upper() for l in args.single_hand_letters.split(",") if l.strip()]

    print(f"Input Dataset  : {args.input.resolve()}")
    print(f"MediaPipe Model: {args.model.resolve()}")
    print(f"Preprocessing  : {args.preprocessing} (target_size={args.target_size}, upscale={args.upscale})")
    print(f"Sample CSV     : {args.sample_csv.resolve() if args.sample_csv else 'None (Full directory)'}")
    print(f"Target Letters : {letters if letters else 'ALL (A-Z)'}")
    print(f"Max Per Letter : {max_per_letter if max_per_letter else 'ALL'}")
    print(f"Single-Hand (1): {single_hand_list}")
    print("=" * 60)

    results = extract_dataset_landmarks(
        dataset_dir=args.input,
        model_path=args.model,
        output_npz_path=output_npz,
        summary_csv_path=summary_csv,
        letters=letters,
        single_hand_letters=single_hand_list,
        max_per_letter=max_per_letter,
        is_mirrored=args.mirrored,
        min_conf=args.min_conf,
        swap_unmirrored=args.swap_unmirrored,
        preprocessing=args.preprocessing,
        target_size=args.target_size,
        upscale=args.upscale,
        sample_csv_path=args.sample_csv,
        verbose=True,
    )

    meta = json.loads(results["metadata_json"])
    print("\n" + "=" * 60)
    print("           EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total Images Audited : {meta['counts']['total_scanned']}")
    print(f"Total Successful     : {meta['counts']['total_successful']}")
    print(f"Total Failed         : {meta['counts']['total_failed']}")
    print(f"Overall Success Rate : {meta['counts']['overall_success_pct']}%")
    print(f"One-Hand Samples (C) : {meta['counts']['one_hand_samples']} (Shape: {results['features_1h'].shape})")
    print(f"Two-Hand Samples     : {meta['counts']['two_hand_samples']} (Shape: {results['features_2h'].shape})")
    print(f"Relative Wrist Disp  : Shape: {results['relative_wrist_displacement_2h'].shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()

