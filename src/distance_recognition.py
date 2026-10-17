"""
SignVisionAI - Distance-Based Recognition Module
Part 3: Distance-Based Recognition (1-NN & KNN)

This module implements non-parametric, distance-based recognition for Sri Lankan Sign
Language (SLS) using normalized MediaPipe hand landmarks.

Key Capabilities:
1. Pure Euclidean Distance:
   Computes exact Euclidean distance d(x, y) = sqrt(sum((x_i - y_i)^2)) using vectorized
   NumPy operations for sub-millisecond real-time webcam inference.

2. 1-Nearest Neighbour (1-NN) & K-Nearest Neighbours (KNN):
   Finds the k closest reference samples, applies majority voting, and breaks ties
   deterministically based on smallest distance. Configurable for k=1, k=3, k=5, etc.

3. Strict Hand-Count Architecture (One-Hand vs Two-Hand):
   - SLS Letter 'C': Single-hand sign represented by exactly 63 normalized coordinates.
   - All other 25 letters (A, B, D..Z): Two-hand signs represented by 126 normalized coordinates
     (ordered strictly Left hand 0..62, Right hand 63..125) or optional 129 augmented coordinates.
   - Separate reference libraries prevent cross-dimensional vector comparisons and eliminate
     the need for harmful zero-padding.

4. Validation-Calibrated Uncertainty Rejection:
   Rejects out-of-distribution gestures or false hand detections when nearest-neighbour
   distance exceeds a configurable threshold, returning status="uncertain" and letter=None.
   Maintains separate thresholds for one-hand and two-hand signs due to dimensional scaling.

5. Seamless Pipeline Integration:
   Accepts raw NumPy feature vectors with hand_count, or HandFeatureResult dataclasses
   directly from src.normalization.format_hand_features, enabling instant Part 4 webcam integration.
"""

import sys
import mediapipe.tasks.python.vision.drawing_styles
from unittest import async_case
from mediapipe.tasks.python.vision import drawing_styles
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Feature dimensions established by Part 2
ONE_HAND_FEATURE_COUNT: int = 63
TWO_HAND_FEATURE_COUNT: int = 126
AUGMENTED_TWO_HAND_FEATURE_COUNT: int = 129
VALID_HAND_COUNTS: Tuple[int, ...] = (1, 2)

DEFAULT_DATASET_NPZ: Path = Path("data/sls_normalized_square_pad.npz")


# =============================================================================
# Core Pure Mathematical Distance Functions
# =============================================================================

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the standard Euclidean distance between two 1D landmark feature vectors:
        d(x, y) = sqrt(sum((x_i - y_i)^2))

    Args:
        x: First feature vector (1D float array).
        y: Second feature vector (1D float array).

    Returns:
        float: Non-negative Euclidean distance.

    Raises:
        ValueError: If vectors have different shapes, are not 1D, or contain NaN/Inf.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"Inputs must be 1D vectors, got shapes x={x.shape}, y={y.shape}.")

    if x.shape != y.shape:
        raise ValueError(f"Vector dimensions do not match: {x.shape} vs {y.shape}.")

    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Input vectors contain NaN or non-finite values.")

    diff = x - y
    return float(np.sqrt(np.dot(diff, diff)))


def compute_distances(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances from a single query vector to an array of reference vectors
    using vectorized NumPy matrix operations.

    Args:
        query: 1D NumPy array of shape (D,).
        references: 2D NumPy array of shape (N, D).

    Returns:
        1D NumPy array of shape (N,) containing Euclidean distances.

    Raises:
        ValueError: If query or reference dimensions are invalid or contain non-finite values.
    """
    if query.ndim != 1:
        raise ValueError(f"Query must be a 1D vector of shape (D,), got shape {query.shape}.")
    if references.ndim != 2:
        raise ValueError(f"References must be a 2D matrix of shape (N, D), got shape {references.shape}.")

    n_samples, dim = references.shape
    if query.shape[0] != dim:
        raise ValueError(
            f"Dimension mismatch between query ({query.shape[0]}) and references ({dim})."
        )

    if n_samples == 0:
        return np.empty((0,), dtype=np.float32)

    if not np.isfinite(query).all():
        raise ValueError("Query contains NaN or non-finite values.")

    diff = references - query
    # Numerically stable Euclidean distance along rows
    distances = np.linalg.norm(diff, axis=1)
    return distances.astype(np.float32)


# =============================================================================
# Structured Prediction Output
# =============================================================================

@dataclass(frozen=True)
class RecognitionResult:
    """
    Structured outcome of a distance-based sign recognition query.

    Attributes:
        letter: Recognized SLS letter (e.g. 'A', 'C'), or None if uncertain / invalid.
        status: Recognition status:
                - 'recognized': Valid prediction with distance <= threshold.
                - 'uncertain': Distance exceeded threshold or confidence too low.
                - 'invalid_input': Malformed features, unsupported hand count, or missing landmarks.
        distance: Euclidean distance to the closest reference sample (float or None).
        method: Recognition method used ('1nn' or 'knn').
        k: Value of k utilized for the query.
        hand_count: Number of hands processed (1 or 2, or None if invalid).
        confidence: Ratio of majority votes to total k (1.0 for 1-NN).
        threshold: Uncertainty distance threshold applied, or None if unconstrained.
        top_k: List of dicts describing the top k nearest neighbors.
        votes: Dictionary mapping candidate letters to their vote count.
        error_message: Explanatory error details if status == 'invalid_input'.
    """
    letter: Optional[str]
    status: str
    distance: Optional[float]
    method: str
    k: int
    hand_count: Optional[int]
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    top_k: Optional[List[Dict[str, Any]]] = None
    votes: Optional[Dict[str, int]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result dataclass to a JSON-serializable dictionary."""
        return asdict(self)


# =============================================================================
# Distance-Based Recognizer (1-NN and KNN)
# =============================================================================

class DistanceRecognizer:
    """
    Distance-based alphabet recognizer for Sri Lankan Sign Language (SLS).

    Stores an in-memory labeled reference library partitioned into:
    - 1-Hand Library: Exclusively for Letter 'C' (63 features).
    - 2-Hand Library: For letters A, B, D..Z (126 baseline features or 129 augmented).

    Supports:
    - 1-Nearest Neighbour (1-NN)
    - K-Nearest Neighbours (KNN) with majority voting and deterministic tie-breaking
    - Separate, validation-calibrated uncertainty thresholds for 1-hand and 2-hand gestures
    - Safe handling of missing hands, non-finite values, and malformed inputs
    """

    def __init__(
        self,
        reference_path: Optional[Union[str, Path]] = None,
        k: int = 3,
        threshold_1h: Optional[float] = None,
        threshold_2h: Optional[float] = None,
        use_augmented_2h: bool = False,
    ):
        """
        Initialize the recognizer.

        Args:
            reference_path: Optional path to .npz file containing normalized features.
                            If provided, automatically loads references.
            k: Default number of nearest neighbours (must be positive integer >= 1).
            threshold_1h: Maximum distance threshold for 1-hand predictions (rejects if > threshold).
            threshold_2h: Maximum distance threshold for 2-hand predictions (rejects if > threshold).
            use_augmented_2h: If True, uses 129-dim augmented features (126 baseline + 3 rel wrist)
                              for two-hand gestures when available.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")

        self.k: int = k
        self.threshold_1h: Optional[float] = float(threshold_1h) if threshold_1h is not None else None
        self.threshold_2h: Optional[float] = float(threshold_2h) if threshold_2h is not None else None
        self.use_augmented_2h: bool = bool(use_augmented_2h)

        # 1-Hand Reference Library (Letter C)
        self._features_1h: Optional[np.ndarray] = None
        self._labels_1h: Optional[np.ndarray] = None
        self._image_ids_1h: Optional[np.ndarray] = None

        # 2-Hand Reference Library (A, B, D..Z)
        self._features_2h: Optional[np.ndarray] = None
        self._relative_wrist_2h: Optional[np.ndarray] = None
        self._labels_2h: Optional[np.ndarray] = None
        self._image_ids_2h: Optional[np.ndarray] = None

        # Load reference dataset if path specified
        if reference_path is not None:
            self.load_reference_library(reference_path)

    # -------------------------------------------------------------------------
    # Reference Library Management
    # -------------------------------------------------------------------------

    def set_references_1h(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        image_ids: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register reference samples for 1-hand signs (Letter C).

        Args:
            features: 2D array of shape (N, 63) with float32 coordinates.
            labels: 1D array of shape (N,) containing sign labels (e.g. 'C').
            image_ids: Optional 1D array of shape (N,) with source image filenames.
        """
        features_arr = np.asarray(features, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=str)

        if features_arr.ndim != 2 or features_arr.shape[1] != ONE_HAND_FEATURE_COUNT:
            raise ValueError(
                f"1-hand features must have shape (N, {ONE_HAND_FEATURE_COUNT}), got {features_arr.shape}."
            )
        if labels_arr.ndim != 1 or len(labels_arr) != len(features_arr):
            raise ValueError("1-hand labels length must match features count.")
        if not np.isfinite(features_arr).all():
            raise ValueError("1-hand reference features contain NaN or non-finite values.")

        self._features_1h = features_arr
        self._labels_1h = labels_arr
        self._image_ids_1h = np.asarray(image_ids, dtype=str) if image_ids is not None else None

    def set_references_2h(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        image_ids: Optional[np.ndarray] = None,
        relative_wrist: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register reference samples for 2-hand signs (A, B, D..Z).

        Args:
            features: 2D array of shape (N, 126) with float32 coordinates.
            labels: 1D array of shape (N,) containing sign labels.
            image_ids: Optional 1D array of shape (N,) with source image filenames.
            relative_wrist: Optional 2D array of shape (N, 3) for augmented representation.
        """
        features_arr = np.asarray(features, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=str)

        expected_dim = (
            AUGMENTED_TWO_HAND_FEATURE_COUNT if self.use_augmented_2h and features_arr.shape[1] == AUGMENTED_TWO_HAND_FEATURE_COUNT
            else TWO_HAND_FEATURE_COUNT
        )

        if features_arr.ndim != 2 or features_arr.shape[1] not in (TWO_HAND_FEATURE_COUNT, AUGMENTED_TWO_HAND_FEATURE_COUNT):
            raise ValueError(
                f"2-hand features must have shape (N, {TWO_HAND_FEATURE_COUNT}) or "
                f"(N, {AUGMENTED_TWO_HAND_FEATURE_COUNT}), got {features_arr.shape}."
            )
        if labels_arr.ndim != 1 or len(labels_arr) != len(features_arr):
            raise ValueError("2-hand labels length must match features count.")
        if not np.isfinite(features_arr).all():
            raise ValueError("2-hand reference features contain NaN or non-finite values.")

        self._features_2h = features_arr
        self._labels_2h = labels_arr
        self._image_ids_2h = np.asarray(image_ids, dtype=str) if image_ids is not None else None
        self._relative_wrist_2h = np.asarray(relative_wrist, dtype=np.float32) if relative_wrist is not None else None

    def load_reference_library(self, npz_path: Union[str, Path]) -> None:
        """
        Load reference samples directly from an extracted .npz file (e.g. data/sls_normalized_square_pad.npz).

        Args:
            npz_path: Path to the .npz archive.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If required arrays are missing in the archive.
        """
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference dataset file not found at: {path.resolve()}")

        archive = np.load(path, allow_pickle=True)

        if "features_1h" not in archive or "labels_1h" not in archive:
            raise KeyError(f"Missing 'features_1h' or 'labels_1h' in {path}")
        if "features_2h" not in archive or "labels_2h" not in archive:
            raise KeyError(f"Missing 'features_2h' or 'labels_2h' in {path}")

        # Set 1-hand library
        self.set_references_1h(
            features=archive["features_1h"],
            labels=archive["labels_1h"],
            image_ids=archive["image_ids_1h"] if "image_ids_1h" in archive else None,
        )

        # Set 2-hand library (with optional augmentation support)
        f2 = archive["features_2h"]
        rw2 = archive["relative_wrist_displacement_2h"] if "relative_wrist_displacement_2h" in archive else None

        if self.use_augmented_2h and rw2 is not None:
            f2_aug = np.concatenate([f2, rw2], axis=1).astype(np.float32)
            self.set_references_2h(
                features=f2_aug,
                labels=archive["labels_2h"],
                image_ids=archive["image_ids_2h"] if "image_ids_2h" in archive else None,
                relative_wrist=rw2,
            )
        else:
            self.set_references_2h(
                features=f2,
                labels=archive["labels_2h"],
                image_ids=archive["image_ids_2h"] if "image_ids_2h" in archive else None,
                relative_wrist=rw2,
            )

    @property
    def sample_counts(self) -> Dict[str, int]:
        """Return the number of registered reference samples."""
        return {
            "one_hand": len(self._features_1h) if self._features_1h is not None else 0,
            "two_hand": len(self._features_2h) if self._features_2h is not None else 0,
        }

    # -------------------------------------------------------------------------
    # Core Recognition & Prediction Logic
    # -------------------------------------------------------------------------

    def predict(
        self,
        input_data: Any,
        hand_count: Optional[int] = None,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Recognize an SLS sign from normalized landmarks.

        Accepts:
        - HandFeatureResult dataclass from src.normalization.format_hand_features
        - 1D NumPy array with explicit hand_count parameter (1 or 2)
        - Dictionary containing 'features' and 'hand_count'

        Args:
            input_data: Features array, HandFeatureResult object, or dict.
            hand_count: Number of hands detected (1 or 2). Required if input_data is a raw array.
            k: Optional override for number of nearest neighbours.
            threshold: Optional override for rejection distance threshold.

        Returns:
            Dictionary with standardized schema:
            {
                "letter": "A",          # Predicted letter or None
                "status": "recognized", # "recognized", "uncertain", or "invalid_input"
                "distance": 0.123,      # Distance to nearest sample or None
                "method": "knn",        # "1nn" or "knn"
                "k": 3,
                "hand_count": 2,
                "confidence": 0.67,
                "threshold": 1.25,
                "top_k": [...],
                "votes": {...},
                "error_message": None
            }
        """
        k_val = k if k is not None else self.k
        method_str = "1nn" if k_val == 1 else "knn"

        # Validate k
        if not isinstance(k_val, int) or k_val <= 0:
            return RecognitionResult(
                letter=None,
                status="invalid_input",
                distance=None,
                method=method_str,
                k=k_val,
                hand_count=hand_count,
                error_message=f"k must be a positive integer, got {k_val}.",
            ).to_dict()

        # Step 1: Extract features and hand_count from polymorphic inputs
        features, resolved_hand_count, extract_err = self._extract_features_and_hand_count(
            input_data, hand_count
        )
        if extract_err is not None:
            return RecognitionResult(
                letter=None,
                status="invalid_input",
                distance=None,
                method=method_str,
                k=k_val,
                hand_count=resolved_hand_count,
                error_message=extract_err,
            ).to_dict()

        # Step 2: Validate feature dimensions against detected hand count
        val_err = self._validate_input_dimensions(features, resolved_hand_count)
        if val_err is not None:
            return RecognitionResult(
                letter=None,
                status="invalid_input",
                distance=None,
                method=method_str,
                k=k_val,
                hand_count=resolved_hand_count,
                error_message=val_err,
            ).to_dict()

        # Step 3: Select compatible reference library
        ref_features, ref_labels, ref_image_ids, lib_err = self._get_compatible_references(
            resolved_hand_count, features.shape[0]
        )
        if lib_err is not None:
            return RecognitionResult(
                letter=None,
                status="invalid_input",
                distance=None,
                method=method_str,
                k=k_val,
                hand_count=resolved_hand_count,
                error_message=lib_err,
            ).to_dict()

        # Check k against reference library size
        n_ref = len(ref_features)
        if k_val > n_ref:
            return RecognitionResult(
                letter=None,
                status="invalid_input",
                distance=None,
                method=method_str,
                k=k_val,
                hand_count=resolved_hand_count,
                error_message=f"k={k_val} exceeds available reference samples ({n_ref}).",
            ).to_dict()

        # Step 4: Compute Euclidean distances to all compatible reference samples
        try:
            distances = compute_distances(features, ref_features)
        except Exception as exc:
            return RecognitionResult(
                letter=None,
                status="invalid_input",
                distance=None,
                method=method_str,
                k=k_val,
                hand_count=resolved_hand_count,
                error_message=f"Distance computation failed: {exc}",
            ).to_dict()

        # Step 5: Find the k closest reference samples (sorted by ascending distance)
        # Using np.argsort gives exact ascending distance order
        top_k_indices = np.argsort(distances)[:k_val]
        top_k_distances = [float(distances[idx]) for idx in top_k_indices]
        top_k_labels = [str(ref_labels[idx]) for idx in top_k_indices]

        nearest_distance = top_k_distances[0]

        # Build top-k metadata list
        top_k_info: List[Dict[str, Any]] = []
        for rank, (idx, dist, lbl) in enumerate(zip(top_k_indices, top_k_distances, top_k_labels)):
            item: Dict[str, Any] = {
                "rank": rank + 1,
                "letter": lbl,
                "distance": round(dist, 5),
                "index": int(idx),
            }
            if ref_image_ids is not None and idx < len(ref_image_ids):
                item["image_id"] = str(ref_image_ids[idx])
            top_k_info.append(item)

        # Step 6: Majority Voting & Deterministic Tie-Breaking
        votes: Dict[str, int] = {}
        for lbl in top_k_labels:
            votes[lbl] = votes.get(lbl, 0) + 1

        if k_val == 1:
            predicted_letter = top_k_labels[0]
            confidence = 1.0
        else:
            max_votes = max(votes.values())
            tied_candidates = [lbl for lbl, count in votes.items() if count == max_votes]

            if len(tied_candidates) == 1:
                predicted_letter = tied_candidates[0]
            else:
                # Deterministic tie-breaking:
                # The tied candidate that has the smallest minimum distance wins.
                # Since top_k_labels is sorted ascending by distance, this is simply
                # the first tied candidate encountered in top_k_labels!
                for candidate in top_k_labels:
                    if candidate in tied_candidates:
                        predicted_letter = candidate
                        break

            confidence = round(float(votes[predicted_letter]) / float(k_val), 4)

        # Step 7: Uncertainty / Distance Threshold Rejection
        active_threshold = threshold
        if active_threshold is None:
            active_threshold = (
                self.threshold_1h if resolved_hand_count == 1 else self.threshold_2h
            )

        if active_threshold is not None and nearest_distance > active_threshold:
            status = "uncertain"
            predicted_letter = None
        else:
            status = "recognized"

        return RecognitionResult(
            letter=predicted_letter,
            status=status,
            distance=round(nearest_distance, 5),
            method=method_str,
            k=k_val,
            hand_count=resolved_hand_count,
            confidence=confidence,
            threshold=active_threshold,
            top_k=top_k_info,
            votes=votes,
            error_message=None,
        ).to_dict()

    # -------------------------------------------------------------------------
    # Batch Prediction & Evaluation Helper
    # -------------------------------------------------------------------------

    def predict_batch(
        self,
        features_list: Union[List[np.ndarray], np.ndarray],
        hand_count: int,
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Efficiently predict a batch of queries with the same hand count.

        Args:
            features_list: 2D array of shape (M, D) or list of 1D arrays.
            hand_count: 1 or 2.
            k: Optional k override.
            threshold: Optional threshold override.

        Returns:
            List of prediction result dictionaries.
        """
        arr = np.asarray(features_list, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        results: List[Dict[str, Any]] = []
        for i in range(len(arr)):
            res = self.predict(
                input_data=arr[i],
                hand_count=hand_count,
                k=k,
                threshold=threshold,
            )
            results.append(res)
        return results

    # -------------------------------------------------------------------------
    # Internal Validation & Extraction Helpers
    # -------------------------------------------------------------------------

    def _extract_features_and_hand_count(
        self,
        input_data: Any,
        explicit_hand_count: Optional[int],
    ) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
        """Extract flat float32 feature array and hand_count from polymorphic inputs."""
        if input_data is None:
            return None, explicit_hand_count, "Input features cannot be None."

        # Case A: HandFeatureResult from src.normalization
        if hasattr(input_data, "is_valid") and hasattr(input_data, "features"):
            if not input_data.is_valid or input_data.features is None:
                err = getattr(input_data, "error_message", None) or getattr(
                    input_data, "status", "Invalid HandFeatureResult"
                )
                return None, getattr(input_data, "hand_count", explicit_hand_count), f"HandFeatureResult is invalid: {err}"

            hand_cnt = getattr(input_data, "hand_count", explicit_hand_count)
            if self.use_augmented_2h and hand_cnt == 2 and hasattr(input_data, "get_augmented_features"):
                feat = input_data.get_augmented_features()
            else:
                feat = input_data.features
            if feat is not None and feat.ndim > 1:
                feat = feat.flatten()
            return feat, hand_cnt, None

        # Case B: Dictionary with 'features' and optional 'hand_count'
        if isinstance(input_data, dict):
            feat_val = input_data.get("features")
            if feat_val is None:
                return None, explicit_hand_count, "Input dictionary missing 'features' key."
            dict_hand_count = input_data.get("hand_count", explicit_hand_count)
            try:
                feat = np.asarray(feat_val, dtype=np.float32)
            except Exception as exc:
                return None, dict_hand_count, f"Cannot convert dictionary features to array: {exc}"
            if feat.ndim > 1:
                feat = feat.flatten()
            return feat, dict_hand_count, None

        # Case C: NumPy array or array-like
        try:
            feat = np.asarray(input_data, dtype=np.float32)
        except Exception as exc:
            return None, explicit_hand_count, f"Cannot convert input data to float32 NumPy array: {exc}"
        if feat.ndim > 1:
            feat = feat.flatten()
        return feat, explicit_hand_count, None

    def _validate_input_dimensions(
        self,
        features: np.ndarray,
        hand_count: Optional[int],
    ) -> Optional[str]:
        """Validate feature dimensions and ensure no NaN/Inf values exist."""
        if hand_count is None:
            return "hand_count must be explicitly provided (1 or 2) when passing raw feature arrays."

        if hand_count not in VALID_HAND_COUNTS:
            return f"Unsupported hand_count {hand_count}. Must be 1 or 2."

        if not np.isfinite(features).all():
            return "Input features contain NaN or non-finite values."

        # Handle auto-flattening of (21, 3) or (42, 3) if provided
        flat_dim = features.size
        if features.ndim > 1:
            features = features.flatten()

        if hand_count == 1:
            if flat_dim != ONE_HAND_FEATURE_COUNT:
                return (
                    f"Dimension mismatch for 1-hand gesture: expected {ONE_HAND_FEATURE_COUNT} "
                    f"features (21 landmarks x 3), but got {flat_dim}."
                )
        elif hand_count == 2:
            expected_dims = (
                (TWO_HAND_FEATURE_COUNT, AUGMENTED_TWO_HAND_FEATURE_COUNT)
                if self.use_augmented_2h
                else (TWO_HAND_FEATURE_COUNT,)
            )
            if flat_dim not in expected_dims:
                return (
                    f"Dimension mismatch for 2-hand gesture: expected {expected_dims} "
                    f"features, but got {flat_dim}."
                )

        return None

    def _get_compatible_references(
        self,
        hand_count: int,
        feature_dim: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """Retrieve the compatible reference subset for the query's hand count."""
        if hand_count == 1:
            if self._features_1h is None or len(self._features_1h) == 0:
                return None, None, None, "1-hand reference library is empty or not loaded."
            if self._features_1h.shape[1] != feature_dim:
                return None, None, None, (
                    f"Feature dimension {feature_dim} does not match 1-hand reference "
                    f"dimension {self._features_1h.shape[1]}."
                )
            return self._features_1h, self._labels_1h, self._image_ids_1h, None

        elif hand_count == 2:
            if self._features_2h is None or len(self._features_2h) == 0:
                return None, None, None, "2-hand reference library is empty or not loaded."
            if self._features_2h.shape[1] != feature_dim:
                return None, None, None, (
                    f"Feature dimension {feature_dim} does not match 2-hand reference "
                    f"dimension {self._features_2h.shape[1]}."
                )
            return self._features_2h, self._labels_2h, self._image_ids_2h, None

        return None, None, None, f"Unsupported hand_count {hand_count}."


# =============================================================================
# Threshold Calibration Helper
# =============================================================================

def calibrate_distance_thresholds(
    val_features_1h: Optional[np.ndarray],
    ref_features_1h: Optional[np.ndarray],
    val_features_2h: Optional[np.ndarray],
    ref_features_2h: Optional[np.ndarray],
    percentile: float = 95.0,
    k: int = 1,
) -> Dict[str, Any]:
    """
    Calibrate empirical distance thresholds for 1-hand and 2-hand signs using validation data.

    Calculates the distribution of nearest-neighbour distances on the validation set against
    the reference set, setting the threshold at the specified percentile (e.g. 95th or 99th).

    Notice:
    1-hand signs (Letter C) have only 1 valid class in SLS. Therefore, false-positive rejection
    against unrelated 1-hand non-SLS gestures cannot be calibrated without external negative samples.
    The 1-hand threshold is thus calibrated as the empirical upper bound of valid intra-class variation.

    Args:
        val_features_1h: Validation features for 1-hand signs (M1, 63).
        ref_features_1h: Reference features for 1-hand signs (N1, 63).
        val_features_2h: Validation features for 2-hand signs (M2, 126).
        ref_features_2h: Reference features for 2-hand signs (N2, 126).
        percentile: Percentile cutoff (default: 95.0, range 0..100).
        k: Neighbour index to calibrate against (default 1).

    Returns:
        Dictionary containing calibrated thresholds and summary statistics.
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"Percentile must be between 0.0 and 100.0, got {percentile}.")

    calibration_result: Dict[str, Any] = {
        "percentile": percentile,
        "k": k,
        "threshold_1h": None,
        "threshold_2h": None,
        "stats_1h": {},
        "stats_2h": {},
    }

    # Calibrate 1-hand threshold
    if val_features_1h is not None and ref_features_1h is not None and len(val_features_1h) > 0 and len(ref_features_1h) > 0:
        dists_1h = []
        for feat in val_features_1h:
            d = compute_distances(feat, ref_features_1h)
            d_sorted = np.sort(d)
            k_idx = min(k - 1, len(d_sorted) - 1)
            dists_1h.append(float(d_sorted[k_idx]))

        dists_1h_arr = np.array(dists_1h)
        thresh_1h = float(np.percentile(dists_1h_arr, percentile))
        calibration_result["threshold_1h"] = round(thresh_1h, 5)
        calibration_result["stats_1h"] = {
            "count": len(dists_1h_arr),
            "min": round(float(np.min(dists_1h_arr)), 5),
            "median": round(float(np.median(dists_1h_arr)), 5),
            "mean": round(float(np.mean(dists_1h_arr)), 5),
            "percentile_val": round(thresh_1h, 5),
            "max": round(float(np.max(dists_1h_arr)), 5),
        }

    # Calibrate 2-hand threshold
    if val_features_2h is not None and ref_features_2h is not None and len(val_features_2h) > 0 and len(ref_features_2h) > 0:
        dists_2h = []
        for feat in val_features_2h:
            d = compute_distances(feat, ref_features_2h)
            d_sorted = np.sort(d)
            k_idx = min(k - 1, len(d_sorted) - 1)
            dists_2h.append(float(d_sorted[k_idx]))

        dists_2h_arr = np.array(dists_2h)
        thresh_2h = float(np.percentile(dists_2h_arr, percentile))
        calibration_result["threshold_2h"] = round(thresh_2h, 5)
        calibration_result["stats_2h"] = {
            "count": len(dists_2h_arr),
            "min": round(float(np.min(dists_2h_arr)), 5),
            "median": round(float(np.median(dists_2h_arr)), 5),
            "mean": round(float(np.mean(dists_2h_arr)), 5),
            "percentile_val": round(thresh_2h, 5),
            "max": round(float(np.max(dists_2h_arr)), 5),
        }

    return calibration_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SignVisionAI Distance-Based Recognition Demo")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbours (default: 3)")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_NPZ), help="Path to normalized .npz dataset")
    parser.add_argument("--test-sample", action="store_true", help="Run a test prediction on sample dataset items")
    args = parser.parse_args()

    print("=" * 70)
    print("SignVisionAI - Distance-Based Recognition Module (Part 3)")
    print("=" * 70)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset archive not found at {dataset_path.resolve()}")
        sys.exit(1)

    print(f"Loading reference dataset: {dataset_path}")
    recognizer = DistanceRecognizer(reference_path=dataset_path, k=args.k)
    counts = recognizer.sample_counts
    print(f"Loaded {counts['one_hand']} single-hand references (Letter C)")
    print(f"Loaded {counts['two_hand']} two-hand references (Letters A, B, D..Z)")

    archive = np.load(dataset_path)

    # Demo 1-hand
    print("\n--- Testing Single-Hand Query (Letter C) ---")
    feat_1h = archive["features_1h"][0]
    res_1h = recognizer.predict(feat_1h, hand_count=1)
    print(f"Status    : {res_1h['status']}")
    print(f"Letter    : {res_1h['letter']}")
    print(f"Distance  : {res_1h['distance']}")
    print(f"Top 3     : {res_1h['top_k']}")

    # Demo 2-hand
    print("\n--- Testing Two-Hand Query ---")
    feat_2h = archive["features_2h"][0]
    expected_lbl = str(archive["labels_2h"][0])
    res_2h = recognizer.predict(feat_2h, hand_count=2)
    print(f"Ground Truth: {expected_lbl}")
    print(f"Status      : {res_2h['status']}")
    print(f"Predicted   : {res_2h['letter']}")
    print(f"Distance    : {res_2h['distance']}")
    print(f"Votes       : {res_2h['votes']}")

    print("\n" + "=" * 70)
    print("Module is ready for Part 4 webcam integration!")
    print("=" * 70)
