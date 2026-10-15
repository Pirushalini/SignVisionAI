"""
SignVisionAI - Landmark Normalization Module
Part 2: Landmark Extraction + Normalization

This module provides reusable, NumPy-based functions to normalize 3D hand landmarks
extracted from MediaPipe (using normalized image coordinates x, y, z) and format them
consistently for Sri Lankan Sign Language (SLS) recognition.

Key Steps & Features:
1. Wrist-Centered Translation:
   Moves the wrist (landmark 0) to the coordinate origin (0, 0, 0).
   This makes the hand's landmark coordinates independent of where the hand
   appears on the camera screen.

2. Palm-Size Scale Normalization:
   Divides the wrist-centered coordinates by the Euclidean distance between
   the wrist (landmark 0) and the middle finger MCP joint (landmark 9).
   This makes hand landmark distances independent of camera distance.

3. Consistent Hand Ordering & Separate Representations:
   - One-hand signs (e.g. Letter 'C'): Exactly 63 float32 features.
   - Two-hand signs (all other SLS letters): Exactly 126 float32 features,
     strictly ordered as Left Hand features (0..62) followed by Right Hand (63..125).
   - Invariance to detection order: Whether MediaPipe detects [Left, Right]
     or [Right, Left], the output feature vector is consistently ordered.
   - Separate representations: Missing hands are NEVER zero-padded to prevent
     distorting Euclidean / KNN distance metrics.

4. Relative Wrist Displacement (Preserving Inter-Hand Geometry):
   - Individual wrist-centering places both wrists at (0, 0, 0), removing global
     inter-hand spatial relationship (e.g., hand distance, vertical/horizontal offset).
   - To preserve this geometry without altering the 126-feature baseline, the relative
     displacement vector between the two wrists (Right wrist - Left wrist) is computed
     from raw un-centered coordinates, normalized by the average palm scale, and stored
     separately as a 3D vector (shape (3,)).
   - This allows Part 3 to benchmark the pure 126-feature baseline against an augmented
     129-feature representation (126 + 3).

5. MediaPipe Handedness & Orientation Conventions:
   - Official MediaPipe Rule: MediaPipe HandLandmarker internally assumes images are
     mirrored (i.e. taken with a front-facing selfie camera with horizontal flip).
   - In mirrored webcam streams (cv2.flip(frame, 1)):
     The image matches MediaPipe's mirror assumption.
   - In un-mirrored images (e.g. third-person dataset photos):
     MediaPipe's output corresponds to the viewer/image perspective (hand on image-left
     is labeled 'Left', hand on image-right is labeled 'Right').
   - Configurable orientation policy: `is_mirrored` and `swap_unmirrored` flags ensure
     the pipeline can enforce consistent anatomical mapping across dataset and webcam.

6. Strict Validation & Ambiguity Rejection:
   - Rejects unsupported hand counts (0 or >2 hands).
   - Rejects ambiguous duplicate labels (e.g. both hands labeled 'Right').
   - Rejects uncertain or low-confidence handedness rather than guessing.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import numpy as np

# Total landmarks per hand in standard MediaPipe models
NUM_LANDMARKS: int = 21
NUM_COORDINATES: int = 3
TOTAL_FEATURES_PER_HAND: int = NUM_LANDMARKS * NUM_COORDINATES  # 63
ONE_HAND_FEATURE_COUNT: int = TOTAL_FEATURES_PER_HAND          # 63
TWO_HAND_FEATURE_COUNT: int = TOTAL_FEATURES_PER_HAND * 2      # 126
RELATIVE_WRIST_FEATURE_COUNT: int = 3                          # (dx, dy, dz)
AUGMENTED_TWO_HAND_FEATURE_COUNT: int = TWO_HAND_FEATURE_COUNT + RELATIVE_WRIST_FEATURE_COUNT  # 129

# Landmark indices defined by MediaPipe Hand model
WRIST_INDEX: int = 0
MIDDLE_FINGER_MCP_INDEX: int = 9

# Small epsilon value to avoid division by zero or near-zero scales
DEFAULT_EPSILON: float = 1e-6

# Supported handedness labels
VALID_HANDEDNESS_LABELS = {"Left", "Right"}


class HandednessError(ValueError):
    """Raised when hand count or handedness cannot be reliably resolved."""
    pass


@dataclass(frozen=True)
class HandFeatureResult:
    """
    Structured result containing normalized hand landmarks and metadata.

    Attributes:
        features: 1D NumPy array of float32 values.
                  Shape is (63,) for one-hand signs, or (126,) for two-hand signs.
                  None if the detection is invalid.
        hand_count: Number of hands detected and resolved (1 or 2).
        resolved_handedness: Tuple of resolved handedness strings (e.g. ('Left', 'Right') or ('Right',)).
        is_valid: True if detection and handedness resolution succeeded.
        status: Status code string ('OK', 'UNSUPPORTED_HAND_COUNT', 'AMBIGUOUS_DUPLICATE_HANDEDNESS',
                'UNCERTAIN_HANDEDNESS', 'EMPTY_DETECTION').
        is_mirrored: Whether camera mirroring correction was applied.
        relative_wrist_displacement: Optional 1D NumPy array of shape (3,) containing [dx, dy, dz]
                                     from Left wrist to Right wrist, normalized by average palm scale.
                                     None for one-hand signs or invalid detections.
        error_message: Detailed explanation if is_valid is False.
    """
    features: Optional[np.ndarray]
    hand_count: int
    resolved_handedness: Tuple[str, ...]
    is_valid: bool
    status: str
    is_mirrored: bool = False
    relative_wrist_displacement: Optional[np.ndarray] = None
    error_message: Optional[str] = None

    def get_augmented_features(self) -> Optional[np.ndarray]:
        """
        Return the augmented feature vector:
        - For two-hand signs: (129,) float32 array = [126 baseline features + 3 relative wrist displacement].
        - For one-hand signs: (63,) float32 array = [63 baseline features].
        - If invalid: returns None.
        """
        if not self.is_valid or self.features is None:
            return None
        if self.hand_count == 1 or self.relative_wrist_displacement is None:
            return self.features.copy()
        return np.concatenate([self.features, self.relative_wrist_displacement], axis=0).astype(np.float32)


# =============================================================================
# Core Single-Hand Normalization Functions (Stage 1)
# =============================================================================

def _to_numpy_landmarks(landmarks: Any) -> np.ndarray:
    """
    Validate and convert input landmarks to a NumPy array of shape (21, 3) with float32 dtype.

    Supports:
    - np.ndarray with shape (21, 3) or flat shape (63,)
    - List/tuple of 21 (x, y, z) coordinates or flat list of 63 numbers
    - List/sequence of 21 objects with .x, .y, .z attributes (MediaPipe Landmark objects)

    Raises:
        TypeError: If input is None or unsupported type.
        ValueError: If input shape is invalid, not 21 landmarks, or contains NaN/Inf.
    """
    if landmarks is None:
        raise TypeError("Input landmarks cannot be None.")

    # Handle MediaPipe Landmark objects (e.g. list of NormalizedLandmark)
    if hasattr(landmarks, "__iter__") and not isinstance(landmarks, (np.ndarray, list, tuple)):
        landmarks = list(landmarks)

    if isinstance(landmarks, (list, tuple)) and len(landmarks) == NUM_LANDMARKS:
        first_item = landmarks[0]
        if hasattr(first_item, "x") and hasattr(first_item, "y") and hasattr(first_item, "z"):
            coords = []
            for lm in landmarks:
                coords.append([lm.x, lm.y, getattr(lm, "z", 0.0)])
            arr = np.asarray(coords, dtype=np.float32)
        else:
            arr = np.asarray(landmarks, dtype=np.float32)
    else:
        try:
            arr = np.asarray(landmarks, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot convert input landmarks to float32 NumPy array: {exc}") from exc

    # Check for NaN or Inf in input
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Input landmarks contain NaN or Inf values.")

    # Validate shape
    if arr.shape == (TOTAL_FEATURES_PER_HAND,):
        arr = arr.reshape((NUM_LANDMARKS, NUM_COORDINATES))
    elif arr.shape != (NUM_LANDMARKS, NUM_COORDINATES):
        raise ValueError(
            f"Expected landmark shape ({NUM_LANDMARKS}, {NUM_COORDINATES}) or "
            f"({TOTAL_FEATURES_PER_HAND},), but got {arr.shape}."
        )

    return arr


def center_landmarks_on_wrist(landmarks: Any) -> np.ndarray:
    """
    Translate hand landmarks so the wrist (landmark 0) is at the origin (0, 0, 0).

    Formula for each landmark i in [0..20]:
        L'_i = L_i - L_wrist

    This removes the effect of hand position relative to the camera frame.

    Args:
        landmarks: 21 landmarks with (x, y, z) coordinates as array, list, or MediaPipe landmarks.

    Returns:
        np.ndarray: Translated landmarks of shape (21, 3) and dtype float32,
                    where landmark 0 is exactly (0.0, 0.0, 0.0).

    Raises:
        ValueError: If input landmarks shape is not (21, 3) or (63,), or contains NaN/Inf.
    """
    points = _to_numpy_landmarks(landmarks)

    wrist = points[WRIST_INDEX].copy()
    centered = points - wrist

    # Explicitly guarantee landmark 0 is exact zero
    centered[WRIST_INDEX] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return centered.astype(np.float32)


def scale_normalize_landmarks(landmarks: Any, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """
    Normalize hand size using the Euclidean distance between the wrist (landmark 0)
    and the middle finger MCP joint (landmark 9) as the palm-size scale factor.

    Formula:
        Scale S = || L_mcp9 - L_wrist ||_2
        L''_i   = L_i / S

    If the scale S is zero or near-zero (< eps), scaling is skipped safely to prevent
    division by zero, returning the input coordinates without producing NaN or Inf.

    Args:
        landmarks: 21 landmarks with (x, y, z) coordinates (typically wrist-centered).
        eps: Small positive threshold below which scale is treated as degenerate.

    Returns:
        np.ndarray: Scale-normalized landmarks of shape (21, 3) and dtype float32.

    Raises:
        ValueError: If input landmarks shape is not (21, 3) or (63,), or contains NaN/Inf.
    """
    points = _to_numpy_landmarks(landmarks)

    wrist = points[WRIST_INDEX]
    middle_mcp = points[MIDDLE_FINGER_MCP_INDEX]

    # Euclidean distance between wrist (0) and middle finger MCP (9)
    scale = float(np.linalg.norm(middle_mcp - wrist))

    # Safe handling of zero or near-zero scale (degenerate or invalid hand)
    if scale < eps or np.isnan(scale) or np.isinf(scale):
        return points.copy().astype(np.float32)

    scaled = points / scale
    return scaled.astype(np.float32)


def normalize_hand_landmarks(landmarks: Any, flatten: bool = False, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """
    Fully normalize hand landmarks by combining wrist-centering and palm-scale normalization.

    Workflow:
    1. Validate input landmarks (shape (21, 3) or (63,)).
    2. Subtract wrist coordinate (landmark 0) to achieve translation invariance.
    3. Divide by the wrist-to-middle-MCP distance to achieve scale invariance.
    4. Return (21, 3) array or optionally flattened 63-element vector.

    Args:
        landmarks: 21 landmarks with (x, y, z) coordinates as array, list, or MediaPipe landmarks.
        flatten: If True, returns a 1D vector of 63 float32 features.
                 If False (default), returns a 2D array of shape (21, 3).
        eps: Small threshold to handle degenerate hand scales safely.

    Returns:
        np.ndarray: Normalized landmarks with shape (21, 3) if flatten=False,
                    or shape (63,) if flatten=True. Dtype is float32.

    Raises:
        ValueError: If input landmarks cannot be processed or contain NaN/Inf.
    """
    centered = center_landmarks_on_wrist(landmarks)
    normalized = scale_normalize_landmarks(centered, eps=eps)

    if flatten:
        return normalized.reshape((TOTAL_FEATURES_PER_HAND,)).astype(np.float32)

    return normalized.astype(np.float32)


# =============================================================================
# Stage 2 & 3: Handedness Resolution, Relative Wrist Geometry & Formatting
# =============================================================================

def resolve_handedness_label(
    handedness_entry: Any,
    is_mirrored: bool = False,
    min_confidence: float = 0.0,
    swap_unmirrored: bool = False,
) -> Tuple[str, float]:
    """
    Extract, validate, and resolve the handedness label ('Left' or 'Right')
    and its confidence score from a MediaPipe handedness entry.

    MediaPipe HandLandmarker Mirroring Convention:
    MediaPipe's internal hand detection model was trained under the assumption that
    the input image is a horizontally mirrored view (like a front-facing selfie camera).
    - In mirrored views (cv2.flip(frame, 1)): MediaPipe's label matches the user's
      anatomical hands directly (Left is Left, Right is Right).
    - In un-mirrored views (standard third-person photos): MediaPipe's label corresponds
      to the screen perspective (the hand on the image-left is labeled 'Left').
      According to official MediaPipe documentation: "If that is not the case [un-mirrored],
      please swap the handedness output in the application" to recover true anatomical handedness.

    Parameters:
        handedness_entry: A string, Category object, dictionary, or sequence containing
                          MediaPipe handedness output.
        is_mirrored: Whether the input image was horizontally flipped.
        min_confidence: Minimum score threshold (0.0 to 1.0) required to accept the label.
        swap_unmirrored: If True, swaps Left <-> Right when is_mirrored is False to map
                         screen-perspective labels to anatomical labels.

    Returns:
        Tuple[str, float]: (resolved_label, confidence_score).
                           If uncertain or invalid, label will be "Unknown".
    """
    if handedness_entry is None:
        return ("Unknown", 0.0)

    # Unwrap sequence if MediaPipe returned a list of Category objects for one hand
    if isinstance(handedness_entry, (list, tuple)):
        if len(handedness_entry) == 0:
            return ("Unknown", 0.0)
        handedness_entry = handedness_entry[0]

    raw_label = ""
    score = 1.0

    if isinstance(handedness_entry, str):
        raw_label = handedness_entry.strip().capitalize()
    elif isinstance(handedness_entry, dict):
        raw_label = str(
            handedness_entry.get("category_name")
            or handedness_entry.get("display_name")
            or handedness_entry.get("label", "")
        ).strip().capitalize()
        score = float(handedness_entry.get("score", 1.0))
    else:
        # MediaPipe Category or custom object
        raw_label = str(
            getattr(handedness_entry, "category_name", None)
            or getattr(handedness_entry, "display_name", None)
            or getattr(handedness_entry, "label", "")
        ).strip().capitalize()
        score = float(getattr(handedness_entry, "score", 1.0))

    if raw_label not in VALID_HANDEDNESS_LABELS or score < min_confidence:
        return ("Unknown", score)

    # Resolve label based on mirroring policy
    resolved = raw_label
    if swap_unmirrored and not is_mirrored:
        resolved = "Right" if raw_label == "Left" else "Left"
    elif is_mirrored and not swap_unmirrored:
        # When comparing mirrored webcam against un-mirrored dataset with screen-space matching:
        resolved = "Right" if raw_label == "Left" else "Left"

    return (resolved, score)


def compute_relative_wrist_displacement(
    left_hand: Any,
    right_hand: Any,
    eps: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Compute the relative 3D displacement vector from the Left wrist to the Right wrist,
    normalized by the average palm scale of both hands.

    Formula:
        Delta_wrist = Wrist_right - Wrist_left
        Scale_left  = || MiddleMCP_left - Wrist_left ||_2
        Scale_right = || MiddleMCP_right - Wrist_right ||_2
        Scale_avg   = (Scale_left + Scale_right) / 2.0
        Normalized  = Delta_wrist / Scale_avg

    Crucial Rule:
    This MUST be calculated using raw, un-centered hand coordinates before individual
    wrist-centering sets both wrists to (0, 0, 0).

    Args:
        left_hand: 21 landmarks of the left hand in raw image coordinates.
        right_hand: 21 landmarks of the right hand in raw image coordinates.
        eps: Small epsilon to avoid division by zero.

    Returns:
        np.ndarray: 1D array of shape (3,) with [dx, dy, dz] in float32.
    """
    raw_left = _to_numpy_landmarks(left_hand)
    raw_right = _to_numpy_landmarks(right_hand)

    wrist_left = raw_left[WRIST_INDEX]
    wrist_right = raw_right[WRIST_INDEX]

    delta_wrist = wrist_right - wrist_left

    # Compute characteristic palm scales
    scale_left = float(np.linalg.norm(raw_left[MIDDLE_FINGER_MCP_INDEX] - wrist_left))
    scale_right = float(np.linalg.norm(raw_right[MIDDLE_FINGER_MCP_INDEX] - wrist_right))
    avg_scale = (scale_left + scale_right) / 2.0

    if avg_scale < eps or np.isnan(avg_scale) or np.isinf(avg_scale):
        return delta_wrist.astype(np.float32)

    norm_disp = delta_wrist / avg_scale
    return norm_disp.astype(np.float32)


def format_two_hand_features(left_hand: Any, right_hand: Any, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """
    Format two known hands (Left and Right) into a strictly ordered 126-feature float32 vector.

    Indices:
        0 .. 62   : Left hand normalized features (21 landmarks x 3)
        63 .. 125 : Right hand normalized features (21 landmarks x 3)

    Args:
        left_hand: Landmarks of the left hand.
        right_hand: Landmarks of the right hand.
        eps: Small epsilon for scale normalization safety.

    Returns:
        np.ndarray: Exactly 126 float32 features.
    """
    left_norm = normalize_hand_landmarks(left_hand, flatten=True, eps=eps)
    right_norm = normalize_hand_landmarks(right_hand, flatten=True, eps=eps)
    return np.concatenate([left_norm, right_norm], axis=0).astype(np.float32)


def format_hand_features(
    detected_hands: Any,
    handedness: Optional[Any] = None,
    is_mirrored: bool = False,
    min_confidence: float = 0.0,
    strict: bool = True,
    swap_unmirrored: bool = False,
    eps: float = DEFAULT_EPSILON,
) -> HandFeatureResult:
    """
    Extract, normalize, and format hand landmarks into consistent one-hand or two-hand
    representations for Sri Lankan Sign Language recognition.

    Rules & Guarantees:
    - One-hand signs: Output has shape (63,) float32. No zero-padding.
    - Two-hand signs: Output has shape (126,) float32, guaranteed ordered as Left then Right.
    - Relative wrist displacement: Computed before wrist-centering and stored separately (shape (3,)).
    - Invariance to detection order: [Left, Right] and [Right, Left] yield identical feature vectors.
    - Ambiguity rejection: Rejects hand counts != {1, 2}, duplicate labels (e.g. Left + Left),
      or low-confidence handedness without guessing.

    Args:
        detected_hands: List of landmark sets from MediaPipe, or a MediaPipe result object.
        handedness: Corresponding handedness list from MediaPipe (or None if embedded in detected_hands).
        is_mirrored: Set True if input frame was horizontally flipped (e.g. webcam).
        min_confidence: Minimum handedness confidence score to accept (default: 0.0).
        strict: If True, raises HandednessError on invalid input.
                If False, returns HandFeatureResult with is_valid=False and error details.
        swap_unmirrored: Set True if unmirrored images should have handedness inverted to match anatomical hands.
        eps: Safe scale epsilon.

    Returns:
        HandFeatureResult: Structured result containing features, hand_count, resolved_handedness,
                           relative_wrist_displacement, validity status, and mirroring metadata.

    Raises:
        HandednessError: If strict=True and detection cannot be validly resolved.
    """
    # Convenience: support passing MediaPipe detection result object directly
    if handedness is None and hasattr(detected_hands, "hand_landmarks"):
        result_obj = detected_hands
        detected_hands = getattr(result_obj, "hand_landmarks", [])
        handedness = getattr(result_obj, "handedness", [])

    if detected_hands is None:
        detected_hands = []
    elif hasattr(detected_hands, "__iter__") and not isinstance(detected_hands, list):
        detected_hands = list(detected_hands)

    if handedness is None:
        handedness = []
    elif hasattr(handedness, "__iter__") and not isinstance(handedness, list):
        handedness = list(handedness)

    n_hands = len(detected_hands)

    # 1. Validate hand count
    if n_hands == 0:
        err = "No hands detected. Expected exactly 1 or 2 hands."
        if strict:
            raise HandednessError(err)
        return HandFeatureResult(
            features=None, hand_count=0, resolved_handedness=(),
            is_valid=False, status="EMPTY_DETECTION",
            is_mirrored=is_mirrored, relative_wrist_displacement=None, error_message=err,
        )

    if n_hands > 2:
        err = f"Unsupported hand count: {n_hands}. Expected exactly 1 or 2 hands."
        if strict:
            raise HandednessError(err)
        return HandFeatureResult(
            features=None, hand_count=n_hands, resolved_handedness=(),
            is_valid=False, status="UNSUPPORTED_HAND_COUNT",
            is_mirrored=is_mirrored, relative_wrist_displacement=None, error_message=err,
        )

    # Check handedness list length matches detected hands
    if len(handedness) != n_hands:
        err = (
            f"Handedness length mismatch: received {n_hands} hand landmark sets "
            f"but {len(handedness)} handedness entries."
        )
        if strict:
            raise HandednessError(err)
        return HandFeatureResult(
            features=None, hand_count=n_hands, resolved_handedness=(),
            is_valid=False, status="UNCERTAIN_HANDEDNESS",
            is_mirrored=is_mirrored, relative_wrist_displacement=None, error_message=err,
        )

    # 2. Resolve handedness for each detected hand and normalize coordinates
    resolved_labels: List[str] = []
    norm_hands: List[np.ndarray] = []

    for i in range(n_hands):
        label, score = resolve_handedness_label(
            handedness[i],
            is_mirrored=is_mirrored,
            min_confidence=min_confidence,
            swap_unmirrored=swap_unmirrored,
        )

        if label not in VALID_HANDEDNESS_LABELS:
            err = (
                f"Hand index {i} has unresolved or low-confidence handedness: "
                f"'{label}' (score={score:.2f}, min_required={min_confidence:.2f})."
            )
            if strict:
                raise HandednessError(err)
            return HandFeatureResult(
                features=None, hand_count=n_hands, resolved_handedness=(),
                is_valid=False, status="UNCERTAIN_HANDEDNESS",
                is_mirrored=is_mirrored, relative_wrist_displacement=None, error_message=err,
            )

        resolved_labels.append(label)
        norm_hand = normalize_hand_landmarks(detected_hands[i], flatten=True, eps=eps)
        norm_hands.append(norm_hand)

    # 3. Handle Single-Hand Sign (Letter C)
    if n_hands == 1:
        features = norm_hands[0]
        resolved_label = resolved_labels[0]
        return HandFeatureResult(
            features=features,
            hand_count=1,
            resolved_handedness=(resolved_label,),
            is_valid=True,
            status="OK",
            is_mirrored=is_mirrored,
            relative_wrist_displacement=None,
        )

    # 4. Handle Two-Hand Sign (All other SLS letters)
    # Check for ambiguous duplicate handedness labels (e.g. Left + Left or Right + Right)
    if resolved_labels[0] == resolved_labels[1]:
        err = (
            f"Ambiguous handedness labels: both detected hands were resolved as '{resolved_labels[0]}'. "
            "Two-hand signs require one Left hand and one Right hand."
        )
        if strict:
            raise HandednessError(err)
        return HandFeatureResult(
            features=None, hand_count=2, resolved_handedness=tuple(resolved_labels),
            is_valid=False, status="AMBIGUOUS_DUPLICATE_HANDEDNESS",
            is_mirrored=is_mirrored, relative_wrist_displacement=None, error_message=err,
        )

    # Order hands consistently: Left hand (0..62), Right hand (63..125)
    idx_left = 0 if resolved_labels[0] == "Left" else 1
    idx_right = 1 if resolved_labels[0] == "Left" else 0

    left_features = norm_hands[idx_left]
    right_features = norm_hands[idx_right]

    combined = np.concatenate([left_features, right_features], axis=0).astype(np.float32)

    # Compute relative wrist displacement using raw, un-centered coordinates
    raw_left = detected_hands[idx_left]
    raw_right = detected_hands[idx_right]
    rel_wrist_disp = compute_relative_wrist_displacement(raw_left, raw_right, eps=eps)

    return HandFeatureResult(
        features=combined,
        hand_count=2,
        resolved_handedness=("Left", "Right"),
        is_valid=True,
        status="OK",
        is_mirrored=is_mirrored,
        relative_wrist_displacement=rel_wrist_disp,
    )
