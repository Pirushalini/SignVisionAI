"""
Tests for SignVisionAI Landmark Normalization Module
Part 2: Landmark Extraction + Normalization

Test Suite Coverage:
- Stage 1: Wrist centering, palm-size scale normalization, translation/scale invariance,
           error handling, degenerate scale safety, landmark order preservation.
- Stage 2: Handedness resolution, camera mirroring compensation, two-hand Left/Right ordering,
           detection-order invariance, single-hand vs two-hand representation separation,
           duplicate/invalid handedness rejection.
- Stage 3: Relative wrist displacement geometry, scale/translation invariance of inter-hand vector,
           augmented 129-feature representation, Member ID inference, and extraction output structure.
"""

import json
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import (
    center_landmarks_on_wrist,
    scale_normalize_landmarks,
    normalize_hand_landmarks,
    resolve_handedness_label,
    compute_relative_wrist_displacement,
    format_hand_features,
    format_two_hand_features,
    HandFeatureResult,
    HandednessError,
    NUM_LANDMARKS,
    NUM_COORDINATES,
    TOTAL_FEATURES_PER_HAND,
    ONE_HAND_FEATURE_COUNT,
    TWO_HAND_FEATURE_COUNT,
    RELATIVE_WRIST_FEATURE_COUNT,
    AUGMENTED_TWO_HAND_FEATURE_COUNT,
)

from src.extract_landmarks import get_member_id


class MockMediaPipeLandmark:
    """Mock object simulating a MediaPipe landmark with .x, .y, .z attributes."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class MockCategory:
    """Mock object simulating a MediaPipe Category object with category_name and score."""
    def __init__(self, category_name: str, score: float = 0.95):
        self.category_name = category_name
        self.score = score


def generate_sample_hand(wrist=(0.5, 0.5, 0.0), scale=0.15, thumb_direction=-1.0) -> np.ndarray:
    """
    Generate synthetic 21 hand landmarks with realistic relative positions.
    Landmark 0 is wrist, Landmark 9 is middle finger MCP joint.
    thumb_direction: -1.0 for left-pointing thumb, +1.0 for right-pointing thumb.
    """
    hand = np.zeros((NUM_LANDMARKS, NUM_COORDINATES), dtype=np.float32)
    wx, wy, wz = wrist

    hand[0] = [wx, wy, wz]  # Wrist
    hand[1] = [wx + 0.2 * scale * thumb_direction, wy - 0.2 * scale, wz]  # Thumb CMC
    hand[2] = [wx + 0.3 * scale * thumb_direction, wy - 0.4 * scale, wz]  # Thumb MCP
    hand[3] = [wx + 0.4 * scale * thumb_direction, wy - 0.6 * scale, wz]  # Thumb IP
    hand[4] = [wx + 0.5 * scale * thumb_direction, wy - 0.7 * scale, wz]  # Thumb tip

    hand[5] = [wx - 0.15 * scale, wy - 0.7 * scale, wz]  # Index MCP
    hand[6] = [wx - 0.15 * scale, wy - 0.9 * scale, wz]  # Index PIP
    hand[7] = [wx - 0.15 * scale, wy - 1.1 * scale, wz]  # Index DIP
    hand[8] = [wx - 0.15 * scale, wy - 1.3 * scale, wz]  # Index tip

    hand[9] = [wx, wy - 0.75 * scale, wz]  # Middle MCP (Key Palm Scale Point)
    hand[10] = [wx, wy - 1.0 * scale, wz]  # Middle PIP
    hand[11] = [wx, wy - 1.25 * scale, wz]  # Middle DIP
    hand[12] = [wx, wy - 1.45 * scale, wz]  # Middle tip

    hand[13] = [wx + 0.15 * scale, wy - 0.7 * scale, wz]  # Ring MCP
    hand[14] = [wx + 0.15 * scale, wy - 0.9 * scale, wz]  # Ring PIP
    hand[15] = [wx + 0.15 * scale, wy - 1.1 * scale, wz]  # Ring DIP
    hand[16] = [wx + 0.15 * scale, wy - 1.3 * scale, wz]  # Ring tip

    hand[17] = [wx + 0.3 * scale, wy - 0.6 * scale, wz]  # Pinky MCP
    hand[18] = [wx + 0.3 * scale, wy - 0.8 * scale, wz]  # Pinky PIP
    hand[19] = [wx + 0.3 * scale, wy - 1.0 * scale, wz]  # Pinky DIP
    hand[20] = [wx + 0.3 * scale, wy - 1.15 * scale, wz]  # Pinky tip

    return hand


# =============================================================================
# Stage 1 Tests: Single-Hand Normalization
# =============================================================================

class TestStage1LandmarkNormalization(unittest.TestCase):
    """Test suite covering the 7 requirements for single-hand normalization."""

    def setUp(self):
        self.sample_hand = generate_sample_hand(wrist=(0.4, 0.6, -0.05), scale=0.18)

    def test_wrist_becomes_origin(self):
        """Verify that wrist (landmark 0) is placed exactly at (0, 0, 0)."""
        centered = center_landmarks_on_wrist(self.sample_hand)
        np.testing.assert_allclose(centered[0], [0.0, 0.0, 0.0], atol=1e-7)

        normalized = normalize_hand_landmarks(self.sample_hand)
        np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-7)

    def test_translation_invariance(self):
        """Moving the hand across the screen must produce identical normalized landmarks."""
        norm_original = normalize_hand_landmarks(self.sample_hand)

        shifts = [
            np.array([0.3, 0.2, 0.1], dtype=np.float32),
            np.array([-0.35, 0.15, -0.2], dtype=np.float32),
            np.array([0.5, -0.4, 0.05], dtype=np.float32),
        ]

        for shift in shifts:
            shifted_hand = self.sample_hand + shift
            norm_shifted = normalize_hand_landmarks(shifted_hand)
            np.testing.assert_allclose(
                norm_original,
                norm_shifted,
                atol=1e-5,
                err_msg=f"Translation invariance failed for shift {shift}",
            )

    def test_scale_invariance(self):
        """Scaling hand coordinates (closer/further from camera) must yield equivalent normalized output."""
        wrist = self.sample_hand[0]
        centered_hand = self.sample_hand - wrist

        norm_base = normalize_hand_landmarks(self.sample_hand)

        scale_factors = [0.5, 0.75, 1.5, 2.0, 3.5]
        for factor in scale_factors:
            scaled_hand = (centered_hand * factor) + wrist
            norm_scaled = normalize_hand_landmarks(scaled_hand)
            np.testing.assert_allclose(
                norm_base,
                norm_scaled,
                atol=1e-5,
                err_msg=f"Scale invariance failed for factor {factor}",
            )

    def test_no_nan_or_infinity(self):
        """Ensure all returned coordinates are finite."""
        normalized = normalize_hand_landmarks(self.sample_hand)
        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertFalse(np.any(np.isnan(normalized)))

        flattened = normalize_hand_landmarks(self.sample_hand, flatten=True)
        self.assertTrue(np.all(np.isfinite(flattened)))
        self.assertEqual(flattened.shape, (TOTAL_FEATURES_PER_HAND,))

    def test_invalid_input_shapes_and_values(self):
        """Invalid shapes, dimensions, types, or NaN values must raise clear errors."""
        with self.assertRaises(ValueError):
            center_landmarks_on_wrist(np.zeros((20, 3)))

        with self.assertRaises(ValueError):
            normalize_hand_landmarks(np.zeros((21, 2)))

        with self.assertRaises(ValueError):
            normalize_hand_landmarks(np.zeros(62))

        with self.assertRaises(TypeError):
            normalize_hand_landmarks(None)

        nan_hand = self.sample_hand.copy()
        nan_hand[5, 1] = np.nan
        with self.assertRaises(ValueError):
            normalize_hand_landmarks(nan_hand)

        inf_hand = self.sample_hand.copy()
        inf_hand[10, 0] = np.inf
        with self.assertRaises(ValueError):
            normalize_hand_landmarks(inf_hand)

    def test_degenerate_scale_handling(self):
        """Zero scale (e.g. wrist and MCP at identical coordinates) must not raise or produce NaN/Inf."""
        zero_hand = np.zeros((NUM_LANDMARKS, NUM_COORDINATES), dtype=np.float32)
        result_zero = normalize_hand_landmarks(zero_hand)
        self.assertTrue(np.all(np.isfinite(result_zero)))
        np.testing.assert_allclose(result_zero, 0.0)

        coincident_mcp = self.sample_hand.copy()
        coincident_mcp[9] = coincident_mcp[0]
        result_coincident = normalize_hand_landmarks(coincident_mcp)
        self.assertTrue(np.all(np.isfinite(result_coincident)))
        self.assertFalse(np.any(np.isnan(result_coincident)))

    def test_landmark_order_preserved(self):
        """Landmarks must remain in their original 0..20 anatomical order."""
        centered = center_landmarks_on_wrist(self.sample_hand)
        normalized = normalize_hand_landmarks(self.sample_hand)

        vec_orig = self.sample_hand[5] - self.sample_hand[0]
        vec_norm = normalized[5] - normalized[0]

        unit_orig = vec_orig / np.linalg.norm(vec_orig)
        unit_norm = vec_norm / np.linalg.norm(vec_norm)
        np.testing.assert_allclose(unit_orig, unit_norm, atol=1e-6)

        mcp9_dist = np.linalg.norm(normalized[9] - normalized[0])
        np.testing.assert_allclose(mcp9_dist, 1.0, atol=1e-6)

    def test_mediapipe_object_compatibility(self):
        """Ensure list of objects with .x, .y, .z attributes can be passed directly."""
        mp_objects = [
            MockMediaPipeLandmark(float(pt[0]), float(pt[1]), float(pt[2]))
            for pt in self.sample_hand
        ]
        normalized = normalize_hand_landmarks(mp_objects)
        self.assertEqual(normalized.shape, (NUM_LANDMARKS, NUM_COORDINATES))
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-7)


# =============================================================================
# Stage 2 Tests: Handedness, Ordering & One-Hand/Two-Hand Representation
# =============================================================================

class TestStage2HandRepresentation(unittest.TestCase):
    """Test suite covering Stage 2 requirements for hand representation."""

    def setUp(self):
        self.left_hand = generate_sample_hand(
            wrist=(0.3, 0.5, 0.0), scale=0.15, thumb_direction=-1.0
        )
        self.right_hand = generate_sample_hand(
            wrist=(0.7, 0.5, 0.0), scale=0.20, thumb_direction=1.0
        )

        self.left_norm = normalize_hand_landmarks(self.left_hand, flatten=True)
        self.right_norm = normalize_hand_landmarks(self.right_hand, flatten=True)

    def test_left_right_ordering(self):
        """Two-hand signs must always place Left hand features in 0..62 and Right hand in 63..125."""
        res = format_hand_features(
            detected_hands=[self.left_hand, self.right_hand],
            handedness=["Left", "Right"],
            is_mirrored=False,
        )

        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, "OK")
        self.assertEqual(res.hand_count, 2)
        self.assertEqual(res.resolved_handedness, ("Left", "Right"))

        np.testing.assert_allclose(res.features[:63], self.left_norm, atol=1e-6)
        np.testing.assert_allclose(res.features[63:], self.right_norm, atol=1e-6)

    def test_detection_order_invariance(self):
        """Whether MediaPipe detects [Left, Right] or [Right, Left], the feature vector must be identical."""
        res_a = format_hand_features(
            detected_hands=[self.left_hand, self.right_hand],
            handedness=["Left", "Right"],
            is_mirrored=False,
        )

        res_b = format_hand_features(
            detected_hands=[self.right_hand, self.left_hand],
            handedness=["Right", "Left"],
            is_mirrored=False,
        )

        self.assertTrue(res_a.is_valid)
        self.assertTrue(res_b.is_valid)
        self.assertEqual(res_a.features.shape, (TWO_HAND_FEATURE_COUNT,))
        self.assertEqual(res_b.features.shape, (TWO_HAND_FEATURE_COUNT,))

        np.testing.assert_allclose(
            res_a.features,
            res_b.features,
            atol=1e-6,
            err_msg="Detection order reversed did not produce identical feature vectors!",
        )

    def test_two_hand_output_shape(self):
        """Two-hand signs must return exactly (126,) float32 features."""
        res = format_hand_features(
            detected_hands=[self.left_hand, self.right_hand],
            handedness=[MockCategory("Left"), MockCategory("Right")],
            is_mirrored=False,
        )

        self.assertEqual(res.features.shape, (126,))
        self.assertEqual(res.features.dtype, np.float32)
        self.assertEqual(res.hand_count, 2)
        self.assertEqual(len(res.features), TWO_HAND_FEATURE_COUNT)

    def test_single_hand_output_shape(self):
        """Single-hand signs (e.g. Letter C) must return exactly (63,) float32 features."""
        res_r = format_hand_features(
            detected_hands=[self.right_hand],
            handedness=["Right"],
            is_mirrored=False,
        )
        self.assertEqual(res_r.features.shape, (63,))
        self.assertEqual(res_r.features.dtype, np.float32)
        self.assertEqual(res_r.hand_count, 1)
        self.assertEqual(res_r.resolved_handedness, ("Right",))
        np.testing.assert_allclose(res_r.features, self.right_norm, atol=1e-6)

        res_l = format_hand_features(
            detected_hands=[self.left_hand],
            handedness=[MockCategory("Left")],
            is_mirrored=False,
        )
        self.assertEqual(res_l.features.shape, (63,))
        self.assertEqual(res_l.features.dtype, np.float32)
        self.assertEqual(res_l.hand_count, 1)
        self.assertEqual(res_l.resolved_handedness, ("Left",))
        np.testing.assert_allclose(res_l.features, self.left_norm, atol=1e-6)

    def test_separate_representations_no_zero_padding(self):
        """One-hand signs must NEVER be zero-padded to 126 features."""
        res_1h = format_hand_features([self.right_hand], ["Right"])
        res_2h = format_hand_features([self.left_hand, self.right_hand], ["Left", "Right"])

        self.assertEqual(len(res_1h.features), ONE_HAND_FEATURE_COUNT)
        self.assertEqual(len(res_2h.features), TWO_HAND_FEATURE_COUNT)
        self.assertNotEqual(res_1h.features.shape, res_2h.features.shape)
        self.assertEqual(res_1h.features.size, 63)

    def test_duplicate_and_invalid_handedness_rejection(self):
        """Reject ambiguous duplicate labels or invalid hand counts rather than guessing."""
        with self.assertRaises(HandednessError):
            format_hand_features(
                [self.left_hand, self.right_hand],
                ["Right", "Right"],
                strict=True,
            )

        res_dup = format_hand_features(
            [self.left_hand, self.right_hand],
            ["Left", "Left"],
            strict=False,
        )
        self.assertFalse(res_dup.is_valid)
        self.assertIsNone(res_dup.features)
        self.assertEqual(res_dup.status, "AMBIGUOUS_DUPLICATE_HANDEDNESS")

        with self.assertRaises(HandednessError):
            format_hand_features(
                [self.left_hand, self.right_hand],
                ["Unknown", "Right"],
                strict=True,
            )

        with self.assertRaises(HandednessError):
            format_hand_features([], [], strict=True)

        res_empty = format_hand_features([], [], strict=False)
        self.assertFalse(res_empty.is_valid)
        self.assertEqual(res_empty.status, "EMPTY_DETECTION")

        with self.assertRaises(HandednessError):
            format_hand_features(
                [self.left_hand, self.right_hand, self.left_hand],
                ["Left", "Right", "Left"],
                strict=True,
            )

        low_conf = MockCategory("Left", score=0.3)
        with self.assertRaises(HandednessError):
            format_hand_features(
                [self.left_hand],
                [low_conf],
                min_confidence=0.6,
                strict=True,
            )

    def test_camera_mirroring_consistency(self):
        """
        Verify mirroring compensation: mirrored frames with is_mirrored=True
        produce consistent feature vectors with un-mirrored references.
        """
        res_dataset = format_hand_features(
            detected_hands=[self.left_hand, self.right_hand],
            handedness=["Left", "Right"],
            is_mirrored=False,
        )

        res_webcam_mirrored = format_hand_features(
            detected_hands=[self.left_hand, self.right_hand],
            handedness=["Right", "Left"],
            is_mirrored=True,
        )

        self.assertTrue(res_dataset.is_valid)
        self.assertTrue(res_webcam_mirrored.is_valid)

        np.testing.assert_allclose(
            res_dataset.features,
            res_webcam_mirrored.features,
            atol=1e-6,
            err_msg="Mirroring compensation did not produce identical feature vector to un-mirrored reference!",
        )

    def test_format_two_hand_features_helper(self):
        """format_two_hand_features should produce (126,) float32 vector matching manual concatenation."""
        combined = format_two_hand_features(self.left_hand, self.right_hand)
        self.assertEqual(combined.shape, (126,))
        self.assertEqual(combined.dtype, np.float32)
        np.testing.assert_allclose(combined[:63], self.left_norm, atol=1e-6)
        np.testing.assert_allclose(combined[63:], self.right_norm, atol=1e-6)


# =============================================================================
# Stage 3 Tests: Relative Wrist Geometry & Pipeline Structures
# =============================================================================

class TestStage3ExtractionAndGeometry(unittest.TestCase):
    """Test suite covering Stage 3 relative wrist geometry and pipeline features."""

    def setUp(self):
        self.left_hand = generate_sample_hand(
            wrist=(0.25, 0.60, 0.0), scale=0.15, thumb_direction=-1.0
        )
        self.right_hand = generate_sample_hand(
            wrist=(0.75, 0.40, 0.05), scale=0.15, thumb_direction=1.0
        )

    def test_relative_wrist_displacement_vector(self):
        """Relative displacement must point from Left wrist to Right wrist, normalized by palm scale."""
        disp = compute_relative_wrist_displacement(self.left_hand, self.right_hand)
        self.assertEqual(disp.shape, (RELATIVE_WRIST_FEATURE_COUNT,))
        self.assertEqual(disp.dtype, np.float32)

        # Right wrist is at x=0.75, Left wrist is at x=0.25 -> dx must be strictly positive
        self.assertGreater(disp[0], 0.0)

        # Right wrist is at y=0.40, Left wrist is at y=0.60 -> dy must be strictly negative
        self.assertLess(disp[1], 0.0)

    def test_relative_wrist_scale_invariance(self):
        """Scaling both hands (e.g. moving closer/further from camera) must preserve normalized wrist displacement."""
        disp_base = compute_relative_wrist_displacement(self.left_hand, self.right_hand)

        # Scale both hands (including their wrist positions) by 2.0x relative to screen origin
        scaled_left = self.left_hand * 2.0
        scaled_right = self.right_hand * 2.0

        disp_scaled = compute_relative_wrist_displacement(scaled_left, scaled_right)
        np.testing.assert_allclose(
            disp_base,
            disp_scaled,
            atol=1e-5,
            err_msg="Normalized relative wrist displacement is not invariant to camera distance scaling!",
        )

    def test_relative_wrist_translation_invariance(self):
        """Translating both hands together across the screen must preserve relative wrist displacement."""
        disp_base = compute_relative_wrist_displacement(self.left_hand, self.right_hand)

        shift = np.array([0.15, -0.20, 0.05], dtype=np.float32)
        shifted_left = self.left_hand + shift
        shifted_right = self.right_hand + shift

        disp_shifted = compute_relative_wrist_displacement(shifted_left, shifted_right)
        np.testing.assert_allclose(
            disp_base,
            disp_shifted,
            atol=1e-5,
            err_msg="Translating both hands changed the relative wrist displacement!",
        )

    def test_augmented_features_generation(self):
        """HandFeatureResult must produce 129 features for two hands without altering the 126 baseline."""
        res_2h = format_hand_features(
            detected_hands=[self.left_hand, self.right_hand],
            handedness=["Left", "Right"],
        )

        self.assertTrue(res_2h.is_valid)
        self.assertEqual(res_2h.features.shape, (TWO_HAND_FEATURE_COUNT,))  # Baseline remains 126!
        self.assertIsNotNone(res_2h.relative_wrist_displacement)
        self.assertEqual(res_2h.relative_wrist_displacement.shape, (RELATIVE_WRIST_FEATURE_COUNT,))

        augmented = res_2h.get_augmented_features()
        self.assertEqual(augmented.shape, (AUGMENTED_TWO_HAND_FEATURE_COUNT,))  # 126 + 3 = 129!
        np.testing.assert_allclose(augmented[:126], res_2h.features)
        np.testing.assert_allclose(augmented[126:], res_2h.relative_wrist_displacement)

        # One-hand signs must return 63 for both baseline and augmented
        res_1h = format_hand_features(
            detected_hands=[self.right_hand],
            handedness=["Right"],
        )
        self.assertEqual(res_1h.features.shape, (ONE_HAND_FEATURE_COUNT,))
        self.assertIsNone(res_1h.relative_wrist_displacement)
        self.assertEqual(res_1h.get_augmented_features().shape, (ONE_HAND_FEATURE_COUNT,))

    def test_member_id_inference(self):
        """Verify correct contributor member mapping from filename convention."""
        self.assertEqual(get_member_id("A_01.jpg"), "Member_1")
        self.assertEqual(get_member_id("B_20.jpeg"), "Member_1")
        self.assertEqual(get_member_id("C_21.png"), "Member_2")
        self.assertEqual(get_member_id("D_40.jpg"), "Member_2")
        self.assertEqual(get_member_id("E_41.jpg"), "Member_3")
        self.assertEqual(get_member_id("F_60.jpeg"), "Member_3")
        self.assertEqual(get_member_id("G_61.png"), "Member_4")
        self.assertEqual(get_member_id("Z_80.jpg"), "Member_4")
        self.assertEqual(get_member_id("random_sample.jpg"), "Unknown")


if __name__ == "__main__":
    unittest.main()
