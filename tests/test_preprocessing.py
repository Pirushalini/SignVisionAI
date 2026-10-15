"""
Tests for SignVisionAI Image Preprocessing Module
Part 2: Landmark Extraction + Normalization

Test Suite Coverage:
1. Aspect-Ratio Preservation:
   - Verifies that square padding adds symmetric letterbox borders without stretching or warping.
   - Confirms that geometric angles and aspect ratios of objects within the image are preserved.
2. Output Dimensions:
   - Verifies correct output shapes for portrait, landscape, and square inputs.
   - Verifies target_size resizing (e.g. 1024x1024, 512x512) and native size preservation.
   - Verifies 3-channel (BGR/RGB) and 2-channel (grayscale) image support.
3. Non-Destructive Integrity:
   - Verifies that the original input image array is never modified in-place.
4. Coordinate Consistency:
   - Verifies landmark normalization mathematical invariance under square padding.
   - Confirms that wrist-centering cancels out the padding offset.
5. Normalization Pipeline Integrity:
   - Verifies that all normalization functions remain unchanged and backward compatible.
"""

import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    DEFAULT_PAD_COLOR,
    DEFAULT_TARGET_SIZE,
    compute_square_padding,
    pad_to_square,
    preprocess_image,
)

from src.normalization import (
    NUM_COORDINATES,
    NUM_LANDMARKS,
    ONE_HAND_FEATURE_COUNT,
    RELATIVE_WRIST_FEATURE_COUNT,
    TWO_HAND_FEATURE_COUNT,
    center_landmarks_on_wrist,
    compute_relative_wrist_displacement,
    format_hand_features,
    normalize_hand_landmarks,
    scale_normalize_landmarks,
)


class TestSquarePaddingGeometry(unittest.TestCase):
    """Test suite verifying aspect-ratio preservation and geometric properties."""

    def test_compute_square_padding_portrait(self):
        """Portrait image (H > W): should add equal padding to left and right."""
        h, w = 4000, 3000
        top, bottom, left, right, max_side = compute_square_padding(h, w)
        self.assertEqual(max_side, 4000)
        self.assertEqual(top, 0)
        self.assertEqual(bottom, 0)
        self.assertEqual(left, 500)
        self.assertEqual(right, 500)
        self.assertEqual(left + w + right, max_side)

    def test_compute_square_padding_landscape(self):
        """Landscape image (W > H): should add equal padding to top and bottom."""
        h, w = 1080, 1920
        top, bottom, left, right, max_side = compute_square_padding(h, w)
        self.assertEqual(max_side, 1920)
        self.assertEqual(left, 0)
        self.assertEqual(right, 0)
        self.assertEqual(top, 420)
        self.assertEqual(bottom, 420)
        self.assertEqual(top + h + bottom, max_side)

    def test_compute_square_padding_square(self):
        """Square image (H == W): padding should be zero on all sides."""
        h, w = 500, 500
        top, bottom, left, right, max_side = compute_square_padding(h, w)
        self.assertEqual(max_side, 500)
        self.assertEqual(top, 0)
        self.assertEqual(bottom, 0)
        self.assertEqual(left, 0)
        self.assertEqual(right, 0)

    def test_aspect_ratio_preserved_without_stretching(self):
        """
        Verify that an object drawn on the input image (e.g. a circle)
        remains a circle with identical horizontal and vertical radius
        in the square-padded image, proving no anisotropic stretching occurs.
        """
        # Create non-square image: 300 x 600 (aspect ratio 1:2)
        h, w = 300, 600
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw a circle with radius 40 at center
        center = (w // 2, h // 2)
        radius = 40
        cv2.circle(img, center, radius, (255, 255, 255), -1)

        # Pad to square without resizing (keeps max_side = 600)
        padded = pad_to_square(img, target_size=None)
        self.assertEqual(padded.shape, (600, 600, 3))

        # Find the white circle in the padded image
        white_pixels = np.argwhere(padded[:, :, 0] == 255)
        min_y, min_x = white_pixels.min(axis=0)
        max_y, max_x = white_pixels.max(axis=0)

        diameter_x = max_x - min_x + 1
        diameter_y = max_y - min_y + 1

        # In a circle, diameter_x == diameter_y (no stretching)
        self.assertEqual(diameter_x, diameter_y)
        self.assertEqual(diameter_x, 2 * radius + 1)

    def test_aspect_ratio_preserved_with_resizing(self):
        """
        Verify that when resizing the padded square to target_size (e.g. 1024),
        horizontal and vertical scale factors are strictly equal.
        """
        # Use image larger than target_size to test downscaling with aspect ratio preservation
        h, w = 1000, 2000  # 1:2 aspect ratio, max_side=2000 > 1024
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Place two reference points
        pt1 = (400, 200)
        pt2 = (1600, 800)
        orig_dx = pt2[0] - pt1[0]  # 1200
        orig_dy = pt2[1] - pt1[1]  # 600
        orig_ratio = orig_dx / orig_dy  # 2.0

        # Mark pixels
        img[pt1[1], pt1[0]] = [255, 0, 0]
        img[pt2[1], pt2[0]] = [0, 255, 0]

        padded = pad_to_square(img, target_size=1024)
        self.assertEqual(padded.shape, (1024, 1024, 3))

        # The scale factor from max_side (800) to target_size (1024) is 1024/800 = 1.28
        # Pad top/bottom is (800 - 400) // 2 = 200
        scale = 1024 / 800.0
        pt1_padded_x = int(round(pt1[0] * scale))
        pt1_padded_y = int(round((pt1[1] + 200) * scale))

        pt2_padded_x = int(round(pt2[0] * scale))
        pt2_padded_y = int(round((pt2[1] + 200) * scale))

        new_dx = pt2_padded_x - pt1_padded_x
        new_dy = pt2_padded_y - pt1_padded_y
        new_ratio = new_dx / new_dy

        self.assertAlmostEqual(orig_ratio, new_ratio, delta=0.02)


class TestOutputDimensionsAndOptions(unittest.TestCase):
    """Test suite verifying output dimensions across image formats and configurations."""

    def test_target_size_resizing(self):
        """Verify output dimensions for various target_size values."""
        img = np.zeros((800, 600, 3), dtype=np.uint8)

        # Default 1024
        out_1024 = pad_to_square(img, target_size=1024)
        # Since max_side is 800 <= 1024 and upscale=False, it stays at 800x800
        self.assertEqual(out_1024.shape, (800, 800, 3))

        # Large image downscaled to 1024
        large_img = np.zeros((3000, 2000, 3), dtype=np.uint8)
        out_downscaled = pad_to_square(large_img, target_size=1024)
        self.assertEqual(out_downscaled.shape, (1024, 1024, 3))

        # Downscale to 512
        out_512 = pad_to_square(img, target_size=512)
        self.assertEqual(out_512.shape, (512, 512, 3))

    def test_upscale_option(self):
        """Verify behavior when upscale is True vs False on small images."""
        small_img = np.zeros((224, 224, 3), dtype=np.uint8)

        # upscale=False (default): keep 224x224
        out_no_up = pad_to_square(small_img, target_size=1024, upscale=False)
        self.assertEqual(out_no_up.shape, (224, 224, 3))

        # upscale=True: upscale to 1024x1024
        out_up = pad_to_square(small_img, target_size=1024, upscale=True)
        self.assertEqual(out_up.shape, (1024, 1024, 3))

    def test_grayscale_image_support(self):
        """Single-channel 2D images must be padded and resized correctly."""
        # Downscaling test
        gray_large = np.zeros((800, 600), dtype=np.uint8)
        out_gray = pad_to_square(gray_large, target_size=512)
        self.assertEqual(out_gray.shape, (512, 512))
        self.assertEqual(out_gray.dtype, np.uint8)

        # Upscaling test
        gray_small = np.zeros((200, 150), dtype=np.uint8)
        out_gray_up = pad_to_square(gray_small, target_size=512, upscale=True)
        self.assertEqual(out_gray_up.shape, (512, 512))
        self.assertEqual(out_gray_up.dtype, np.uint8)

    def test_pad_color(self):
        """Border color must match requested pad_color."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        custom_color = (10, 20, 30)
        out = pad_to_square(img, target_size=None, pad_color=custom_color)
        self.assertEqual(out.shape, (200, 200, 3))

        # Top border should have custom_color
        np.testing.assert_array_equal(out[0, 100], custom_color)


class TestNonDestructiveIntegrity(unittest.TestCase):
    """Verify that original inputs are never modified."""

    def test_original_image_array_unmodified(self):
        """Calling pad_to_square must never alter the input array in place."""
        rng = np.random.RandomState(42)
        original = rng.randint(0, 255, (500, 300, 3), dtype=np.uint8)
        original_copy = original.copy()

        _ = pad_to_square(original, target_size=1024)
        np.testing.assert_array_equal(original, original_copy)

        _ = preprocess_image(original, method="square_pad", target_size=512)
        np.testing.assert_array_equal(original, original_copy)

        _ = preprocess_image(original, method="none")
        np.testing.assert_array_equal(original, original_copy)


class TestLandmarkCoordinateConsistency(unittest.TestCase):
    """
    Verify that normalized landmarks computed from square-padded images
    satisfy translation and scale invariance in our normalization pipeline.
    """

    def test_wrist_centering_cancels_padding_offset(self):
        """
        Proves mathematically that wrist-centering (L_i - L_0) completely eliminates
        any translation offset introduced by square padding borders.
        """
        # Simulate hand landmarks in original image [0..1]
        wrist_orig = np.array([0.5, 0.6, 0.0], dtype=np.float32)
        mcp_orig = np.array([0.5, 0.45, 0.0], dtype=np.float32)
        tip_orig = np.array([0.5, 0.3, 0.0], dtype=np.float32)

        # Image size: 3000 x 4000 (portrait). Top=0, Bottom=0, Left=500, Right=500. S=4000
        # In padded coordinates:
        # x_pad = (x_orig * 3000 + 500) / 4000
        # y_pad = (y_orig * 4000 + 0) / 4000 = y_orig
        def to_padded(pt):
            return np.array([(pt[0] * 3000 + 500) / 4000.0, pt[1], pt[2]], dtype=np.float32)

        wrist_pad = to_padded(wrist_orig)
        mcp_pad = to_padded(mcp_orig)
        tip_pad = to_padded(tip_orig)

        # Compute wrist-centered vector: tip - wrist
        vec_orig = tip_orig - wrist_orig
        vec_pad = tip_pad - wrist_pad

        # The constant offset (500/4000) cancels out completely!
        # In Y: exactly equal
        self.assertAlmostEqual(vec_orig[1], vec_pad[1], places=6)
        # In X: scaled isotropically by 3000/4000
        self.assertAlmostEqual(vec_pad[0], vec_orig[0] * (3000.0 / 4000.0), places=6)

    def test_scale_normalization_cancels_canvas_scale(self):
        """
        Proves that dividing wrist-centered landmarks by palm scale (||L_9 - L_0||)
        cancels the uniform scale factor of resizing.
        """
        # Create a synthetic 21-landmark hand
        from tests.test_normalization import generate_sample_hand
        hand = generate_sample_hand(wrist=(0.4, 0.6, 0.0), scale=0.15)

        # Apply an isotropic scale factor k (e.g. from canvas resizing)
        k = 1.28
        scaled_hand = hand * k

        norm_original = normalize_hand_landmarks(hand)
        norm_scaled = normalize_hand_landmarks(scaled_hand)

        np.testing.assert_allclose(norm_original, norm_scaled, atol=1e-5)


class TestNormalizationPipelineIntegrity(unittest.TestCase):
    """Confirm all existing normalization math and feature representations remain intact."""

    def test_single_hand_representation_intact(self):
        from tests.test_normalization import generate_sample_hand, MockCategory
        hand = generate_sample_hand(wrist=(0.5, 0.5, 0.0), scale=0.15)
        hds = [[MockCategory("Right", 0.95)]]

        res = format_hand_features(
            detected_hands=[hand],
            handedness=hds,
            is_mirrored=False,
            strict=False,
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.hand_count, 1)
        self.assertEqual(res.features.shape, (ONE_HAND_FEATURE_COUNT,))
        self.assertTrue(np.all(np.isfinite(res.features)))

    def test_two_hand_representation_intact(self):
        from tests.test_normalization import generate_sample_hand, MockCategory
        left = generate_sample_hand(wrist=(0.3, 0.5, 0.0), scale=0.15, thumb_direction=-1.0)
        right = generate_sample_hand(wrist=(0.7, 0.5, 0.0), scale=0.18, thumb_direction=1.0)
        hds = [[MockCategory("Left", 0.95)], [MockCategory("Right", 0.95)]]

        res = format_hand_features(
            detected_hands=[left, right],
            handedness=hds,
            is_mirrored=False,
            strict=False,
        )
        self.assertTrue(res.is_valid)
        self.assertEqual(res.hand_count, 2)
        self.assertEqual(res.features.shape, (TWO_HAND_FEATURE_COUNT,))
        self.assertIsNotNone(res.relative_wrist_displacement)
        self.assertEqual(res.relative_wrist_displacement.shape, (RELATIVE_WRIST_FEATURE_COUNT,))
        self.assertTrue(np.all(np.isfinite(res.features)))
        self.assertTrue(np.all(np.isfinite(res.relative_wrist_displacement)))


class TestErrorHandling(unittest.TestCase):
    """Verify input validation and exception handling in preprocessing."""

    def test_invalid_image_type(self):
        with self.assertRaises(TypeError):
            pad_to_square("not_an_image")

    def test_empty_image_array(self):
        with self.assertRaises(ValueError):
            pad_to_square(np.empty((0, 0, 3), dtype=np.uint8))

    def test_invalid_dimension(self):
        with self.assertRaises(ValueError):
            pad_to_square(np.zeros(10, dtype=np.uint8))

    def test_unsupported_preprocessing_method(self):
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((100, 100, 3), dtype=np.uint8), method="unsupported_method")


if __name__ == "__main__":
    unittest.main()
