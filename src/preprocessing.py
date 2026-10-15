"""
SignVisionAI - Image Preprocessing Module
Part 2: Landmark Extraction + Normalization

This module provides reusable, non-destructive image preprocessing functions to improve
MediaPipe HandLandmarker detection performance on Sri Lankan Sign Language (SLS) images.

Key Insight & Mathematical Principles:
1. Aspect-Ratio-Preserving Square Padding (Letterbox):
   - MediaPipe HandLandmarker internally operates on a square region of interest (ROI).
     When given non-square images (e.g. 3:4 portrait or 16:9 landscape), MediaPipe's
     internal landmark projection calculator issues warnings and applies anisotropic
     normalization (where 1 normalized unit in X does not equal 1 normalized unit in Y).
   - By symmetrically padding the shorter dimension with black borders (0, 0, 0) up to
     max(H, W), the original content's aspect ratio is 100% preserved without any stretching.
   - Resizing the resulting square to a target size (e.g., 1024x1024) scales both axes by
     the identical scalar factor (target_size / max_side), preserving all geometric angles.

2. Landmark Invariance to Padding:
   - For any landmark i:
     x_padded = (x_orig * W + left_pad) / S
     y_padded = (y_orig * H + top_pad) / S
     where S = max(H, W).
   - In our normalization pipeline (src/normalization.py):
     - Wrist centering subtracts wrist landmark 0:
       (x_i - x_0)_padded = (x_i - x_0)_orig * (W / S)
       Because W is constant across all landmarks of the hand, the padding offset
       (left_pad / S, top_pad / S) cancels out completely!
     - Palm scale normalization divides by Euclidean distance ||L_9 - L_0||:
       Both numerator and denominator scale by 1 / S, which cancels out completely!
   - Therefore, wrist-centered, palm-scale-normalized landmarks are invariant to both the
     padding margin and the canvas dimension.

3. Webcam Integration Compatibility:
   - During live webcam capture (which typically produces 640x480 or 1280x720 frames):
     1. Flip horizontally if in mirrored selfie mode: cv2.flip(frame, 1).
     2. Call `pad_to_square(frame, target_size=1024)`.
     3. Convert BGR to RGB: cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).
     4. Pass to MediaPipe detector: mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb).
     5. Pass detected landmarks to `format_hand_features()`.
   - This ensures live webcam input shares the exact same isotropic coordinate geometry
     as the dataset reference features.
"""

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


DEFAULT_TARGET_SIZE: int = 1024
DEFAULT_PAD_COLOR: Tuple[int, int, int] = (0, 0, 0)


def compute_square_padding(
    height: int,
    width: int,
) -> Tuple[int, int, int, int, int]:
    """
    Compute symmetric padding borders to make an image square (max_side x max_side).

    Args:
        height: Height of the original image in pixels.
        width: Width of the original image in pixels.

    Returns:
        Tuple[top, bottom, left, right, max_side]:
        - top: Top padding in pixels.
        - bottom: Bottom padding in pixels.
        - left: Left padding in pixels.
        - right: Right padding in pixels.
        - max_side: Resulting square dimension = max(height, width).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive integers, got height={height}, width={width}")

    max_side = max(height, width)
    top = (max_side - height) // 2
    bottom = max_side - height - top
    left = (max_side - width) // 2
    right = max_side - width - left

    return top, bottom, left, right, max_side


def pad_to_square(
    image: np.ndarray,
    target_size: Optional[int] = DEFAULT_TARGET_SIZE,
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
    upscale: bool = False,
    interpolation_down: int = cv2.INTER_AREA,
    interpolation_up: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Pad an image symmetrically to a square canvas and optionally resize to target_size.

    Guarantees:
    - Never modifies the input image in place (pure, non-destructive function).
    - Preserves the exact aspect ratio of the visual content without any stretching.
    - Preserves color channels (works for 3-channel BGR/RGB and single-channel grayscale).

    Args:
        image: Input image as a NumPy array of shape (H, W, C) or (H, W).
        target_size: Target square dimension (e.g. 1024). If None, keeps the square
                     at max(H, W) without resizing.
        pad_color: Border fill color (default: black (0, 0, 0)).
        upscale: If True, upscales images whose max_side is smaller than target_size.
                 If False (default), only downscales images larger than target_size,
                 preserving native resolution for small images (e.g. 224x224).
        interpolation_down: OpenCV interpolation method for downsampling (default: INTER_AREA).
        interpolation_up: OpenCV interpolation method for upsampling (default: INTER_LINEAR).

    Returns:
        np.ndarray: Padded (and optionally resized) square image with identical dtype.

    Raises:
        TypeError: If image is not a NumPy array.
        ValueError: If image is empty or has invalid dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be a numpy.ndarray, got {type(image).__name__}")

    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D non-empty image array, got shape {image.shape}")

    h, w = image.shape[:2]
    top, bottom, left, right, max_side = compute_square_padding(h, w)

    # Apply symmetric border padding
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        padded = image.copy()
    else:
        padded = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )

    # Optional resizing to target_size
    if target_size is not None and target_size > 0:
        if max_side > target_size:
            padded = cv2.resize(
                padded,
                (target_size, target_size),
                interpolation=interpolation_down,
            )
        elif upscale and max_side < target_size:
            padded = cv2.resize(
                padded,
                (target_size, target_size),
                interpolation=interpolation_up,
            )

    return padded


def preprocess_image(
    image: np.ndarray,
    method: str = "none",
    target_size: int = DEFAULT_TARGET_SIZE,
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
    upscale: bool = False,
) -> np.ndarray:
    """
    Apply a configurable preprocessing method to an image.

    Supported methods:
    - 'none' (or 'original'): Returns a copy of the original image unmodified.
    - 'square_pad': Applies aspect-ratio-preserving square padding + resizing.

    Args:
        image: Input image array (BGR or RGB).
        method: Preprocessing strategy name.
        target_size: Target dimension for square padding (default: 1024).
        pad_color: Border padding color (default: (0, 0, 0)).
        upscale: Whether to upscale small images.

    Returns:
        np.ndarray: Preprocessed image.
    """
    norm_method = (method or "none").strip().lower()

    if norm_method in ("none", "original", "baseline"):
        return image.copy()
    elif norm_method in ("square_pad", "square_padding", "square"):
        return pad_to_square(
            image=image,
            target_size=target_size,
            pad_color=pad_color,
            upscale=upscale,
        )
    else:
        raise ValueError(
            f"Unsupported preprocessing method: '{method}'. "
            f"Supported methods are: 'none', 'square_pad'."
        )
