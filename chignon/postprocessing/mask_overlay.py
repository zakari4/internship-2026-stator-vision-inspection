"""
Mask overlay rendering for chignon detections.

Blends a coloured segmentation mask onto a video frame.
"""

import cv2
import numpy as np


def apply_mask_overlay(
    frame: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Blend a single-colour mask onto *frame* and return the result.

    Parameters
    ----------
    frame       : BGR image to annotate.
    binary_mask : uint8 mask where values > 127 are foreground.
    color       : BGR colour tuple for the mask fill.
    alpha       : Opacity of the colour overlay (0 = invisible, 1 = opaque).
    """
    color_layer = np.zeros_like(frame)
    color_layer[binary_mask > 127] = color
    return cv2.addWeighted(frame, 1.0, color_layer, alpha, 0)
