"""Grid detection and perspective correction for the Sudoku Grid Extractor."""

import cv2
import numpy as np

from sudoku_grid_extractor.exceptions import GridDetectionError

# Side length of the output square image after perspective correction.
_OUTPUT_SIZE = 900


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left].

    Uses the sum and difference of (x, y) coordinates to determine position:
      - top-left has the smallest sum
      - bottom-right has the largest sum
      - top-right has the smallest difference (y - x)
      - bottom-left has the largest difference (y - x)
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def detect_grid(image: np.ndarray) -> np.ndarray:
    """Detect and extract the sudoku grid region from an image.

    Algorithm:
        1. Convert to grayscale and apply adaptive thresholding.
        2. Find contours and select the largest quadrilateral by area.
        3. Apply a perspective warp to produce a square, top-down view.

    Args:
        image: Source image as a NumPy array (BGR).

    Returns:
        A perspective-corrected square image of the grid region.

    Raises:
        GridDetectionError: If no suitable quadrilateral contour is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    # Sort contours by area (largest first) and look for a quadrilateral.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx
            break

    if quad is None:
        raise GridDetectionError(
            "No quadrilateral contour found in the image. "
            "Could not detect a sudoku grid."
        )

    # Reshape to (4, 2) and order the corners consistently.
    corners = quad.reshape(4, 2).astype(np.float32)
    ordered = _order_corners(corners)

    # Destination points for a square output image.
    dst = np.array(
        [[0, 0], [_OUTPUT_SIZE - 1, 0],
         [_OUTPUT_SIZE - 1, _OUTPUT_SIZE - 1], [0, _OUTPUT_SIZE - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (_OUTPUT_SIZE, _OUTPUT_SIZE))

    return warped
