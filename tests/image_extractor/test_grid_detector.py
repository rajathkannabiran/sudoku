"""Unit tests for the GridDetector module.

Requirements: 2.1, 2.2, 2.3
"""

import numpy as np
import cv2
import pytest

from sudoku_grid_extractor.grid_detector import detect_grid
from sudoku_grid_extractor.exceptions import GridDetectionError


class TestDetectGridWithSyntheticImage:
    """Requirements 2.1, 2.2: Grid detection returns a square perspective-corrected image."""

    def test_synthetic_grid_returns_square_numpy_array(self):
        """A white image with a black rectangle should be detected as a grid."""
        # Create a 500x500 white BGR image
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255

        # Draw a thick black rectangle to simulate a grid outline
        cv2.rectangle(img, (50, 50), (450, 450), (0, 0, 0), thickness=3)

        result = detect_grid(img)

        assert isinstance(result, np.ndarray)
        # Output should be a 900x900 square image
        assert result.shape[0] == result.shape[1]
        assert result.shape[0] == 900
        assert result.shape[1] == 900

    def test_synthetic_grid_result_has_three_channels(self):
        """The warped output should preserve the BGR color channels."""
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (60, 60), (440, 440), (0, 0, 0), thickness=4)

        result = detect_grid(img)

        assert len(result.shape) == 3
        assert result.shape[2] == 3


class TestDetectGridWithBlankImage:
    """Requirement 2.3: No grid found raises GridDetectionError."""

    def test_blank_white_image_raises_error(self):
        """A plain white image with no features should raise GridDetectionError."""
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255

        with pytest.raises(GridDetectionError):
            detect_grid(img)

    def test_blank_black_image_raises_error(self):
        """A plain black image with no features should raise GridDetectionError."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)

        with pytest.raises(GridDetectionError):
            detect_grid(img)
