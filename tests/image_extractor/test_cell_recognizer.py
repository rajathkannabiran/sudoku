"""Unit tests for the CellRecognizer module.

Tests cover:
- recognize_cells with a synthetic grid image containing known digits (Req 3.2, 3.4)
- Empty cells represented as 0 (Req 3.3)
- segment_cells returns correct 9x9 structure of numpy arrays (Req 3.1)
"""

import cv2
import numpy as np
import pytest

from sudoku_grid_extractor.cell_recognizer import recognize_cells, segment_cells


def _make_blank_grid(size: int = 450) -> np.ndarray:
    """Create a blank white image of the given size."""
    return np.full((size, size, 3), 255, dtype=np.uint8)


def _make_grid_with_digits(size: int = 450) -> np.ndarray:
    """Create a white image with grid lines and some digits drawn in cells.

    Draws a few known digits into specific cells using cv2.putText.
    """
    img = _make_blank_grid(size)
    cell_h = size // 9
    cell_w = size // 9

    # Draw grid lines so the image looks like a real grid.
    for i in range(10):
        y = i * cell_h
        x = i * cell_w
        cv2.line(img, (0, y), (size, y), (0, 0, 0), 2)
        cv2.line(img, (x, 0), (x, size), (0, 0, 0), 2)

    # Place some digits in specific cells.
    digits = {
        (0, 0): "5",
        (0, 4): "7",
        (1, 0): "6",
        (4, 4): "9",
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (r, c), digit in digits.items():
        x = c * cell_w + cell_w // 4
        y = (r + 1) * cell_h - cell_h // 4
        cv2.putText(img, digit, (x, y), font, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

    return img


class TestRecognizeCellsStructure:
    """Tests that recognize_cells returns a valid 9x9 GridMatrix."""

    def test_returns_9x9_matrix_with_digit_grid(self):
        """A synthetic grid with digits should produce a 9x9 matrix with values in [0,9]."""
        grid_img = _make_grid_with_digits()
        result = recognize_cells(grid_img)

        assert isinstance(result, list)
        assert len(result) == 9, f"Expected 9 rows, got {len(result)}"
        for row_idx, row in enumerate(result):
            assert isinstance(row, list)
            assert len(row) == 9, f"Row {row_idx}: expected 9 cols, got {len(row)}"
            for col_idx, val in enumerate(row):
                assert isinstance(val, int), (
                    f"Cell ({row_idx},{col_idx}) is not int: {type(val)}"
                )
                assert 0 <= val <= 9, (
                    f"Cell ({row_idx},{col_idx}) value {val} out of range [0,9]"
                )


class TestEmptyCells:
    """Tests that empty cells are represented as 0 (Req 3.3)."""

    def test_blank_image_produces_all_zeros(self):
        """A blank white image with no digits should yield all zeros."""
        blank = _make_blank_grid(450)
        result = recognize_cells(blank)

        assert len(result) == 9
        for row_idx, row in enumerate(result):
            assert len(row) == 9
            for col_idx, val in enumerate(row):
                assert val == 0, (
                    f"Cell ({row_idx},{col_idx}) expected 0 on blank image, got {val}"
                )


class TestSegmentCells:
    """Tests that segment_cells returns the right structure (Req 3.1)."""

    def test_returns_9x9_list_of_numpy_arrays(self):
        """segment_cells should return a 9x9 nested list of numpy arrays."""
        img = _make_blank_grid(450)
        cells = segment_cells(img)

        assert isinstance(cells, list)
        assert len(cells) == 9
        for row_idx, row in enumerate(cells):
            assert isinstance(row, list)
            assert len(row) == 9
            for col_idx, cell in enumerate(row):
                assert isinstance(cell, np.ndarray), (
                    f"Cell ({row_idx},{col_idx}) is not ndarray"
                )
                h, w = cell.shape[:2]
                assert h > 0 and w > 0, (
                    f"Cell ({row_idx},{col_idx}) has non-positive dims: {h}x{w}"
                )
