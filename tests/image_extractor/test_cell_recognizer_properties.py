# Feature: sudoku-grid-extractor, Property 2: Cell segmentation produces 81 cells
"""Property-based tests for the CellRecognizer module."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sudoku_grid_extractor.cell_recognizer import segment_cells


def square_images():
    """Generate square images of sufficient size (>= 9x9 pixels).

    Produces numpy arrays with random pixel values representing grayscale
    or BGR images of varying sizes.
    """
    size = st.integers(min_value=9, max_value=500)
    channels = st.sampled_from([None, 3])  # grayscale or BGR

    @st.composite
    def _build(draw):
        s = draw(size)
        ch = draw(channels)
        if ch is None:
            return np.random.randint(0, 256, (s, s), dtype=np.uint8)
        return np.random.randint(0, 256, (s, s, ch), dtype=np.uint8)

    return _build()


# **Validates: Requirements 3.1**
@given(image=square_images())
@settings(max_examples=100)
def test_segmentation_produces_81_cells(image):
    """For any square image of sufficient size (>= 9x9), segmenting it into
    a 9x9 grid should produce exactly 81 non-empty cell images, each with
    positive width and height."""
    cells = segment_cells(image)

    # Exactly 9 rows
    assert len(cells) == 9, f"Expected 9 rows, got {len(cells)}"

    total_cells = 0
    for row_idx, row in enumerate(cells):
        # Each row has exactly 9 cells
        assert len(row) == 9, f"Row {row_idx}: expected 9 cells, got {len(row)}"

        for col_idx, cell in enumerate(row):
            # Each cell is a numpy array with positive dimensions
            assert isinstance(cell, np.ndarray), (
                f"Cell ({row_idx},{col_idx}) is not a numpy array"
            )
            h, w = cell.shape[:2]
            assert h > 0, f"Cell ({row_idx},{col_idx}) has non-positive height: {h}"
            assert w > 0, f"Cell ({row_idx},{col_idx}) has non-positive width: {w}"
            total_cells += 1

    # Total cells should be 81
    assert total_cells == 81, f"Expected 81 total cells, got {total_cells}"
