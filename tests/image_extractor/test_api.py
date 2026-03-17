"""Unit tests for the public API (api.py) and package-level imports.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import cv2
import numpy as np
import pytest

import sudoku_grid_extractor
from sudoku_grid_extractor.api import extract_all, extract_grid


class TestExtractGrid:
    """Test extract_grid with a synthetic sudoku image (Requirement 8.1)."""

    def test_extract_grid_returns_valid_grid_matrix(self, tmp_path):
        """A synthetic image with a clear grid outline should produce a 9x9 matrix."""
        img_path = tmp_path / "grid.png"
        # White background with a thick black rectangle – enough for grid detection.
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (40, 40), (460, 460), (0, 0, 0), 4)
        cv2.imwrite(str(img_path), img)

        result = extract_grid(img_path)

        # Structural checks only – OCR on a blank grid won't produce real digits.
        assert isinstance(result, list)
        assert len(result) == 9
        for row in result:
            assert isinstance(row, list)
            assert len(row) == 9
            for val in row:
                assert isinstance(val, int)
                assert 0 <= val <= 9


class TestExtractAll:
    """Test extract_all batch processing (Requirement 8.2)."""

    def test_extract_all_returns_partial_results_on_errors(self, tmp_path):
        """One valid image + one corrupt file → partial results, no exception."""
        # Valid image with a detectable grid outline.
        valid_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(valid_img, (40, 40), (460, 460), (0, 0, 0), 4)
        cv2.imwrite(str(tmp_path / "good.png"), valid_img)

        # Corrupt PNG – a few random bytes.
        corrupt_path = tmp_path / "bad.png"
        corrupt_path.write_bytes(b"\x00\x01\x02\x03")

        results = extract_all(tmp_path)

        # Should contain at least the valid image's result without raising.
        assert isinstance(results, list)
        # The valid image should succeed; the corrupt one should be skipped.
        assert len(results) >= 1
        for r in results:
            assert hasattr(r, "source_file")
            assert hasattr(r, "grid")
            assert len(r.grid) == 9


class TestPublicImports:
    """Verify all public symbols are importable from the package root (Reqs 8.1-8.4)."""

    def test_extract_grid_importable(self):
        assert callable(sudoku_grid_extractor.extract_grid)

    def test_extract_all_importable(self):
        assert callable(sudoku_grid_extractor.extract_all)

    def test_save_results_importable(self):
        assert callable(sudoku_grid_extractor.save_results)

    def test_load_results_importable(self):
        assert callable(sudoku_grid_extractor.load_results)

    def test_extraction_result_importable(self):
        assert sudoku_grid_extractor.ExtractionResult is not None

    def test_grid_matrix_importable(self):
        assert sudoku_grid_extractor.GridMatrix is not None

    def test_exception_classes_importable(self):
        assert issubclass(sudoku_grid_extractor.SudokuExtractorError, Exception)
        assert issubclass(sudoku_grid_extractor.ImageLoadError, sudoku_grid_extractor.SudokuExtractorError)
        assert issubclass(sudoku_grid_extractor.GridDetectionError, sudoku_grid_extractor.SudokuExtractorError)
        assert issubclass(sudoku_grid_extractor.GridValidationError, sudoku_grid_extractor.SudokuExtractorError)
        assert issubclass(sudoku_grid_extractor.OutputStoreError, sudoku_grid_extractor.SudokuExtractorError)
