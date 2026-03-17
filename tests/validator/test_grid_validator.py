"""Unit tests for the GridValidator module.

Requirements: 4.1, 4.2, 4.3
"""

import pytest

from sudoku_grid_extractor.grid_validator import validate_grid
from sudoku_grid_extractor.exceptions import GridValidationError


def _make_valid_grid(fill=0):
    """Return a valid 9x9 grid filled with the given value."""
    return [[fill] * 9 for _ in range(9)]


class TestValidateGridValid:
    """Tests that valid matrices pass validation."""

    def test_all_zeros(self):
        grid = _make_valid_grid(0)
        assert validate_grid(grid) == grid

    def test_all_nines(self):
        grid = _make_valid_grid(9)
        assert validate_grid(grid) == grid

    def test_mixed_values(self):
        grid = _make_valid_grid(0)
        grid[0] = [5, 3, 0, 0, 7, 0, 0, 0, 0]
        grid[1] = [6, 0, 0, 1, 9, 5, 0, 0, 0]
        result = validate_grid(grid)
        assert result is grid


class TestValidateGridWrongRowCount:
    """Tests that matrices with wrong row counts are rejected (Req 4.1)."""

    def test_eight_rows(self):
        grid = [[0] * 9 for _ in range(8)]
        with pytest.raises(GridValidationError, match="Expected 9 rows"):
            validate_grid(grid)

    def test_ten_rows(self):
        grid = [[0] * 9 for _ in range(10)]
        with pytest.raises(GridValidationError, match="Expected 9 rows"):
            validate_grid(grid)

    def test_zero_rows(self):
        with pytest.raises(GridValidationError, match="Expected 9 rows"):
            validate_grid([])


class TestValidateGridWrongColumnCount:
    """Tests that matrices with wrong column counts are rejected (Req 4.1)."""

    def test_row_with_eight_columns(self):
        grid = _make_valid_grid(0)
        grid[3] = [0] * 8
        with pytest.raises(GridValidationError, match="expected 9 columns"):
            validate_grid(grid)

    def test_row_with_ten_columns(self):
        grid = _make_valid_grid(0)
        grid[0] = [0] * 10
        with pytest.raises(GridValidationError, match="expected 9 columns"):
            validate_grid(grid)


class TestValidateGridOutOfRange:
    """Tests that out-of-range cell values are rejected (Req 4.2, 4.3)."""

    def test_value_ten(self):
        grid = _make_valid_grid(0)
        grid[4][4] = 10
        with pytest.raises(GridValidationError, match="outside range"):
            validate_grid(grid)

    def test_negative_value(self):
        grid = _make_valid_grid(0)
        grid[0][0] = -1
        with pytest.raises(GridValidationError, match="outside range"):
            validate_grid(grid)
