# Feature: sudoku-grid-extractor, Property 3: Grid validation accepts valid and rejects invalid matrices
"""Property-based tests for the GridValidator module."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sudoku_grid_extractor.grid_validator import validate_grid
from sudoku_grid_extractor.exceptions import GridValidationError


def grid_matrices():
    """Generate valid 9x9 matrices with values in [0, 9]."""
    cell = st.integers(min_value=0, max_value=9)
    row = st.lists(cell, min_size=9, max_size=9)
    return st.lists(row, min_size=9, max_size=9)


def invalid_grid_matrices():
    """Generate matrices violating at least one constraint.

    Produces matrices with:
    - Wrong number of rows (not 9)
    - Wrong number of columns in at least one row (not 9)
    - Out-of-range cell values (< 0 or > 9)
    """
    cell_valid = st.integers(min_value=0, max_value=9)
    row_valid = st.lists(cell_valid, min_size=9, max_size=9)

    # Wrong row count: 0-8 or 10-15 rows of valid rows
    wrong_row_count = st.integers(min_value=0, max_value=15).filter(lambda n: n != 9).flatmap(
        lambda n: st.lists(row_valid, min_size=n, max_size=n)
    )

    # Wrong column count in at least one row
    @st.composite
    def wrong_col_count(draw):
        bad_len = draw(st.integers(min_value=0, max_value=15).filter(lambda n: n != 9))
        bad_row = draw(st.lists(cell_valid, min_size=bad_len, max_size=bad_len))
        bad_idx = draw(st.integers(min_value=0, max_value=8))
        rows = [draw(row_valid) for _ in range(9)]
        rows[bad_idx] = bad_row
        return rows

    # Out-of-range values
    @st.composite
    def out_of_range_values(draw):
        bad_value = draw(st.one_of(
            st.integers(max_value=-1),
            st.integers(min_value=10),
        ))
        bad_row_idx = draw(st.integers(min_value=0, max_value=8))
        bad_col_idx = draw(st.integers(min_value=0, max_value=8))
        rows = [draw(row_valid) for _ in range(9)]
        rows[bad_row_idx] = list(rows[bad_row_idx])  # ensure mutable copy
        rows[bad_row_idx][bad_col_idx] = bad_value
        return rows

    return st.one_of(wrong_row_count, wrong_col_count(), out_of_range_values())


# **Validates: Requirements 4.1, 4.2, 4.3**
@given(matrix=grid_matrices())
@settings(max_examples=100)
def test_grid_validation_accepts_valid(matrix):
    """For any 9x9 matrix of integers where every value is in [0, 9],
    validate_grid should return the matrix unchanged."""
    result = validate_grid(matrix)
    assert result == matrix


# **Validates: Requirements 4.1, 4.2, 4.3**
@given(matrix=invalid_grid_matrices())
@settings(max_examples=100)
def test_grid_validation_rejects_invalid(matrix):
    """For any matrix that violates any constraint (wrong row count,
    wrong column count, or value outside [0, 9]), validate_grid should
    raise GridValidationError."""
    with pytest.raises(GridValidationError):
        validate_grid(matrix)
