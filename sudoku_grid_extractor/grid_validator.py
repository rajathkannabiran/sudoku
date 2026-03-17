"""Grid validation for the Sudoku Grid Extractor."""

from sudoku_grid_extractor.models import GridMatrix
from sudoku_grid_extractor.exceptions import GridValidationError


def validate_grid(matrix: GridMatrix) -> GridMatrix:
    """
    Validate that a matrix is a structurally correct 9x9 sudoku grid.

    Checks:
        - Exactly 9 rows.
        - Each row has exactly 9 columns.
        - Every cell value is an integer in [0, 9].

    Returns:
        The same matrix if valid.

    Raises:
        GridValidationError: With a message describing the first violation found.
    """
    if not isinstance(matrix, list):
        raise GridValidationError(
            f"Expected a list of rows, got {type(matrix).__name__}"
        )

    if len(matrix) != 9:
        raise GridValidationError(
            f"Expected 9 rows, got {len(matrix)}"
        )

    for row_idx, row in enumerate(matrix):
        if not isinstance(row, list):
            raise GridValidationError(
                f"Row {row_idx}: expected a list, got {type(row).__name__}"
            )
        if len(row) != 9:
            raise GridValidationError(
                f"Row {row_idx}: expected 9 columns, got {len(row)}"
            )
        for col_idx, value in enumerate(row):
            if not isinstance(value, int):
                raise GridValidationError(
                    f"Cell ({row_idx}, {col_idx}): expected int, got {type(value).__name__}"
                )
            if value < 0 or value > 9:
                raise GridValidationError(
                    f"Cell ({row_idx}, {col_idx}): value {value} is outside range [0, 9]"
                )

    return matrix
