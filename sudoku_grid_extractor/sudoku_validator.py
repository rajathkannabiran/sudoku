"""Sudoku puzzle validation via constraint checking and solve-based uniqueness.

Validates that an extracted grid represents a legitimate sudoku puzzle by:
1. Checking no duplicate non-zero digits in any row, column, or 3x3 box.
2. Attempting to solve the puzzle and counting solutions:
   - Exactly 1 solution  → valid extraction
   - 0 solutions         → a digit was likely misread (conflict)
   - Multiple solutions  → a digit was likely missed (read as 0)
"""

from __future__ import annotations
from dataclasses import dataclass
from sudoku_grid_extractor.models import GridMatrix


@dataclass
class ValidationResult:
    """Result of sudoku validation."""
    valid: bool
    solution_count: int  # 0, 1, or 2 (capped at 2 — we stop early)
    message: str


def _has_duplicates(grid: GridMatrix) -> list[str]:
    """Check for duplicate non-zero digits in rows, columns, and boxes."""
    errors: list[str] = []

    # Rows
    for r in range(9):
        seen: dict[int, int] = {}
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            if v in seen:
                errors.append(f"Row {r}: duplicate {v} at cols {seen[v]} and {c}")
            else:
                seen[v] = c

    # Columns
    for c in range(9):
        seen = {}
        for r in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            if v in seen:
                errors.append(f"Col {c}: duplicate {v} at rows {seen[v]} and {r}")
            else:
                seen[v] = r

    # 3x3 boxes
    for br in range(3):
        for bc in range(3):
            seen = {}
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    v = grid[r][c]
                    if v == 0:
                        continue
                    if v in seen:
                        errors.append(
                            f"Box ({br},{bc}): duplicate {v} at {seen[v]} and ({r},{c})"
                        )
                    else:
                        seen[v] = (r, c)

    return errors


def _count_solutions(grid: GridMatrix, max_count: int = 2) -> int:
    """Count solutions using backtracking. Stops early once max_count is reached."""
    # Work on a flat copy for speed
    board = [grid[r][c] for r in range(9) for c in range(9)]
    empties = [i for i in range(81) if board[i] == 0]
    count = [0]

    def _is_valid(pos: int, num: int) -> bool:
        r, c = divmod(pos, 9)
        # Check row
        start = r * 9
        for i in range(start, start + 9):
            if board[i] == num:
                return False
        # Check column
        for i in range(c, 81, 9):
            if board[i] == num:
                return False
        # Check 3x3 box
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                if board[(br + dr) * 9 + bc + dc] == num:
                    return False
        return True

    def _solve(idx: int) -> None:
        if count[0] >= max_count:
            return
        if idx == len(empties):
            count[0] += 1
            return
        pos = empties[idx]
        for num in range(1, 10):
            if _is_valid(pos, num):
                board[pos] = num
                _solve(idx + 1)
                board[pos] = 0
                if count[0] >= max_count:
                    return

    _solve(0)
    return count[0]


def validate_sudoku(grid: GridMatrix) -> ValidationResult:
    """Validate an extracted sudoku grid.

    Returns a ValidationResult with:
    - valid=True only if the puzzle has exactly 1 unique solution
    - solution_count: 0, 1, or 2 (capped)
    - message: human-readable explanation
    """
    # First: quick constraint check
    dupes = _has_duplicates(grid)
    if dupes:
        return ValidationResult(
            valid=False,
            solution_count=0,
            message=f"Constraint violations found: {'; '.join(dupes[:3])}"
            + (f" (and {len(dupes) - 3} more)" if len(dupes) > 3 else ""),
        )

    # Second: solve-based uniqueness check
    solutions = _count_solutions(grid, max_count=2)

    if solutions == 0:
        return ValidationResult(
            valid=False,
            solution_count=0,
            message="No solution exists — likely a digit was misread.",
        )
    elif solutions == 1:
        return ValidationResult(
            valid=True,
            solution_count=1,
            message="Valid puzzle with a unique solution.",
        )
    else:
        return ValidationResult(
            valid=False,
            solution_count=2,
            message="Multiple solutions found — likely a digit was missed (read as empty).",
        )
