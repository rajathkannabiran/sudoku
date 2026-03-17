"""Extract sudoku grids from images, validate, and write to puzzle/puzzle.py.

Usage:
    python extract_puzzles.py [image_folder]

Defaults to data/image/ if no folder is provided.
Only puzzles with a unique solution are written. Failed images are reported.
"""

import sys
from pathlib import Path

from sudoku_grid_extractor.image_loader import load_image
from sudoku_grid_extractor.grid_detector import detect_grid
from sudoku_grid_extractor.cell_recognizer import recognize_cells
from sudoku_grid_extractor.sudoku_validator import validate_sudoku

PUZZLE_FILE = Path("puzzle/puzzle.py")


def _format_grid(grid: list[list[int]]) -> str:
    """Format a 9x9 grid as a Python list literal."""
    lines = []
    lines.append("    [")
    for row in grid:
        lines.append(f"        {row},")
    lines.append("    ]")
    return "\n".join(lines)


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/image")
    if not folder.exists():
        print(f"Error: folder {folder} does not exist.")
        sys.exit(1)

    extensions = ("*.png", "*.jpg", "*.jpeg")
    images: list[Path] = []
    for ext in extensions:
        images.extend(folder.glob(ext))
    images.sort()

    if not images:
        print(f"No images found in {folder}")
        sys.exit(1)

    print(f"Found {len(images)} images in {folder}\n")

    passed: list[tuple[Path, list[list[int]]]] = []
    failed: list[tuple[str, str]] = []

    for img_path in images:
        name = img_path.name
        try:
            img = load_image(img_path)
            grid_img = detect_grid(img)
            matrix = recognize_cells(grid_img)
            result = validate_sudoku(matrix)

            if result.valid:
                print(f"  PASS: {name}")
                passed.append((img_path, matrix))
            else:
                print(f"  FAIL: {name} — {result.message}")
                failed.append((name, result.message))
        except Exception as e:
            print(f"  ERROR: {name} — {e}")
            failed.append((name, str(e)))

    # Write passing puzzles to puzzle.py
    PUZZLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    grids_str = ",\n".join(_format_grid(g) for _, g in passed)
    content = f"question = [\n{grids_str}\n]\n"
    PUZZLE_FILE.write_text(content)

    # Delete successfully extracted images
    deleted = 0
    for img_path, _ in passed:
        try:
            img_path.unlink()
            deleted += 1
        except OSError as e:
            print(f"  Warning: could not delete {img_path.name}: {e}")

    print(f"\nDone: {len(passed)} passed, {len(failed)} failed.")
    print(f"Written {len(passed)} puzzles to {PUZZLE_FILE}")
    print(f"Deleted {deleted} processed images.")

    if failed:
        print("\nFailed images:")
        for name, reason in failed:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
