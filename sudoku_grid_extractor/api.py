"""Public API for the Sudoku Grid Extractor.

Orchestrates the full extraction pipeline and delegates persistence
to the output_store module.
"""

from pathlib import Path

from sudoku_grid_extractor.cell_recognizer import recognize_cells
from sudoku_grid_extractor.exceptions import SudokuExtractorError
from sudoku_grid_extractor.folder_manager import list_images
from sudoku_grid_extractor.grid_detector import detect_grid
from sudoku_grid_extractor.grid_validator import validate_grid
from sudoku_grid_extractor.image_loader import load_image
from sudoku_grid_extractor.models import ExtractionResult, GridMatrix
from sudoku_grid_extractor.output_store import (
    load_results as _load_results,
    save_results as _save_results,
)


def extract_grid(image_path: str | Path) -> GridMatrix:
    """Extract a GridMatrix from a single image file.

    Pipeline: load_image → detect_grid → recognize_cells → validate_grid.

    Args:
        image_path: Path to a PNG, JPG, or JPEG image file.

    Returns:
        A validated 9×9 GridMatrix.

    Raises:
        ImageLoadError: If the image cannot be loaded.
        GridDetectionError: If no grid is found in the image.
        GridValidationError: If the recognised matrix is structurally invalid.
    """
    image = load_image(image_path)
    grid_image = detect_grid(image)
    matrix = recognize_cells(grid_image)
    return validate_grid(matrix)


def extract_all(images_folder: str | Path | None = None) -> list[ExtractionResult]:
    """Batch-extract grids from all images in the folder.

    Iterates over every supported image file in *images_folder* (defaulting
    to ``images/``), calls :func:`extract_grid` for each, and collects the
    results.  Per-image errors are caught so that a single bad image does
    not abort the entire batch.

    Args:
        images_folder: Path to the images directory, or ``None`` to use
            the default ``images/`` folder.

    Returns:
        A list of :class:`ExtractionResult` entries — one per successfully
        processed image.
    """
    image_paths = list_images(images_folder)
    results: list[ExtractionResult] = []
    for path in image_paths:
        try:
            grid = extract_grid(path)
            results.append(ExtractionResult(source_file=path.name, grid=grid))
        except SudokuExtractorError:
            # Collect errors per-image and continue with the next one.
            continue
    return results


def save_results(results: list[ExtractionResult], output_path: str | Path) -> None:
    """Save extraction results to JSON.

    Delegates to :func:`sudoku_grid_extractor.output_store.save_results`.

    Args:
        results: List of extraction results to persist.
        output_path: Destination file path.

    Raises:
        OutputStoreError: If the file cannot be written.
    """
    _save_results(results, output_path)


def load_results(output_path: str | Path) -> list[ExtractionResult]:
    """Load extraction results from JSON.

    Delegates to :func:`sudoku_grid_extractor.output_store.load_results`.

    Args:
        output_path: Path to the JSON file.

    Returns:
        List of :class:`ExtractionResult` entries.

    Raises:
        OutputStoreError: If the file does not exist or contains invalid JSON.
    """
    return _load_results(output_path)
