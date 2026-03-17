"""Output store for serializing and deserializing extraction results."""

import json
from pathlib import Path

from sudoku_grid_extractor.exceptions import OutputStoreError
from sudoku_grid_extractor.models import ExtractionResult, GridMatrix


def save_results(results: list[ExtractionResult], output_path: str | Path) -> None:
    """Serialize extraction results to a JSON file.

    The JSON structure is a list of objects, each with "source_file" (str)
    and "grid" (9x9 list of ints). Output is indented with 2 spaces.

    Args:
        results: List of extraction results to persist.
        output_path: Destination file path.

    Raises:
        OutputStoreError: If the file cannot be written.
    """
    output_path = Path(output_path)
    data = [
        {"source_file": r.source_file, "grid": r.grid}
        for r in results
    ]
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except (OSError, TypeError) as e:
        raise OutputStoreError(f"Failed to write results to {output_path}: {e}") from e


def load_results(output_path: str | Path) -> list[ExtractionResult]:
    """Deserialize extraction results from a JSON file.

    Args:
        output_path: Path to the JSON file.

    Returns:
        List of ExtractionResult entries.

    Raises:
        OutputStoreError: If the file does not exist or contains invalid JSON.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        raise OutputStoreError(f"File not found: {output_path}")
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise OutputStoreError(f"Invalid JSON in {output_path}: {e}") from e
    except OSError as e:
        raise OutputStoreError(f"Failed to read {output_path}: {e}") from e

    if not isinstance(data, list):
        raise OutputStoreError(
            f"Expected a JSON array in {output_path}, got {type(data).__name__}"
        )

    results = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise OutputStoreError(
                f"Expected object at index {i} in {output_path}, got {type(entry).__name__}"
            )
        if "source_file" not in entry or "grid" not in entry:
            raise OutputStoreError(
                f"Missing 'source_file' or 'grid' key at index {i} in {output_path}"
            )
        results.append(
            ExtractionResult(source_file=entry["source_file"], grid=entry["grid"])
        )
    return results
