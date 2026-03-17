"""Images folder management for the Sudoku Grid Extractor."""

from pathlib import Path

from sudoku_grid_extractor.image_loader import SUPPORTED_EXTENSIONS

DEFAULT_IMAGES_FOLDER = "images"


def ensure_folder(folder_path: str | Path | None = None) -> Path:
    """
    Ensure the images folder exists, creating it if necessary.

    Args:
        folder_path: Custom path, or None to use the default "images/" directory.

    Returns:
        The resolved Path to the images folder.
    """
    path = Path(folder_path) if folder_path is not None else Path(DEFAULT_IMAGES_FOLDER)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder_path: str | Path | None = None) -> list[Path]:
    """
    List all supported image files in the images folder.

    Ensures the folder exists first, then returns a sorted list of files
    with supported extensions (.png, .jpg, .jpeg).

    Args:
        folder_path: Custom path, or None to use the default "images/" directory.

    Returns:
        Sorted list of Paths to PNG/JPG/JPEG files.
    """
    path = ensure_folder(folder_path)
    return sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
