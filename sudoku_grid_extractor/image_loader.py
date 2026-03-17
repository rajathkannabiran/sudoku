"""Image loading and validation for the Sudoku Grid Extractor."""

from pathlib import Path

import cv2
import numpy as np

from sudoku_grid_extractor.exceptions import ImageLoadError

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_image(file_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        file_path: Path to a PNG, JPG, or JPEG image file.

    Returns:
        The image as a NumPy array (BGR color space).

    Raises:
        ImageLoadError: If the file does not exist, has an unsupported
            extension, or cannot be decoded as an image.
    """
    path = Path(file_path)

    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file extension '{path.suffix}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    image = cv2.imread(str(path))
    if image is None:
        raise ImageLoadError(f"Failed to decode image: {path}")

    return image
