"""Unit tests for the ImageLoader module.

Requirements: 1.1, 1.2, 1.3
"""

import numpy as np
import cv2
import pytest

from sudoku_grid_extractor.image_loader import load_image
from sudoku_grid_extractor.exceptions import ImageLoadError


class TestLoadValidImage:
    """Requirement 1.1: Valid file path to a PNG, JPG, or JPEG loads successfully."""

    def test_load_valid_png(self, tmp_path):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test_image.png"
        cv2.imwrite(str(path), img)

        result = load_image(str(path))

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_load_valid_jpg(self, tmp_path):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(path), img)

        result = load_image(str(path))

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0 and result.shape[1] > 0

    def test_load_valid_jpeg(self, tmp_path):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test_image.jpeg"
        cv2.imwrite(str(path), img)

        result = load_image(str(path))

        assert isinstance(result, np.ndarray)
        assert result.shape[0] > 0 and result.shape[1] > 0


class TestNonExistentPath:
    """Requirement 1.2: Invalid or non-existent path raises ImageLoadError."""

    def test_nonexistent_file(self):
        with pytest.raises(ImageLoadError, match="File not found"):
            load_image("/nonexistent/path/image.png")

    def test_nonexistent_relative_path(self):
        with pytest.raises(ImageLoadError, match="File not found"):
            load_image("does_not_exist.jpg")


class TestUnsupportedExtension:
    """Requirement 1.2: Unsupported format raises ImageLoadError."""

    def test_bmp_extension(self, tmp_path):
        path = tmp_path / "image.bmp"
        path.write_bytes(b"fake")

        with pytest.raises(ImageLoadError, match="Unsupported file extension"):
            load_image(str(path))

    def test_gif_extension(self, tmp_path):
        path = tmp_path / "image.gif"
        path.write_bytes(b"fake")

        with pytest.raises(ImageLoadError, match="Unsupported file extension"):
            load_image(str(path))

    def test_txt_extension(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("not an image")

        with pytest.raises(ImageLoadError, match="Unsupported file extension"):
            load_image(str(path))


class TestCorruptFile:
    """Requirement 1.3: Corrupt/unreadable image raises ImageLoadError."""

    def test_corrupt_png(self, tmp_path):
        path = tmp_path / "corrupt.png"
        path.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        with pytest.raises(ImageLoadError, match="Failed to decode"):
            load_image(str(path))

    def test_corrupt_jpg(self, tmp_path):
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not a real jpeg at all")

        with pytest.raises(ImageLoadError, match="Failed to decode"):
            load_image(str(path))
