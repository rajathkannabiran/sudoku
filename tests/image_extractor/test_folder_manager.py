"""Unit tests for the FolderManager module.

Requirements: 7.1, 7.2, 7.3
"""

from pathlib import Path

from sudoku_grid_extractor.folder_manager import (
    DEFAULT_IMAGES_FOLDER,
    ensure_folder,
    list_images,
)


class TestDefaultFolderPath:
    """Requirement 7.1: Default folder path is images/."""

    def test_default_images_folder_value(self):
        assert DEFAULT_IMAGES_FOLDER == "images"


class TestEnsureFolder:
    """Requirement 7.2: Create directory when missing."""

    def test_creates_directory_when_missing(self, tmp_path):
        target = tmp_path / "new_folder"
        assert not target.exists()

        result = ensure_folder(target)

        assert target.is_dir()
        assert result == target

    def test_with_custom_path(self, tmp_path):
        custom = tmp_path / "custom" / "nested"
        result = ensure_folder(custom)

        assert custom.is_dir()
        assert result == custom

    def test_existing_directory_is_noop(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()

        result = ensure_folder(target)

        assert target.is_dir()
        assert result == target

    def test_returns_path_object(self, tmp_path):
        result = ensure_folder(str(tmp_path / "str_path"))
        assert isinstance(result, Path)


class TestListImages:
    """Requirement 7.3: List only supported image files, sorted."""

    def test_returns_only_supported_files(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"")
        (tmp_path / "b.jpg").write_bytes(b"")
        (tmp_path / "c.jpeg").write_bytes(b"")
        (tmp_path / "d.txt").write_bytes(b"")
        (tmp_path / "e.pdf").write_bytes(b"")

        result = list_images(tmp_path)

        names = [p.name for p in result]
        assert names == ["a.png", "b.jpg", "c.jpeg"]

    def test_results_are_sorted(self, tmp_path):
        (tmp_path / "z.png").write_bytes(b"")
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "m.jpeg").write_bytes(b"")

        result = list_images(tmp_path)

        names = [p.name for p in result]
        assert names == ["a.jpg", "m.jpeg", "z.png"]

    def test_empty_folder_returns_empty_list(self, tmp_path):
        result = list_images(tmp_path)
        assert result == []

    def test_ignores_subdirectories(self, tmp_path):
        (tmp_path / "subdir.png").mkdir()
        (tmp_path / "real.png").write_bytes(b"")

        result = list_images(tmp_path)

        assert len(result) == 1
        assert result[0].name == "real.png"
