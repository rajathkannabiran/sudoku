# Feature: sudoku-grid-extractor, Property 7: Folder listing filters by supported extension
"""Property-based tests for the FolderManager module."""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from sudoku_grid_extractor.folder_manager import list_images
from sudoku_grid_extractor.image_loader import SUPPORTED_EXTENSIONS


# --- Strategies ---

UNSUPPORTED_EXTENSIONS = [".txt", ".pdf", ".bmp", ".gif", ".doc", ".csv", ".xml", ".svg"]


def _safe_filename_text():
    """Generate safe filename base strings (letters, digits, underscore, hyphen)."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=15,
    )


def supported_filenames():
    """Generate filenames with supported extensions (.png, .jpg, .jpeg)."""
    ext = st.sampled_from(sorted(SUPPORTED_EXTENSIONS))
    return st.tuples(_safe_filename_text(), ext).map(lambda t: t[0] + t[1])


def unsupported_filenames():
    """Generate filenames with unsupported extensions."""
    ext = st.sampled_from(UNSUPPORTED_EXTENSIONS)
    return st.tuples(_safe_filename_text(), ext).map(lambda t: t[0] + t[1])


def mixed_filenames():
    """Generate a list containing a mix of supported and unsupported filenames."""
    return st.tuples(
        st.lists(supported_filenames(), min_size=0, max_size=10),
        st.lists(unsupported_filenames(), min_size=0, max_size=10),
    ).filter(lambda t: len(t[0]) + len(t[1]) > 0)


# --- Property 7: Folder listing filters by supported extension ---

# **Validates: Requirement 7.3**
@given(data=mixed_filenames())
@settings(max_examples=100)
def test_folder_listing_filters_extensions(data):
    """For any directory containing a mix of files with supported and unsupported
    extensions, list_images should return only files with supported extensions,
    and the count of returned files should equal the count of supported-extension
    files in the directory."""
    supported_names, unsupported_names = data

    with tempfile.TemporaryDirectory() as tmpdir:
        # Deduplicate filenames (case-insensitive) to avoid collisions
        all_names = list(dict.fromkeys(supported_names + unsupported_names))
        created_supported = set()

        for name in all_names:
            filepath = Path(tmpdir) / name
            filepath.write_bytes(b"")
            if filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                created_supported.add(filepath.name)

        result = list_images(tmpdir)

        # All returned paths should have supported extensions
        for p in result:
            assert p.suffix.lower() in SUPPORTED_EXTENSIONS, (
                f"Returned file {p.name} has unsupported extension {p.suffix}"
            )

        # Count should match the number of supported-extension files created
        assert len(result) == len(created_supported), (
            f"Expected {len(created_supported)} supported files, got {len(result)}"
        )

        # Returned filenames should match exactly the supported files we created
        returned_names = {p.name for p in result}
        assert returned_names == created_supported
