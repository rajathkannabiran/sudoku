# Feature: sudoku-grid-extractor, Property 1: Invalid paths raise errors
"""Property-based tests for the ImageLoader module."""

import os
import string

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from sudoku_grid_extractor.image_loader import load_image, SUPPORTED_EXTENSIONS
from sudoku_grid_extractor.exceptions import ImageLoadError


def invalid_paths():
    """Generate strings that are not valid file paths or have unsupported extensions.

    Strategy combines:
    - Random text strings (not real file paths)
    - Paths with unsupported extensions (.txt, .pdf, .bmp, .gif, etc.)
    - Empty strings
    - Paths with special characters
    """
    random_text = st.text(
        alphabet=string.ascii_letters + string.digits + "_-/.",
        min_size=0,
        max_size=100,
    )

    unsupported_extensions = st.sampled_from([
        ".txt", ".pdf", ".bmp", ".gif", ".tiff", ".webp", ".svg", ".doc",
        ".csv", ".xml", ".html", ".py", ".md", ".zip", ".tar",
    ])

    filename_base = st.text(
        alphabet=string.ascii_letters + string.digits + "_-",
        min_size=1,
        max_size=50,
    )

    paths_with_bad_ext = st.builds(
        lambda name, ext: name + ext,
        filename_base,
        unsupported_extensions,
    )

    return st.one_of(random_text, paths_with_bad_ext)


# **Validates: Requirements 1.2**
@given(path=invalid_paths())
@settings(max_examples=100)
def test_invalid_paths_raise_error(path):
    """For any string that is not a path to an existing file with a supported
    extension, load_image should raise ImageLoadError."""
    # Ensure the generated path doesn't accidentally point to a real
    # supported-extension file
    assume(not (
        os.path.exists(path)
        and os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS
    ))

    with pytest.raises(ImageLoadError):
        load_image(path)
