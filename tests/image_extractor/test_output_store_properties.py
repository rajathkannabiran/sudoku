# Feature: sudoku-grid-extractor, Property 4: Serialization round-trip
# Feature: sudoku-grid-extractor, Property 5: Serialization output structure
# Feature: sudoku-grid-extractor, Property 6: Invalid JSON raises deserialization errors
"""Property-based tests for the OutputStore module."""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sudoku_grid_extractor.output_store import save_results, load_results
from sudoku_grid_extractor.models import ExtractionResult
from sudoku_grid_extractor.exceptions import OutputStoreError


# --- Strategies ---

def grid_matrices():
    """Generate valid 9x9 matrices with values in [0, 9]."""
    cell = st.integers(min_value=0, max_value=9)
    row = st.lists(cell, min_size=9, max_size=9)
    return st.lists(row, min_size=9, max_size=9)


def supported_filenames():
    """Generate filenames with .png, .jpg, or .jpeg extensions."""
    name = st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    )
    ext = st.sampled_from([".png", ".jpg", ".jpeg"])
    return st.tuples(name, ext).map(lambda t: t[0] + t[1])


def extraction_results():
    """Generate valid ExtractionResult objects with random filenames and valid grids."""
    return st.builds(ExtractionResult, source_file=supported_filenames(), grid=grid_matrices())


def invalid_json_strings():
    """Generate strings that are not valid JSON."""
    return st.one_of(
        st.just("{not valid json"),
        st.just("[{]"),
        st.just(""),
        st.just("{"),
        st.just("not json at all"),
        st.text(min_size=1, max_size=50).filter(lambda s: _is_invalid_json(s)),
    )


def _is_invalid_json(s: str) -> bool:
    """Check if a string is not valid JSON."""
    try:
        json.loads(s)
        return False
    except (json.JSONDecodeError, ValueError):
        return True


# --- Property 4: Serialization round-trip ---

# **Validates: Requirements 5.4, 6.1**
@given(results=st.lists(extraction_results(), min_size=0, max_size=5))
@settings(max_examples=100)
def test_serialization_round_trip(results):
    """For any list of valid ExtractionResult objects, serializing to JSON
    then deserializing should produce an equivalent list of results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.json"
        save_results(results, path)
        loaded = load_results(path)

        assert len(loaded) == len(results)
        for original, restored in zip(results, loaded):
            assert restored.source_file == original.source_file
            assert restored.grid == original.grid


# --- Property 5: Serialization output structure ---

# **Validates: Requirements 5.1, 5.2, 5.3**
@given(results=st.lists(extraction_results(), min_size=1, max_size=5))
@settings(max_examples=100)
def test_serialization_output_structure(results):
    """For any list of valid ExtractionResult objects, the serialized JSON
    should contain proper structure: each entry has 'source_file' string and
    'grid' 9x9 array, and the output contains newline characters (indentation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.json"
        save_results(results, path)

        raw = path.read_text(encoding="utf-8")

        # JSON should contain newlines (indentation/formatting)
        assert "\n" in raw

        # Parse and verify structure
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == len(results)

        for entry in data:
            assert isinstance(entry, dict)
            # "source_file" is a string
            assert "source_file" in entry
            assert isinstance(entry["source_file"], str)
            # "grid" is a 9x9 array
            assert "grid" in entry
            grid = entry["grid"]
            assert isinstance(grid, list)
            assert len(grid) == 9
            for row in grid:
                assert isinstance(row, list)
                assert len(row) == 9


# --- Property 6: Invalid JSON raises deserialization errors ---

# **Validates: Requirement 6.2**
@given(bad_json=invalid_json_strings())
@settings(max_examples=100)
def test_invalid_json_raises_error(bad_json):
    """For any invalid JSON string written to a file, or a non-existent file path,
    load_results should raise OutputStoreError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write invalid JSON to a file and try to load
        path = Path(tmpdir) / "bad.json"
        path.write_text(bad_json, encoding="utf-8")
        with pytest.raises(OutputStoreError):
            load_results(path)


def test_nonexistent_file_raises_error():
    """Loading from a non-existent file path should raise OutputStoreError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "does_not_exist.json"
        with pytest.raises(OutputStoreError):
            load_results(path)
