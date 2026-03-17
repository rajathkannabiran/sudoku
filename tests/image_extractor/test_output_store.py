"""Unit tests for the OutputStore module.

Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2
"""

import json

import pytest

from sudoku_grid_extractor.output_store import save_results, load_results
from sudoku_grid_extractor.models import ExtractionResult
from sudoku_grid_extractor.exceptions import OutputStoreError


SAMPLE_GRID = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

SAMPLE_RESULTS = [
    ExtractionResult(source_file="puzzle_01.png", grid=SAMPLE_GRID),
    ExtractionResult(source_file="puzzle_02.jpg", grid=[
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0],
    ]),
]


class TestSaveAndLoadResults:
    """Test saving and loading a known result list (Req 5.4, 6.1)."""

    def test_round_trip_preserves_data(self, tmp_path):
        path = tmp_path / "results.json"
        save_results(SAMPLE_RESULTS, path)
        loaded = load_results(path)

        assert len(loaded) == len(SAMPLE_RESULTS)
        for original, restored in zip(SAMPLE_RESULTS, loaded):
            assert restored.source_file == original.source_file
            assert restored.grid == original.grid

    def test_empty_list_round_trip(self, tmp_path):
        path = tmp_path / "empty.json"
        save_results([], path)
        loaded = load_results(path)
        assert loaded == []

    def test_single_result_round_trip(self, tmp_path):
        path = tmp_path / "single.json"
        single = [SAMPLE_RESULTS[0]]
        save_results(single, path)
        loaded = load_results(path)

        assert len(loaded) == 1
        assert loaded[0].source_file == "puzzle_01.png"
        assert loaded[0].grid == SAMPLE_GRID


class TestLoadNonExistentFile:
    """Test loading a non-existent file raises OutputStoreError (Req 6.2)."""

    def test_raises_on_missing_file(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        with pytest.raises(OutputStoreError, match="File not found"):
            load_results(path)


class TestLoadInvalidJSON:
    """Test loading a file with invalid JSON raises OutputStoreError (Req 6.2)."""

    def test_raises_on_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(OutputStoreError, match="Invalid JSON"):
            load_results(path)

    def test_raises_on_non_array_json(self, tmp_path):
        path = tmp_path / "object.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(OutputStoreError, match="Expected a JSON array"):
            load_results(path)

    def test_raises_on_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("", encoding="utf-8")
        with pytest.raises(OutputStoreError):
            load_results(path)


class TestJSONIndentation:
    """Test JSON output is indented (Req 5.3)."""

    def test_output_contains_newlines(self, tmp_path):
        path = tmp_path / "indented.json"
        save_results(SAMPLE_RESULTS, path)
        raw = path.read_text(encoding="utf-8")
        assert "\n" in raw

    def test_output_uses_2_space_indent(self, tmp_path):
        path = tmp_path / "indented.json"
        save_results([SAMPLE_RESULTS[0]], path)
        raw = path.read_text(encoding="utf-8")
        # 2-space indentation means lines starting with exactly 2 spaces
        lines = raw.splitlines()
        indented_lines = [l for l in lines if l.startswith("  ") and not l.startswith("    ")]
        assert len(indented_lines) > 0

    def test_output_is_valid_json(self, tmp_path):
        path = tmp_path / "indented.json"
        save_results(SAMPLE_RESULTS, path)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_json_structure_has_expected_keys(self, tmp_path):
        path = tmp_path / "indented.json"
        save_results(SAMPLE_RESULTS, path)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        for entry in data:
            assert "source_file" in entry
            assert "grid" in entry
            assert isinstance(entry["source_file"], str)
            assert isinstance(entry["grid"], list)
            assert len(entry["grid"]) == 9
            for row in entry["grid"]:
                assert len(row) == 9
