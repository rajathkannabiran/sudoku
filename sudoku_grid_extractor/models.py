"""Data models for the Sudoku Grid Extractor."""

from dataclasses import dataclass

GridMatrix = list[list[int]]


@dataclass
class ExtractionResult:
    """A single extraction result pairing a source filename with its grid."""
    source_file: str
    grid: GridMatrix
