"""Exception classes for the Sudoku Grid Extractor."""


class SudokuExtractorError(Exception):
    """Base exception for all sudoku grid extractor errors."""
    pass


class ImageLoadError(SudokuExtractorError):
    """Raised when an image cannot be loaded."""
    pass


class GridDetectionError(SudokuExtractorError):
    """Raised when no grid can be found in the image."""
    pass


class GridValidationError(SudokuExtractorError):
    """Raised when a GridMatrix fails structural validation."""
    pass


class OutputStoreError(SudokuExtractorError):
    """Raised on serialization/deserialization failures."""
    pass
