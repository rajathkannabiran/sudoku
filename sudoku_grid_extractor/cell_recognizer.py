"""Cell segmentation and digit recognition for the Sudoku Grid Extractor.

Uses a trained CNN (DigitCNN) for digit classification instead of OCR engines.
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from sudoku_grid_extractor.digit_model import DigitCNN
from sudoku_grid_extractor.models import GridMatrix

_GRID_SIZE = 9
_MARGIN_FRACTION = 0.15
_IMG_SIZE = 28
_MODEL_PATH = Path(__file__).parent / "digit_model.pth"

# Lazy-loaded model singleton
_model: DigitCNN | None = None


def _get_model() -> DigitCNN:
    """Load the trained CNN model (cached after first call)."""
    global _model
    if _model is None:
        _model = DigitCNN()
        _model.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu", weights_only=True))
        _model.eval()
    return _model


def segment_cells(grid_image: np.ndarray) -> list[list[np.ndarray]]:
    """Divide a grid image into a 9x9 array of cell images."""
    h, w = grid_image.shape[:2]
    cell_h = h // _GRID_SIZE
    cell_w = w // _GRID_SIZE
    rows: list[list[np.ndarray]] = []
    for r in range(_GRID_SIZE):
        row: list[np.ndarray] = []
        for c in range(_GRID_SIZE):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            row.append(grid_image[y1:y2, x1:x2])
        rows.append(row)
    return rows


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img


def _crop_margins(gray: np.ndarray, margin: float = _MARGIN_FRACTION) -> np.ndarray:
    h, w = gray.shape[:2]
    my, mx = int(h * margin), int(w * margin)
    c = gray[my : h - my, mx : w - mx]
    return c if c.size > 0 else gray


def _preprocess_for_cnn(cell_image: np.ndarray) -> torch.Tensor:
    """Preprocess a cell image for CNN inference.

    Matches the training pipeline: grayscale -> crop margins -> resize 28x28 -> [0,1].
    Returns a (1, 1, 28, 28) tensor ready for the model.
    """
    gray = _to_gray(cell_image)
    cropped = _crop_margins(gray)
    resized = cv2.resize(cropped, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor


def _recognize_digit(cell_image: np.ndarray) -> int:
    """Recognize a digit using the trained CNN model."""
    model = _get_model()
    tensor = _preprocess_for_cnn(cell_image)
    with torch.no_grad():
        output = model(tensor)
        predicted = output.argmax(dim=1).item()
    return predicted


def recognize_cells(grid_image: np.ndarray) -> GridMatrix:
    """Segment a grid image into 81 cells and recognize digits."""
    cells = segment_cells(grid_image)
    matrix: GridMatrix = []
    for row in cells:
        matrix_row: list[int] = []
        for cell in row:
            matrix_row.append(_recognize_digit(cell))
        matrix.append(matrix_row)
    return matrix
