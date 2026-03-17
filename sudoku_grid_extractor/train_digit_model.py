"""Train the digit CNN on synthetic printed digits + real cell images.

Usage:
    python -m sudoku_grid_extractor.train_digit_model

Produces: sudoku_grid_extractor/digit_model.pth
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sudoku_grid_extractor.digit_model import DigitCNN

_IMG_SIZE = 28
_MODEL_PATH = Path(__file__).parent / "digit_model.pth"


def _generate_synthetic_digits(samples_per_class: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic printed digit images with augmentation.

    Returns (images, labels) where images are (N, 28, 28) uint8 and
    labels are (N,) int.
    """
    images: list[np.ndarray] = []
    labels: list[int] = []

    # Fonts available in OpenCV
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    ]

    # Class 0: empty cells (white/near-white images with slight noise)
    for _ in range(samples_per_class):
        img = np.ones((_IMG_SIZE, _IMG_SIZE), dtype=np.uint8) * 255
        # Add slight noise
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Occasionally add a faint line (grid artifact)
        if random.random() < 0.3:
            thickness = random.randint(1, 2)
            color = random.randint(180, 230)
            if random.random() < 0.5:
                y = random.randint(0, _IMG_SIZE - 1)
                cv2.line(img, (0, y), (_IMG_SIZE, y), color, thickness)
            else:
                x = random.randint(0, _IMG_SIZE - 1)
                cv2.line(img, (x, 0), (x, _IMG_SIZE), color, thickness)
        images.append(img)
        labels.append(0)

    # Classes 1-9: printed digits
    for digit in range(1, 10):
        for _ in range(samples_per_class):
            canvas_size = 64
            img = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

            font = random.choice(fonts)
            scale = random.uniform(1.0, 2.0)
            thickness = random.randint(1, 3)
            text = str(digit)

            # Get text size and center it
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            x = (canvas_size - tw) // 2 + random.randint(-4, 4)
            y = (canvas_size + th) // 2 + random.randint(-4, 4)

            cv2.putText(img, text, (x, y), font, scale, 0, thickness, cv2.LINE_AA)

            # Random rotation (-10 to +10 degrees)
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((canvas_size / 2, canvas_size / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (canvas_size, canvas_size),
                                 borderValue=255)

            # Add noise
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Slight blur occasionally
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)

            # Resize to 28x28
            img = cv2.resize(img, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)

            images.append(img)
            labels.append(digit)

    return np.array(images), np.array(labels)


def _extract_real_cells() -> tuple[np.ndarray, np.ndarray]:
    """Extract labeled cells from the actual puzzle images.

    Uses the known correct grids from puzzle/puzzle.py as labels.
    Returns (images, labels) or empty arrays if images aren't available.
    """
    from sudoku_grid_extractor.image_loader import load_image
    from sudoku_grid_extractor.grid_detector import detect_grid
    from sudoku_grid_extractor.cell_recognizer import segment_cells, _to_gray, _crop_margins

    known_grids = [
        [
            [8, 0, 5, 0, 4, 0, 0, 7, 1],
            [0, 0, 0, 7, 0, 0, 9, 0, 3],
            [0, 0, 7, 2, 0, 5, 0, 6, 4],
            [0, 8, 2, 0, 3, 0, 0, 0, 6],
            [5, 3, 0, 0, 6, 0, 4, 2, 0],
            [0, 6, 0, 0, 7, 2, 5, 3, 0],
            [1, 0, 0, 4, 2, 0, 0, 0, 5],
            [9, 0, 0, 3, 0, 0, 7, 0, 0],
            [0, 5, 0, 0, 8, 7, 6, 0, 0],
        ],
        [
            [6, 0, 0, 7, 0, 2, 9, 0, 4],
            [8, 9, 0, 6, 4, 0, 0, 0, 0],
            [0, 2, 0, 1, 5, 0, 0, 6, 0],
            [0, 3, 8, 0, 1, 5, 0, 9, 2],
            [0, 1, 9, 3, 0, 0, 5, 0, 8],
            [0, 0, 0, 0, 9, 8, 0, 0, 1],
            [5, 0, 0, 8, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 7, 0, 2, 0, 3],
            [1, 8, 0, 0, 0, 4, 7, 0, 0],
        ],
    ]

    folder = Path("data/image")
    if not folder.exists():
        return np.array([]), np.array([])

    image_paths = sorted(folder.glob("*.png"))[:len(known_grids)]
    images: list[np.ndarray] = []
    labels_list: list[int] = []

    for img_path, grid_labels in zip(image_paths, known_grids):
        try:
            img = load_image(img_path)
            grid_img = detect_grid(img)
            cells = segment_cells(grid_img)
            for r in range(9):
                for c in range(9):
                    cell = cells[r][c]
                    gray = _to_gray(cell)
                    cropped = _crop_margins(gray)
                    resized = cv2.resize(cropped, (_IMG_SIZE, _IMG_SIZE),
                                         interpolation=cv2.INTER_AREA)
                    images.append(resized)
                    labels_list.append(grid_labels[r][c])

                    # Augment each real cell with slight variations
                    for _ in range(5):
                        aug = resized.copy()
                        noise = np.random.randint(-8, 8, aug.shape, dtype=np.int16)
                        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        if random.random() < 0.5:
                            angle = random.uniform(-5, 5)
                            M = cv2.getRotationMatrix2D(
                                (_IMG_SIZE / 2, _IMG_SIZE / 2), angle, 1.0
                            )
                            aug = cv2.warpAffine(aug, M, (_IMG_SIZE, _IMG_SIZE),
                                                 borderValue=255)
                        images.append(aug)
                        labels_list.append(grid_labels[r][c])
        except Exception as e:
            print(f"Warning: could not process {img_path}: {e}")

    if not images:
        return np.array([]), np.array([])
    return np.array(images), np.array(labels_list)


def _prepare_tensors(
    images: np.ndarray, labels: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize images to [0, 1] and convert to tensors."""
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
    y = torch.from_numpy(labels).long()
    return x, y


def train(epochs: int = 20, batch_size: int = 64, lr: float = 0.001) -> None:
    """Train the digit CNN and save to disk."""
    print("Generating synthetic training data...")
    syn_images, syn_labels = _generate_synthetic_digits(samples_per_class=3000)
    print(f"  Synthetic: {len(syn_images)} samples")

    print("Extracting real cell images...")
    real_images, real_labels = _extract_real_cells()
    print(f"  Real: {len(real_images)} samples")

    # Combine datasets
    if len(real_images) > 0:
        all_images = np.concatenate([syn_images, real_images])
        all_labels = np.concatenate([syn_labels, real_labels])
    else:
        all_images = syn_images
        all_labels = syn_labels

    # Shuffle
    idx = np.random.permutation(len(all_images))
    all_images = all_images[idx]
    all_labels = all_labels[idx]

    print(f"Total training samples: {len(all_images)}")
    for d in range(10):
        count = (all_labels == d).sum()
        print(f"  Class {d}: {count}")

    x, y = _prepare_tensors(all_images, all_labels)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_x.size(0)

        avg_loss = total_loss / total
        acc = correct / total * 100
        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.1f}%")

    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"\nModel saved to {_MODEL_PATH}")


if __name__ == "__main__":
    train()
