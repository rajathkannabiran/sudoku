"""CNN model for sudoku digit recognition.

A lightweight CNN that classifies 28×28 grayscale cell images into
10 classes: 0 (empty) or 1-9 (digit).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Small CNN for single-digit classification (10 classes: 0-9)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))       # -> (batch, 32, 28, 28)
        x = self.pool(x)                # -> (batch, 32, 14, 14)
        x = F.relu(self.conv2(x))       # -> (batch, 64, 14, 14)
        x = self.pool(x)                # -> (batch, 64, 7, 7)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
