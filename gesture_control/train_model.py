"""Train a PyTorch gesture classifier on the recorded dataset.

Architecture:  42 → 128 (ReLU) → 64 (ReLU) → 6
Loss:          CrossEntropyLoss
Optimiser:     Adam

Usage:
    python train_model.py                 # defaults
    python train_model.py --epochs 100 --lr 0.001 --batch 64
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import INPUT_DIM, NUM_GESTURES

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pth")
DATASET_PATH = os.path.join("dataset", "gestures.csv")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GestureNet(nn.Module):
    """Feedforward classifier: 42 → 128 → 64 → 6."""

    def __init__(self, input_dim: int = INPUT_DIM, num_classes: int = NUM_GESTURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: str):
    """Load the CSV dataset and return (features, labels) numpy arrays."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    X = data[:, :-1]            # first 42 columns
    y = data[:, -1].astype(np.int64)  # last column
    return X, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(epochs: int = 60, lr: float = 1e-3, batch_size: int = 64):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Prefer CUDA (RTX 3070 Ti) when available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    X, y = load_dataset(DATASET_PATH)
    print(f"Dataset: {len(X)} samples, {NUM_GESTURES} gesture classes")

    # 80 / 20 train-validation split
    n = len(X)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X[idx[:split]], y[idx[:split]]
    X_val,   y_val   = X[idx[split:]], y[idx[split:]]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
    )

    model = GestureNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train --------------------------------------------------------
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(xb)

        # --- Validate -----------------------------------------------------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                total += len(yb)

        avg_loss = total_loss / max(len(X_train), 1)
        val_acc = correct / total if total else 0.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.2%}")

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved → {MODEL_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch)
