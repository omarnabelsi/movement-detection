"""Real-time gesture classifier — loads the trained PyTorch model and
predicts gestures from a 42-value landmark vector.
"""

import os

import numpy as np
import torch

from train_model import GestureNet
from utils import GESTURE_LABELS, INPUT_DIM, NUM_GESTURES

MODEL_PATH = os.path.join("models", "gesture_model.pth")


class GestureClassifier:
    """Wrap the trained GestureNet for single-frame inference."""

    def __init__(self, model_path: str = MODEL_PATH, device: str | None = None):
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model = GestureNet(INPUT_DIM, NUM_GESTURES).to(self._device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self._model.eval()

    @torch.no_grad()
    def predict(self, landmarks: np.ndarray) -> tuple[int, str, float]:
        """Predict gesture from a 42-element landmark vector.

        Returns:
            (label_id, label_name, confidence)
        """
        x = torch.from_numpy(landmarks).float().unsqueeze(0).to(self._device)
        logits = self._model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        label = idx.item()
        return label, GESTURE_LABELS[label], conf.item()
