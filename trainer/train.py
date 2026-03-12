"""YOLOv8 training wrapper for CS2 bot detection.

Wraps ``ultralytics`` training with sensible defaults, custom callbacks,
and easy integration with the rest of the pipeline.

Usage (CLI):
    python -m trainer.train --data data/cs2_dataset/data.yaml --epochs 100
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Optional

import torch
import yaml
from ultralytics import YOLO


# ── Default training settings ────────────────────────────────────────

DEFAULT_YOLO_MODEL = "yolov8n.pt"     # nano — fast training; upgrade to s/m for accuracy
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_IMG_SIZE = 640
DEFAULT_LR = 0.01
DEFAULT_OPTIMIZER = "SGD"


def _bootstrap_splits(dataset_root: Path) -> None:
    """Create minimal valid/test splits from train if they are missing."""
    tr_i = dataset_root / "train" / "images"
    tr_l = dataset_root / "train" / "labels"
    va_i = dataset_root / "valid" / "images"
    va_l = dataset_root / "valid" / "labels"
    te_i = dataset_root / "test" / "images"
    te_l = dataset_root / "test" / "labels"

    if not tr_i.is_dir():
        return

    # Only bootstrap when valid split is missing/empty.
    if va_i.is_dir() and any(va_i.iterdir()):
        return

    va_i.mkdir(parents=True, exist_ok=True)
    va_l.mkdir(parents=True, exist_ok=True)
    te_i.mkdir(parents=True, exist_ok=True)
    te_l.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in tr_i.iterdir() if p.is_file()])
    if not imgs:
        return

    random.seed(42)
    random.shuffle(imgs)
    n = len(imgs)
    n_val = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.05))
    val_imgs = imgs[:n_val]
    test_imgs = imgs[n_val:n_val + n_test]

    for subset, dst_i, dst_l in ((val_imgs, va_i, va_l), (test_imgs, te_i, te_l)):
        for img in subset:
            lbl = tr_l / f"{img.stem}.txt"
            shutil.copy2(img, dst_i / img.name)
            if lbl.exists():
                shutil.copy2(lbl, dst_l / lbl.name)

    print(f"[trainer] Bootstrapped splits: val={len(val_imgs)} test={len(test_imgs)}")


def _normalize_data_yaml(data_yaml: str) -> str:
    """Normalize Roboflow YAML into stable YOLO paths rooted at the YAML folder.

    This fixes common breakage after moving downloaded datasets.
    """
    data_file = Path(data_yaml)
    if not data_file.exists():
        return data_yaml

    with open(data_file, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    if not isinstance(meta, dict):
        return data_yaml

    dataset_root = data_file.parent.resolve()
    _bootstrap_splits(dataset_root)

    # Keep class metadata if present.
    names = meta.get("names", {0: "player"})
    if isinstance(names, list):
        names = list(names)

    normalized = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": meta.get("nc", len(names) if isinstance(names, list) else len(names.keys())),
        "names": names,
    }

    with open(data_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(normalized, f, sort_keys=False)

    print(f"[trainer] Normalized dataset YAML → {data_file}")
    return str(data_file)


def create_dataset_yaml(
    dataset_dir: str,
    class_names: Optional[dict] = None,
    output_path: str | None = None,
) -> str:
    """Generate a YOLO ``data.yaml`` for the given dataset directory.

    Expects the standard Roboflow directory layout::

        dataset_dir/
        ├── train/images/  train/labels/
        ├── valid/images/  valid/labels/
        └── test/images/   test/labels/   (optional)

    Returns:
        Path to the generated ``data.yaml``.
    """
    ds = Path(dataset_dir)
    if output_path is None:
        output_path = str(ds / "data.yaml")

    # Auto-detect class names from existing data.yaml if present
    existing = ds / "data.yaml"
    if class_names is None and existing.exists():
        with open(existing) as f:
            meta = yaml.safe_load(f)
            if isinstance(meta, dict) and "names" in meta:
                class_names = meta["names"]

    if class_names is None:
        class_names = {0: "player"}

    data = {
        "path": str(ds.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "names": class_names,
    }

    test_dir = ds / "test" / "images"
    if test_dir.is_dir() and any(test_dir.iterdir()):
        data["test"] = "test/images"

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"[trainer] data.yaml written → {output_path}")
    return output_path


def train(
    data_yaml: str,
    model: str = DEFAULT_YOLO_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    img_size: int = DEFAULT_IMG_SIZE,
    lr0: float = DEFAULT_LR,
    optimizer: str = DEFAULT_OPTIMIZER,
    project: str = "runs/detect",
    name: str = "cs2_bot",
    device: str = "",
    resume: bool = False,
    class_weights: Optional[dict] = None,
) -> Path:
    """Train a YOLOv8 model.

    Args:
        data_yaml:     Path to the YOLO ``data.yaml`` file.
        model:         Pre-trained model to fine-tune.
        epochs / batch / img_size / lr0 / optimizer: Training hyper-params.
        project / name: Where to save training artefacts.
        device:        ``""`` = auto.
        resume:        Resume interrupted training.
        class_weights: Not directly supported by ultralytics CLI but logged
                       for study script use.

    Returns:
        Path to the ``best.pt`` weights.
    """
    data_yaml = _normalize_data_yaml(data_yaml)

    if device == "":
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[trainer] Model: {model}  Device: {device}")
    print(f"[trainer] Epochs: {epochs}  Batch: {batch}  LR: {lr0}  Opt: {optimizer}")

    yolo = YOLO(model)
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        lr0=lr0,
        optimizer=optimizer,
        project=project,
        name=name,
        device=device,
        resume=resume,
        verbose=True,
    )

    save_dir = Path(getattr(getattr(yolo, "trainer", None), "save_dir", Path(project) / name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback: search recursively in case Ultralytics nested project folders.
        candidates = sorted(Path(project).glob("**/weights/best.pt"))
        if candidates:
            best_pt = candidates[-1]

    print(f"[trainer] Training complete → {best_pt}")
    return best_pt


def export_model(weights: str, fmt: str = "onnx") -> Path:
    """Export a trained model to ONNX / TensorRT / etc."""
    yolo = YOLO(weights)
    path = yolo.export(format=fmt)
    print(f"[trainer] Exported → {path}")
    return Path(path)


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for CS2 bot detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--optimizer", default=DEFAULT_OPTIMIZER)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="cs2_bot")
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        img_size=args.img_size,
        lr0=args.lr,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
