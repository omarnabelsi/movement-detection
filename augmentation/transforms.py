"""Data augmentation pipeline optimised for CS2 gameplay frames.

Provides both:
- An Albumentations-based transform pipeline for offline dataset expansion.
- A lightweight OpenCV-only augmenter for real-time / fallback use.

Usage (CLI) — expand a YOLO dataset with augmented copies:
    python -m augmentation.transforms --images data/cs2_dataset/train/images \
                                      --labels data/cs2_dataset/train/labels \
                                      --output data/cs2_dataset_augmented \
                                      --copies 3
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Try albumentations; fall back to pure OpenCV if unavailable
try:
    import albumentations as A
    from albumentations import BboxParams
    _HAS_ALBUM = True
except ImportError:
    _HAS_ALBUM = False


# ═══════════════════════════════════════════════════════════════════════
# Albumentations pipeline (preferred)
# ═══════════════════════════════════════════════════════════════════════

def get_album_transform() -> "A.Compose":
    """Return an Albumentations augmentation pipeline tuned for CS2 frames."""
    if not _HAS_ALBUM:
        raise ImportError("albumentations is not installed")

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6,
            ),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(p=0.2),
            A.RandomScale(scale_limit=(-0.2, 0.3), p=0.4),
            A.RandomCrop(height=480, width=640, p=0.3),
            A.PadIfNeeded(min_height=480, min_width=640, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ],
        bbox_params=BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=0.3,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# Pure-OpenCV fallback augmenter
# ═══════════════════════════════════════════════════════════════════════

class CVAugmenter:
    """Lightweight OpenCV-only augmenter (no extra deps)."""

    @staticmethod
    def random_brightness(img: np.ndarray, limit: float = 0.25) -> np.ndarray:
        factor = 1.0 + random.uniform(-limit, limit)
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def random_contrast(img: np.ndarray, limit: float = 0.25) -> np.ndarray:
        factor = 1.0 + random.uniform(-limit, limit)
        mean = img.mean()
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def motion_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
        kernel = np.zeros((ksize, ksize))
        kernel[ksize // 2, :] = 1.0 / ksize
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def horizontal_flip(
        img: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Flip image and adjust YOLO bboxes (cx, cy, w, h)."""
        flipped = cv2.flip(img, 1)
        new_bboxes = [(1.0 - cx, cy, w, h) for cx, cy, w, h in bboxes]
        return flipped, new_bboxes

    @staticmethod
    def random_scale(img: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
        s = random.uniform(low, high)
        h, w = img.shape[:2]
        new_w, new_h = int(w * s), int(h * s)
        scaled = cv2.resize(img, (new_w, new_h))
        # Pad / crop back to original size
        canvas = np.zeros_like(img)
        ch, cw = min(new_h, h), min(new_w, w)
        canvas[:ch, :cw] = scaled[:ch, :cw]
        return canvas

    def augment(self, img: np.ndarray) -> np.ndarray:
        """Apply a random subset of augmentations (image only)."""
        if random.random() < 0.5:
            img = self.random_brightness(img)
        if random.random() < 0.5:
            img = self.random_contrast(img)
        if random.random() < 0.3:
            img = self.motion_blur(img, ksize=random.choice([3, 5, 7]))
        if random.random() < 0.4:
            img = self.random_scale(img)
        return img


# ═══════════════════════════════════════════════════════════════════════
# Offline dataset expansion
# ═══════════════════════════════════════════════════════════════════════

def _read_yolo_labels(path: Path) -> Tuple[List[int], List[List[float]]]:
    """Read a YOLO label file → (class_ids, [[cx, cy, w, h], …])."""
    classes, boxes = [], []
    if not path.exists():
        return classes, boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            classes.append(int(parts[0]))
            boxes.append([float(x) for x in parts[1:5]])
    return classes, boxes


def _write_yolo_labels(path: Path, classes: List[int], boxes: List[List[float]]) -> None:
    with open(path, "w") as f:
        for cls, box in zip(classes, boxes):
            f.write(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")


def expand_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    copies: int = 3,
) -> int:
    """Create augmented copies of every image+label pair.

    Returns the number of augmented images created.
    """
    img_path = Path(images_dir)
    lbl_path = Path(labels_dir)
    out_img = Path(output_dir) / "images"
    out_lbl = Path(output_dir) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    aug = get_album_transform() if _HAS_ALBUM else None
    cv_aug = CVAugmenter()
    count = 0

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(p for p in img_path.iterdir() if p.suffix.lower() in exts)
    print(f"[augmentation] Expanding {len(image_files)} images × {copies} copies → {output_dir}")

    for img_file in image_files:
        lbl_file = lbl_path / (img_file.stem + ".txt")
        classes, boxes = _read_yolo_labels(lbl_file)

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        for c in range(copies):
            suffix = f"_aug{c}"
            new_name = img_file.stem + suffix + img_file.suffix

            if aug and boxes:
                try:
                    result = aug(
                        image=img,
                        bboxes=boxes,
                        class_labels=classes,
                    )
                    aug_img = result["image"]
                    aug_boxes = [list(b) for b in result["bboxes"]]
                    aug_classes = result["class_labels"]
                except Exception:
                    aug_img = cv_aug.augment(img.copy())
                    aug_boxes = boxes
                    aug_classes = classes
            else:
                aug_img = cv_aug.augment(img.copy())
                aug_boxes = boxes
                aug_classes = classes

            cv2.imwrite(str(out_img / new_name), aug_img)
            _write_yolo_labels(out_lbl / (img_file.stem + suffix + ".txt"), aug_classes, aug_boxes)
            count += 1

    print(f"[augmentation] Created {count} augmented images")
    return count


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment YOLO dataset")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copies", type=int, default=3)
    args = parser.parse_args()
    expand_dataset(args.images, args.labels, args.output, args.copies)


if __name__ == "__main__":
    main()
