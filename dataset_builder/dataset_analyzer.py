"""Dataset analysis and bias detection tools.

Checks YOLO-format label directories for class distribution, bounding-box
size statistics, and generates visualisation charts.

Usage (CLI):
    python -m dataset_builder.dataset_analyzer --labels data/cs2_dataset/train/labels
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ── Core analysis ────────────────────────────────────────────────────

def parse_yolo_labels(label_dir: str) -> Tuple[Counter, List[Tuple[float, float]]]:
    """Parse all YOLO ``.txt`` label files in *label_dir*.

    Returns:
        class_counts: Counter mapping class_id → number of instances.
        box_sizes:    List of (relative_width, relative_height) for every box.
    """
    label_path = Path(label_dir)
    if not label_path.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    class_counts: Counter = Counter()
    box_sizes: List[Tuple[float, float]] = []

    for txt in sorted(label_path.glob("*.txt")):
        with open(txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                class_counts[cls] += 1
                box_sizes.append((w, h))

    return class_counts, box_sizes


def class_distribution(
    label_dir: str,
    class_names: Dict[int, str] | None = None,
    save_path: str | None = None,
) -> Counter:
    """Print and optionally plot class distribution.

    Args:
        label_dir:   Path to YOLO labels directory.
        class_names: Optional mapping class_id → human name.
        save_path:   If given, save the bar chart to this file.

    Returns:
        Counter of class frequencies.
    """
    counts, _ = parse_yolo_labels(label_dir)

    print("\n── Class Distribution ──────────────────────────")
    total = sum(counts.values())
    for cls, n in sorted(counts.items()):
        name = class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}"
        pct = 100.0 * n / total if total else 0
        print(f"  {name:>20s}: {n:>6d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':>20s}: {total:>6d}")

    if save_path:
        names = [
            (class_names.get(c, f"class_{c}") if class_names else f"class_{c}")
            for c in sorted(counts)
        ]
        values = [counts[c] for c in sorted(counts)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(names, values, color="#1f6feb")
        ax.set_xlabel("Instances")
        ax.set_title("Class Distribution")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Chart saved → {save_path}")

    return counts


def box_size_analysis(
    label_dir: str,
    save_path: str | None = None,
) -> None:
    """Print bounding-box size statistics and optionally plot a scatter."""
    _, sizes = parse_yolo_labels(label_dir)
    if not sizes:
        print("No bounding boxes found.")
        return

    ws, hs = zip(*sizes)
    ws = np.array(ws)
    hs = np.array(hs)
    areas = ws * hs

    print("\n── Bounding Box Size Stats ────────────────────")
    print(f"  Count:  {len(sizes)}")
    print(f"  Width:  mean={ws.mean():.4f}  std={ws.std():.4f}  min={ws.min():.4f}  max={ws.max():.4f}")
    print(f"  Height: mean={hs.mean():.4f}  std={hs.std():.4f}  min={hs.min():.4f}  max={hs.max():.4f}")
    print(f"  Area:   mean={areas.mean():.4f}  std={areas.std():.4f}")

    if save_path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(ws, hs, s=4, alpha=0.4, color="#3fb950")
        ax.set_xlabel("Relative Width")
        ax.set_ylabel("Relative Height")
        ax.set_title("Bounding Box Size Distribution")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Chart saved → {save_path}")


def balance_report(label_dir: str, class_names: Dict[int, str] | None = None) -> Dict:
    """Analyse class balance and suggest oversampling weights.

    Returns:
        Dict with ``counts``, ``weights`` (inverse frequency), ``imbalance_ratio``.
    """
    counts, _ = parse_yolo_labels(label_dir)
    if not counts:
        return {"counts": {}, "weights": {}, "imbalance_ratio": 0}

    total = sum(counts.values())
    n_classes = len(counts)

    # Inverse-frequency weighting
    weights = {}
    for cls, n in counts.items():
        weights[cls] = (total / (n_classes * n)) if n > 0 else 1.0

    max_c = max(counts.values())
    min_c = min(counts.values())

    print("\n── Balance Report ─────────────────────────────")
    for cls in sorted(counts):
        name = class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}"
        print(f"  {name:>20s}: count={counts[cls]:>6d}  weight={weights[cls]:.3f}")
    ratio = max_c / min_c if min_c > 0 else float("inf")
    print(f"  Imbalance ratio (max/min): {ratio:.2f}")
    if ratio > 3:
        print("  ⚠ Significant imbalance — consider oversampling minority classes")

    return {"counts": dict(counts), "weights": weights, "imbalance_ratio": ratio}


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze YOLO dataset labels")
    parser.add_argument("--labels", required=True, help="Path to labels directory")
    parser.add_argument("--class-chart", default=None, help="Save class distribution chart")
    parser.add_argument("--box-chart", default=None, help="Save box size scatter chart")
    args = parser.parse_args()

    class_distribution(args.labels, save_path=args.class_chart)
    box_size_analysis(args.labels, save_path=args.box_chart)
    balance_report(args.labels)


if __name__ == "__main__":
    main()
