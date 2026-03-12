"""Model evaluation metrics for CS2 bot detection.

Computes precision, recall, F1 score, confusion matrix, and false-positive
rate by comparing YOLO predictions against ground-truth YOLO labels.

Usage (CLI):
    python -m evaluator.metrics --model runs/detect/cs2_bot/weights/best.pt \
                                --data data/cs2_dataset/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


BBox = Tuple[float, float, float, float]


# ═══════════════════════════════════════════════════════════════════════
# IoU helpers
# ═══════════════════════════════════════════════════════════════════════

def _iou(a: BBox, b: BBox) -> float:
    """IoU of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Core evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_predictions(
    gt_boxes: List[List[BBox]],
    gt_classes: List[List[int]],
    pred_boxes: List[List[BBox]],
    pred_classes: List[List[int]],
    pred_confs: List[List[float]],
    iou_threshold: float = 0.5,
    num_classes: int = 1,
) -> Dict:
    """Compute evaluation metrics across a dataset.

    Args:
        gt_boxes / gt_classes:   Ground truth per image.
        pred_boxes / pred_classes / pred_confs: Predictions per image.
        iou_threshold: IoU threshold for a TP match.
        num_classes: Number of object classes.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``fpr``,
        ``confusion_matrix``, ``per_class``.
    """
    # Confusion matrix: rows = GT, cols = predicted, extra row/col for background
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gt_b, gt_c, pr_b, pr_c, pr_conf in zip(
        gt_boxes, gt_classes, pred_boxes, pred_classes, pred_confs,
    ):
        matched_gt = set()
        matched_pr = set()

        # Sort predictions by confidence (highest first)
        order = sorted(range(len(pr_conf)), key=lambda i: pr_conf[i], reverse=True)

        for pi in order:
            best_iou = 0.0
            best_gi = -1
            for gi, (gb, gc) in enumerate(zip(gt_b, gt_c)):
                if gi in matched_gt:
                    continue
                iou_val = _iou(pr_b[pi], gb)
                if iou_val >= iou_threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi

            if best_gi >= 0:
                # True positive
                matched_gt.add(best_gi)
                matched_pr.add(pi)
                total_tp += 1
                cm[gt_c[best_gi]][pr_c[pi]] += 1
            else:
                # False positive
                total_fp += 1
                cm[num_classes][pr_c[pi]] += 1  # background predicted as class

        # Unmatched ground truths = false negatives
        for gi in range(len(gt_b)):
            if gi not in matched_gt:
                total_fn += 1
                cm[gt_c[gi]][num_classes] += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    fpr = total_fp / max(total_fp + total_tp, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "confusion_matrix": cm,
    }


def print_metrics(metrics: Dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n── Evaluation Metrics ─────────────────────────")
    # Support both custom metric keys and Ultralytics val() keys.
    precision = metrics.get("precision", metrics.get("precision_yolo"))
    recall = metrics.get("recall", metrics.get("recall_yolo"))
    f1 = metrics.get("f1")
    fpr = metrics.get("fpr")
    tp = metrics.get("tp")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    map50 = metrics.get("mAP50")
    map50_95 = metrics.get("mAP50_95")

    if precision is not None:
        print(f"  Precision:         {precision:.4f}")
    if recall is not None:
        print(f"  Recall:            {recall:.4f}")
    if f1 is not None:
        print(f"  F1 Score:          {f1:.4f}")
    if fpr is not None:
        print(f"  False Positive Rate: {fpr:.4f}")
    if map50 is not None:
        print(f"  mAP50:             {map50:.4f}")
    if map50_95 is not None:
        print(f"  mAP50-95:          {map50_95:.4f}")
    if tp is not None or fp is not None or fn is not None:
        print(f"  TP={tp}  FP={fp}  FN={fn}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
) -> None:
    """Plot and save a confusion matrix heatmap."""
    labels = class_names + ["background"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Evaluate a YOLO model on the validation/test set
# ═══════════════════════════════════════════════════════════════════════

def evaluate_yolo_model(
    model_path: str,
    data_yaml: str,
    split: str = "val",
    conf: float = 0.25,
    iou: float = 0.5,
    device: str = "",
    save_dir: str = "runs/evaluate",
) -> Dict:
    """Run YOLO validation and compute custom metrics.

    Uses ultralytics' built-in ``val()`` for the main mAP numbers, then
    computes our own precision / recall / confusion matrix for fine control.

    Returns:
        Dict of metrics.
    """
    import torch

    if device == "":
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        device=device,
        project=save_dir,
        name="cs2_eval",
        verbose=True,
    )

    metrics = {}
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        metrics["mAP50"] = rd.get("metrics/mAP50(B)", 0.0)
        metrics["mAP50_95"] = rd.get("metrics/mAP50-95(B)", 0.0)
        metrics["precision_yolo"] = rd.get("metrics/precision(B)", 0.0)
        metrics["recall_yolo"] = rd.get("metrics/recall(B)", 0.0)

    print_metrics(metrics) if metrics else print("[evaluator] No detailed metrics from val()")
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CS2 YOLO model")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    evaluate_yolo_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )


if __name__ == "__main__":
    main()
