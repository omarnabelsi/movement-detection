"""Hyperparameter optimization for CS2 bot detection using Optuna.

Searches over learning rate, batch size, optimizer, confidence threshold,
and IoU threshold using Bayesian optimization (TPE sampler).

Usage (CLI):
    python -m trainer.hyperopt --data data/cs2_dataset/data.yaml --trials 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import optuna
import torch
import yaml
from ultralytics import YOLO


DEFAULT_TRIALS = 20
DEFAULT_MODEL = "yolov8n.pt"


def _objective(
    trial: optuna.Trial,
    data_yaml: str,
    model: str,
    epochs: int,
    device: str,
) -> float:
    """Single Optuna trial — train with sampled hyper-params and return val mAP50."""
    lr0 = trial.suggest_float("lr0", 1e-4, 0.05, log=True)
    batch = trial.suggest_categorical("batch", [8, 16, 32])
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    img_size = trial.suggest_categorical("imgsz", [416, 512, 640])

    print(f"\n[hyperopt] Trial {trial.number}: lr={lr0:.5f} batch={batch} "
          f"opt={optimizer} imgsz={img_size}")

    yolo = YOLO(model)
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        lr0=lr0,
        optimizer=optimizer,
        project="runs/hyperopt",
        name=f"trial_{trial.number}",
        device=device,
        verbose=False,
    )

    # Extract best mAP50 from results
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    map50 = metrics.get("metrics/mAP50(B)", 0.0)

    trial.set_user_attr("map50_95", metrics.get("metrics/mAP50-95(B)", 0.0))

    return map50


def optimize(
    data_yaml: str,
    model: str = DEFAULT_MODEL,
    n_trials: int = DEFAULT_TRIALS,
    epochs_per_trial: int = 15,
    device: str = "",
) -> dict:
    """Run Optuna hyper-parameter search.

    Args:
        data_yaml:        Path to YOLO ``data.yaml``.
        model:            Base model to fine-tune in each trial.
        n_trials:         Number of Optuna trials.
        epochs_per_trial: Short training per trial (enough to compare).
        device:           ``""`` = auto.

    Returns:
        Dict with ``best_params`` and ``best_value`` (mAP50).
    """
    if device == "":
        device = "0" if torch.cuda.is_available() else "cpu"

    study = optuna.create_study(
        study_name="cs2_yolo_hyperopt",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: _objective(trial, data_yaml, model, epochs_per_trial, device),
        n_trials=n_trials,
    )

    print("\n── Optuna Search Complete ──────────────────────")
    print(f"  Best mAP50:  {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Save best params
    out = Path("runs/hyperopt/best_params.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(study.best_params, f)
    print(f"  Saved → {out}")

    return {"best_params": study.best_params, "best_value": study.best_value}


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CS2 YOLO")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--epochs-per-trial", type=int, default=15)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    optimize(
        data_yaml=args.data,
        model=args.model,
        n_trials=args.trials,
        epochs_per_trial=args.epochs_per_trial,
        device=args.device,
    )


if __name__ == "__main__":
    main()
