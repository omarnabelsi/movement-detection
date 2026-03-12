"""Download CS2 Player Detection dataset from Roboflow Universe.

Usage (CLI):
    python -m dataset_builder.roboflow_download --api-key YOUR_KEY

The dataset is saved in YOLO format under ``data/cs2_dataset/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ── Default Roboflow coordinates ─────────────────────────────────────
WORKSPACE = "cs2-719ou"
PROJECT = "cs2-player-detection-ob7hc"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov8"
DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cs2_dataset")


def download_dataset(
    api_key: str,
    workspace: str = WORKSPACE,
    project: str = PROJECT,
    version: int = DEFAULT_VERSION,
    fmt: str = DEFAULT_FORMAT,
    dest: str = DEFAULT_DEST,
) -> Path:
    """Download the Roboflow dataset and return the local path.

    Args:
        api_key:   Roboflow API key (free tier works for public datasets).
        workspace: Roboflow workspace slug.
        project:   Roboflow project slug.
        version:   Dataset version number.
        fmt:       Export format (``yolov8``, ``yolov5``, ``coco``, …).
        dest:      Local directory to save the dataset.

    Returns:
        Path to the downloaded dataset root.
    """
    from roboflow import Roboflow

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"[roboflow] Connecting to workspace '{workspace}' …")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)

    print(f"[roboflow] Downloading version {version} in '{fmt}' format …")
    ds = proj.version(version).download(fmt, location=str(dest_path))

    print(f"[roboflow] Dataset saved to: {ds.location}")
    return Path(ds.location)


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download CS2 dataset from Roboflow")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT, dest="fmt")
    parser.add_argument("--dest", default=DEFAULT_DEST)
    args = parser.parse_args()

    download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        fmt=args.fmt,
        dest=args.dest,
    )


if __name__ == "__main__":
    main()
