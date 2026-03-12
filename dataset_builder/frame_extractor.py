"""Extract frames from CS2 gameplay videos for labeling/training.

Usage (CLI):
    python -m dataset_builder.frame_extractor --video path/to/gameplay.mp4

Frames are written as numbered PNGs into ``data/extracted_frames/``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "extracted_frames",
)


def extract_frames(
    video_path: str,
    output_dir: str = DEFAULT_OUTPUT,
    every_n: int = 10,
    max_frames: int = 0,
    resize_width: int = 0,
) -> int:
    """Extract every *every_n*-th frame from a video file.

    Args:
        video_path:   Path to the gameplay video.
        output_dir:   Directory to save extracted frames.
        every_n:      Save one frame every *every_n* frames (e.g. 10 → ~3 fps for 30 fps video).
        max_frames:   Stop after saving this many frames (0 = no limit).
        resize_width: Resize frames to this width (0 = keep original).

    Returns:
        Number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    idx = 0
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[frame_extractor] Video: {video_path}  ({total} total frames)")
    print(f"[frame_extractor] Saving every {every_n}-th frame → {output_dir}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % every_n == 0:
            if resize_width > 0:
                h, w = frame.shape[:2]
                r = resize_width / w
                frame = cv2.resize(frame, (resize_width, int(h * r)))

            fname = out / f"frame_{idx:06d}.png"
            cv2.imwrite(str(fname), frame)
            saved += 1

            if saved % 100 == 0:
                print(f"  … {saved} frames saved")

            if max_frames and saved >= max_frames:
                break

        idx += 1

    cap.release()
    print(f"[frame_extractor] Done — {saved} frames saved to {output_dir}")
    return saved


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from gameplay video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--every-n", type=int, default=10, help="Save every N-th frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to save (0=all)")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize width (0=original)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.every_n, args.max_frames, args.resize_width)


if __name__ == "__main__":
    main()
