"""Convert a HaGRID-style image dataset into the landmarks CSV used for training.

How it works
------------
1. Point --root at your HaGRID download folder (contains sub-folders like
   one/, ok/, fist/, two_up/, like/, dislike/).
2. Each sub-folder is mapped to one of our 6 gesture labels.
3. MediaPipe Hands runs in static-image mode (more accurate on still photos)
   to extract 21 landmarks → 42 normalised (x, y) values per image.
4. Images where no hand is detected are skipped automatically.
5. Samples are appended to dataset/gestures.csv — the same format used by
   dataset_recorder.py, so webcam and image samples can be mixed freely.

HaGRID class  →  our gesture label
-------------------------------------
  one          →  0  move
  ok           →  1  click
  two_up       →  2  scroll
  fist         →  3  drag
  like         →  4  volume_up
  dislike      →  5  volume_down

Usage
-----
    python image_dataset_to_csv.py --root C:\\hagrid
    python image_dataset_to_csv.py --root C:\\hagrid --max-per-class 500
    python image_dataset_to_csv.py --root C:\\hagrid --max-per-class 300 --output dataset/gestures.csv
"""

import argparse
import csv
import os
import random

import cv2
import mediapipe as mp

from utils import GESTURE_LABELS

# ---------------------------------------------------------------------------
# HaGRID subfolder  →  our numeric gesture label
# ---------------------------------------------------------------------------
HAGRID_TO_LABEL: dict[str, int] = {
    "one":     0,   # move
    "ok":      1,   # click
    "two_up":  2,   # scroll
    "fist":    3,   # drag
    "like":    4,   # volume_up
    "dislike": 5,   # volume_down
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_DIR = "dataset"
DEFAULT_OUTPUT = os.path.join(DATASET_DIR, "gestures.csv")
DEFAULT_MAX_PER_CLASS = 400   # 400 × 6 = 2 400 balanced samples; fast on CPU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_header() -> list[str]:
    header: list[str] = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    header.append("label")
    return header


def _collect_images(class_dir: str, max_count: int, rng: random.Random) -> list[str]:
    """Return up to *max_count* image paths, randomly sampled."""
    all_files = [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]
    if len(all_files) > max_count:
        all_files = rng.sample(all_files, max_count)
    return all_files


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_landmarks_from_images(
    root: str,
    output_csv: str = DEFAULT_OUTPUT,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = 42,
) -> None:
    """Run MediaPipe on sampled images and append landmark rows to CSV."""
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # static_image_mode=True gives better accuracy on individual photos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    file_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0
    total_written = 0
    counts: dict[int, int] = {i: 0 for i in GESTURE_LABELS}

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_csv_header())

        for hagrid_class, label_id in HAGRID_TO_LABEL.items():
            class_dir = os.path.join(root, hagrid_class)
            if not os.path.isdir(class_dir):
                print(f"  [SKIP] folder not found: {class_dir}")
                continue

            images = _collect_images(class_dir, max_per_class, rng)
            detected = skipped = 0
            print(f"\n[{GESTURE_LABELS[label_id]}]  {len(images)} images  ←  '{hagrid_class}/'")

            for img_path in images:
                bgr = cv2.imread(img_path)
                if bgr is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if not results.multi_hand_landmarks:
                    skipped += 1
                    continue

                coords: list[float] = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    coords.extend([lm.x, lm.y])

                writer.writerow(coords + [float(label_id)])
                detected += 1
                counts[label_id] += 1
                total_written += 1

            print(f"  saved {detected}  |  no-hand / unreadable: {skipped}")

    hands.close()

    print(f"\n{'='*52}")
    print(f"Done!  {total_written} samples written  →  {output_csv}")
    print("\nPer-gesture breakdown:")
    for lid, name in GESTURE_LABELS.items():
        print(f"  {name:15s}: {counts[lid]}")
    print("\nNext step:  python main.py --train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from a HaGRID image dataset"
    )
    parser.add_argument(
        "--root", required=True,
        help="HaGRID root directory (must contain one/, ok/, fist/, two_up/, like/, dislike/)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"Max images sampled per gesture class (default {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible image sampling")
    args = parser.parse_args()

    extract_landmarks_from_images(
        root=args.root,
        output_csv=args.output,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
