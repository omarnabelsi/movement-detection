"""Extract gesture landmarks from HaGRID annotation JSON files.

The HaGRID annotations already contain pre-computed MediaPipe hand landmarks,
so no image download is needed — only the small JSON files.

How it works
------------
1. Download the 6 gesture annotation JSONs from Kaggle:

       kaggle datasets download kapitanov/hagrid -f ann_subsample/one.json     --path C:\\hagrid_json
       kaggle datasets download kapitanov/hagrid -f ann_subsample/ok.json      --path C:\\hagrid_json
       kaggle datasets download kapitanov/hagrid -f ann_subsample/two_up.json  --path C:\\hagrid_json
       kaggle datasets download kapitanov/hagrid -f ann_subsample/fist.json    --path C:\\hagrid_json
       kaggle datasets download kapitanov/hagrid -f ann_subsample/like.json    --path C:\\hagrid_json
       kaggle datasets download kapitanov/hagrid -f ann_subsample/dislike.json --path C:\\hagrid_json

   Each file is only a few MB.

2. Run this script:

       python hagrid_json_to_csv.py --json-dir C:\\hagrid_json

JSON format used by HaGRID
--------------------------
{
  "<image_id>": {
    "labels": ["one"],
    "hand_landmarks": [
      [[x0, y0], [x1, y1], ..., [x20, y20]],   // first hand
      [...]                                       // optional second hand
    ]
  }
}

Landmarks are normalised to [0, 1] and already computed with MediaPipe —
the same representation used by hand_tracker.py and train_model.py.
"""

import argparse
import csv
import json
import os
import random

from utils import GESTURE_LABELS

# ---------------------------------------------------------------------------
# HaGRID JSON filename  →  our numeric gesture label
# ---------------------------------------------------------------------------
HAGRID_TO_LABEL: dict[str, int] = {
    "one":     0,   # move
    "ok":      1,   # click
    "two_up":  2,   # scroll
    "fist":    3,   # drag
    "like":    4,   # volume_up
    "dislike": 5,   # volume_down
}

DATASET_DIR = "dataset"
DEFAULT_OUTPUT = os.path.join(DATASET_DIR, "gestures.csv")
DEFAULT_MAX_PER_CLASS = 500   # 500 × 6 = 3 000 balanced samples


# ---------------------------------------------------------------------------
# CSV header (matches dataset_recorder.py and image_dataset_to_csv.py)
# ---------------------------------------------------------------------------
def _csv_header() -> list[str]:
    cols = []
    for i in range(21):
        cols += [f"x{i}", f"y{i}"]
    cols.append("label")
    return cols


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_landmarks_from_json(
    json_dir: str,
    output_csv: str = DEFAULT_OUTPUT,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
    seed: int = 42,
) -> None:
    """Load HaGRID annotation JSONs and write landmark rows to *output_csv*.

    Parameters
    ----------
    json_dir:
        Folder containing the downloaded JSON files (e.g. one.json, ok.json …).
        Files may be placed directly in *json_dir* or inside an
        ``ann_subsample/`` sub-folder — both layouts are handled.
    output_csv:
        Destination CSV path.  Rows are **appended** so webcam samples are
        preserved.
    max_per_class:
        Maximum landmark rows taken from each gesture class.
    seed:
        Random seed used for shuffling before capping.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    write_header = not os.path.exists(output_csv)
    total_written = 0

    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(_csv_header())

        for gesture_name, label_id in HAGRID_TO_LABEL.items():
            # Try direct path, then ann_subsample/ sub-folder
            candidates = [
                os.path.join(json_dir, f"{gesture_name}.json"),
                os.path.join(json_dir, "ann_subsample", f"{gesture_name}.json"),
            ]
            json_path = next((p for p in candidates if os.path.isfile(p)), None)

            if json_path is None:
                print(
                    f"  [SKIP] {gesture_name}.json not found in {json_dir} "
                    f"(tried {candidates})"
                )
                continue

            print(f"  Loading {gesture_name}.json  (label {label_id} = "
                  f"{GESTURE_LABELS[label_id]})  …", end="", flush=True)

            with open(json_path, encoding="utf-8") as f:
                data: dict = json.load(f)

            rows: list[list] = []
            skipped = 0

            for _img_id, annotation in data.items():
                labels = annotation.get("labels", [])
                landmarks_list = annotation.get("landmarks") or annotation.get("hand_landmarks")
                if not landmarks_list or not labels:
                    skipped += 1
                    continue

                # Find the hand whose label matches the target gesture class.
                # Each entry may contain multiple hands (one per bbox/label).
                hand = None
                for i, lbl in enumerate(labels):
                    if lbl == gesture_name and i < len(landmarks_list):
                        hand = landmarks_list[i]
                        break

                # Fallback: if no matching label, take the first hand
                if hand is None and landmarks_list:
                    hand = landmarks_list[0]

                if hand is None or len(hand) != 21:
                    skipped += 1
                    continue

                # Each point may be [x, y] or [x, y, z] — keep only x, y
                flat = []
                for pt in hand:
                    flat.append(float(pt[0]))
                    flat.append(float(pt[1]))

                rows.append(flat + [label_id])

            # Shuffle and cap
            random.shuffle(rows)
            rows = rows[:max_per_class]

            for row in rows:
                writer.writerow(row)

            total_written += len(rows)
            print(f" {len(rows)} rows written  ({skipped} skipped — no landmarks)")

    print(f"\nDone.  {total_written} total rows → {output_csv}")
    _print_label_summary(output_csv)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------
def _print_label_summary(csv_path: str) -> None:
    counts: dict[int, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            try:
                lbl = int(row[-1])
                counts[lbl] = counts.get(lbl, 0) + 1
            except (ValueError, IndexError):
                pass
    print("\nLabel distribution in CSV:")
    for lbl, name in GESTURE_LABELS.items():
        print(f"  {lbl}  {name:<12}  {counts.get(lbl, 0):>5} rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract HaGRID landmarks from annotation JSONs → gestures.csv"
    )
    p.add_argument(
        "--json-dir", required=True,
        help="Folder with downloaded HaGRID annotation JSON files"
    )
    p.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})"
    )
    p.add_argument(
        "--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"Max rows per gesture class (default: {DEFAULT_MAX_PER_CLASS})"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    extract_landmarks_from_json(
        json_dir=args.json_dir,
        output_csv=args.output,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
