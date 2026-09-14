"""
Parse Chula-RBC-12 point annotations into:
  1. Pseudo segmentation masks (fixed-radius circles per point) for U-Net training
  2. Single-cell classification crops for EfficientNet training

Dataset layout (from Chula-PIC-Lab/Chula-RBC-12-Dataset):
  Dataset/           -- RBC blood smear images, 640x480
  Label/             -- one .txt file per image, same base filename
                        each line: "x y type"  (type = int, see CLASS_NAMES)

IMPORTANT: this dataset provides POINT annotations only -- no ground-truth
segmentation masks. The masks generated here are a weak-supervision proxy
(fixed-radius circle per point), not true cell boundaries. Say this
explicitly in the README/demo video -- don't imply pixel-accurate ground
truth where none exists.
"""

import os
import re
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

# Class index -> name, per the dataset's own README (13 classes, not 12 --
# Ovalocyte and Elliptocyte are distinct classes, added at different times).
CLASS_NAMES = {
    0: "Normal",
    1: "Macrocyte",
    2: "Microcyte",
    3: "Spherocyte",
    4: "Target_cell",
    5: "Stomatocyte",
    6: "Ovalocyte",
    7: "Teardrop",
    8: "Burr_cell",
    9: "Schistocyte",
    10: "Uncategorised",
    11: "Hypochromia",
    12: "Elliptocyte",
}

# Pseudo-mask circle radius in pixels. Tune against the actual image scale
# once you have real images in hand -- RBCs at this magnification/resolution
# are roughly this order of size, but verify rather than trust this constant.
MASK_RADIUS_PX = 12

# Crop size (square, pixels) for classification patches, centered on each point.
CROP_SIZE_PX = 64


def _parse_label_line(line: str):
    """Parse one line of a label file: 'x y type', whitespace- or comma-separated."""
    parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
    if len(parts) != 3:
        return None
    x, y, cls = parts
    return float(x), float(y), int(cls)


def load_annotations(raw_dir: str):
    """
    Load all point annotations.

    Returns: dict[image_path] -> list of (x, y, class_id)
    Expects raw_dir/Dataset/*.{png,jpg,...} and raw_dir/Label/*.txt with
    matching base filenames.
    """
    raw_dir = Path(raw_dir)
    label_dir = raw_dir / "Label"
    dataset_dir = raw_dir / "Dataset"

    if not label_dir.exists() or not dataset_dir.exists():
        raise FileNotFoundError(
            f"Expected {dataset_dir} and {label_dir} to exist. "
            "Download the dataset from "
            "https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset "
            "and point raw_dir at its root."
        )

    annotations = {}
    for label_file in sorted(label_dir.glob("*.txt")):
        image_stem = label_file.stem
        matches = list(dataset_dir.glob(f"{image_stem}.*"))
        if not matches:
            print(f"WARNING: no image found for label {label_file.name}, skipping")
            continue
        image_path = matches[0]

        points = []
        with open(label_file, "r") as f:
            for line in f:
                parsed = _parse_label_line(line)
                if parsed is None:
                    if line.strip():
                        print(f"WARNING: unparseable line in {label_file.name}: {line!r}")
                    continue
                points.append(parsed)

        annotations[str(image_path)] = points

    return annotations


def make_segmentation_masks(annotations: dict, out_dir: str):
    """
    Generate one binary pseudo-mask per image: a filled circle of radius
    MASK_RADIUS_PX at every annotated point, regardless of class.
    (Binary cell-vs-background mask -- not per-class, since the goal here
    is separating overlapping cells, not classifying them.)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path, points in annotations.items():
        with Image.open(image_path) as img:
            w, h = img.size

        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for x, y, _cls in points:
            draw.ellipse(
                [x - MASK_RADIUS_PX, y - MASK_RADIUS_PX,
                 x + MASK_RADIUS_PX, y + MASK_RADIUS_PX],
                fill=255,
            )

        out_name = Path(image_path).stem + "_mask.png"
        mask.save(out_dir / out_name)

    print(f"Wrote {len(annotations)} pseudo-masks to {out_dir}")


def make_classification_crops(annotations: dict, out_dir: str):
    """
    Crop a CROP_SIZE_PX square around every annotated point, saved into
    per-class subfolders: out_dir/<ClassName>/<image_stem>_<x>_<y>.png
    """
    out_dir = Path(out_dir)
    half = CROP_SIZE_PX // 2
    counts = Counter()

    for image_path, points in annotations.items():
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            stem = Path(image_path).stem

            for x, y, cls in points:
                class_name = CLASS_NAMES.get(cls, f"unknown_{cls}")
                class_dir = out_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                left = int(x - half)
                top = int(y - half)
                right = left + CROP_SIZE_PX
                bottom = top + CROP_SIZE_PX

                # Pad at image edges rather than dropping edge cells.
                crop = Image.new("RGB", (CROP_SIZE_PX, CROP_SIZE_PX), (0, 0, 0))
                src_box = (max(left, 0), max(top, 0), min(right, w), min(bottom, h))
                dst_offset = (max(-left, 0), max(-top, 0))
                region = img.crop(src_box)
                crop.paste(region, dst_offset)

                out_name = f"{stem}_{int(x)}_{int(y)}.png"
                crop.save(class_dir / out_name)
                counts[class_name] += 1

    print(f"Wrote {sum(counts.values())} crops to {out_dir}")
    return counts


def report_class_distribution(annotations: dict, out_csv: str = None):
    """
    Print per-class counts and the majority:minority imbalance ratio.
    Optionally write counts to a CSV for the PRD/README.
    """
    counts = Counter()
    for points in annotations.values():
        for _x, _y, cls in points:
            counts[CLASS_NAMES.get(cls, f"unknown_{cls}")] += 1

    total = sum(counts.values())
    print(f"Total labeled cells: {total}")
    for name, count in counts.most_common():
        print(f"  {name:15s} {count:6d}  ({100 * count / total:.1f}%)")

    if counts:
        majority = counts.most_common(1)[0][1]
        minority = min(counts.values())
        print(f"Imbalance ratio (majority:minority): {majority / minority:.1f}:1")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "count", "pct"])
            for name, count in counts.most_common():
                writer.writerow([name, count, round(100 * count / total, 2)])

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="../data/raw",
                         help="Path to the downloaded dataset root (contains Dataset/ and Label/)")
    parser.add_argument("--out-dir", default="../data/processed",
                         help="Where to write masks/ and crops/")
    args = parser.parse_args()

    annotations = load_annotations(args.raw_dir)
    report_class_distribution(annotations, out_csv=os.path.join(args.out_dir, "class_distribution.csv"))
    make_segmentation_masks(annotations, os.path.join(args.out_dir, "masks"))
    make_classification_crops(annotations, os.path.join(args.out_dir, "crops"))
