"""Merge fruit datasets and no-fruit datasets (faces, scenes) into one YOLO dataset.

    python merge_datasets.py --out datasets/combined \
        --fruit Datasets/RottenFruitDataset3 Datasets/RottenFruitDataset4 \
        --negatives Datasets/FaceDataset1 Datasets/FaceDataset2 Datasets/BackgroundDataset1 Datasets/BackgroundDataset2

Fruit sources must already share one class list (the first one's data.yaml is used).
Negative images are copied with empty label files, so they teach the model what isn't
fruit. Sources are only read.
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml
from ultralytics.data.utils import IMG_FORMATS

SPLITS = ("train", "valid", "test")


def copy_split(src, dst, empty_labels):
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted((src / "images").glob("*")):
        if img.suffix[1:].lower() not in IMG_FORMATS:
            continue
        label = src / "labels" / f"{img.stem}.txt"
        shutil.copy2(img, dst / "images" / img.name)
        text = "" if empty_labels or not label.exists() else label.read_text()
        (dst / "labels" / f"{img.stem}.txt").write_text(text)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", nargs="+", required=True, help="datasets whose boxes are kept")
    ap.add_argument("--negatives", nargs="*", default=[], help="datasets copied with empty labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", help="also write per-source image counts to this JSON file")
    args = ap.parse_args()

    out, summary = Path(args.out), {}
    for src, empty in [(Path(p), False) for p in args.fruit] + [(Path(p), True) for p in args.negatives]:
        counts = {s: copy_split(src / s, out / s, empty) for s in SPLITS if (src / s / "images").is_dir()}
        summary[src.name] = {"role": "negative" if empty else "fruit", **counts}
        print(f"{src.name:<22} {'negative' if empty else 'fruit':<9} {counts}")
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2))

    names = yaml.safe_load((Path(args.fruit[0]) / "data.yaml").read_text())["names"]
    cfg = {"train": "train/images", "val": "valid/images", "test": "test/images", "names": names}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
