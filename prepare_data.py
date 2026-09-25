"""Relabel a dataset to the models' 3- or 6-class scheme without touching the original.

    python prepare_data.py --classes 3    # datasets/combined -> datasets/combined-3class (apple / banana / orange)
    python prepare_data.py --src datasets/studio --classes 6 --fix-boxes --out datasets/studio-6class

Classes are matched by name (fresh_apple and rotten_apple both become apple for 3 classes);
images with other classes are dropped, background images are kept. Images are hard-linked,
not copied. data_clean.yaml points val/test at lists without copies of training images.

--fix-boxes is for the 51k studio-photo download only: its labels were converted to YOLO from
COCO-style (x_min, y_min, width, height) boxes as if those were (x_min, y_min, x_max, y_max),
which shifts every box and gives 30% of them a negative size. The conversion is exactly
invertible, so the boxes are rebuilt (see notebooks/04_studio_dataset.ipynb).
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml
from ultralytics.data.utils import img2label_paths

from dataset_stats import dhash, list_images, load_dataset

# class order of the released models
CLASSES = {
    3: ["apple", "banana", "orange"],
    6: ["rotten_apple", "rotten_banana", "rotten_orange", "fresh_apple", "fresh_banana", "fresh_orange"],
}


def fix_box(xc, yc, w, h):
    """Undo the (x, y, w, h)-read-as-(x1, y1, x2, y2) conversion. Returns a proper YOLO box."""
    x1, bw, y1, bh = xc - w / 2, xc + w / 2, yc - h / 2, yc + h / 2
    x1, y1 = max(x1, 0.0), max(y1, 0.0)
    bw, bh = min(bw, 1 - x1), min(bh, 1 - y1)
    return x1 + bw / 2, y1 + bh / 2, bw, bh


def link(src, dst):
    if not dst.exists():
        try:
            os.link(src, dst)
        except OSError:  # different drive or no hard-link support
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/combined/data.yaml", help="data.yaml or dataset folder (read-only)")
    ap.add_argument("--classes", type=int, choices=(3, 6), default=3)
    ap.add_argument("--out", help="output folder (default: datasets/combined<classes>)")
    ap.add_argument("--fix-boxes", action="store_true", help="rebuild the mis-converted studio boxes (see above)")
    args = ap.parse_args()

    src, src_names = load_dataset(args.src)
    names = CLASSES[args.classes]
    rename = (lambda n: n.split("_", 1)[1]) if args.classes == 3 else (lambda n: n)
    remap = {i: names.index(rename(n)) for i, n in src_names.items() if rename(n) in names}
    out = Path(args.out or f"datasets/combined{args.classes}")

    kept = {}
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        kept[split] = []
        images = list_images(src[split])
        for img, label in zip(images, map(Path, img2label_paths([str(f) for f in images]))):
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
            if any(int(float(r[0])) not in remap for r in rows):
                continue  # has a class the models don't use
            link(img, out / split / "images" / img.name)
            boxes = [fix_box(*map(float, r[1:5])) if args.fix_boxes else map(float, r[1:5]) for r in rows]
            lines = [f"{remap[int(float(r[0]))]} " + " ".join(f"{v:.6f}" for v in b) + "\n" for r, b in zip(rows, boxes)]
            (out / split / "labels" / f"{img.stem}.txt").write_text("".join(lines))
            kept[split].append(out / split / "images" / img.name)
        print(f"{split}: kept {len(kept[split])} of {len(images)} images")

    print("hashing images to find val/test copies of training images...")
    train = {dhash(f) for f in kept["train"]}
    for split in ("val", "test"):
        unique = [f for f in kept[split] if dhash(f) not in train]
        (out / f"{split}_clean.txt").write_text("".join(f"./{split}/images/{f.name}\n" for f in unique))
        print(f"{split}: {len(kept[split]) - len(unique)} copies left out of {split}_clean.txt ({len(unique)} remain)")

    for name, val, test in (("data.yaml", "val/images", "test/images"), ("data_clean.yaml", "val_clean.txt", "test_clean.txt")):
        cfg = {"names": names, "train": "train/images", "val": val, "test": test}
        (out / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out / 'data.yaml'} and {out / 'data_clean.yaml'}")


if __name__ == "__main__":
    main()
