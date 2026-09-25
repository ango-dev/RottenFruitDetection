"""Image and instance counts per split and class for a YOLO dataset.

    python dataset_stats.py --dedup                                          # the combined dataset
    python dataset_stats.py --data datasets/studio   # a folder works too

--dedup hashes every image and counts val/test images that also appear in train
(catches resized or re-encoded copies, not crops or flips).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset, img2label_paths

SPLITS = ("train", "val", "test")


def load_dataset(data):
    """Split paths and class names from a data.yaml, or from a dataset folder laid out as
    <split>/images with a data.yaml whose own paths are ignored (e.g. a Colab export)."""
    p = Path(data)
    if p.is_dir():
        names = yaml.safe_load((p / "data.yaml").read_text())["names"]
        splits = {s: p / s / "images" for s in SPLITS if (p / s / "images").is_dir()}
    else:
        d = check_det_dataset(data)
        names, splits = d["names"], {s: d[s] for s in SPLITS if d.get(s)}
    return splits, dict(enumerate(names)) if isinstance(names, list) else names


def list_images(split):
    """Image paths for a split given as a folder, a .txt file list, or a list of those."""
    files = []
    for p in map(Path, split if isinstance(split, list) else [split]):
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.suffix[1:].lower() in IMG_FORMATS]
        elif p.suffix == ".txt":
            files += [p.parent / line.strip() for line in p.read_text().splitlines() if line.strip()]
    return sorted(files)


def box_ok(xc, yc, w, h, tol=0.01):
    """Positive size and inside the image, for a normalised YOLO box."""
    return w > 0 and h > 0 and xc - w / 2 > -tol and yc - h / 2 > -tol and xc + w / 2 < 1 + tol and yc + h / 2 < 1 + tol


def count_split(images, names):
    per_class, backgrounds, invalid = Counter(), 0, 0
    for label in map(Path, img2label_paths([str(f) for f in images])):
        lines = label.read_text().splitlines() if label.exists() else []
        rows = [line.split() for line in lines if line.strip()]
        per_class.update(int(float(r[0])) for r in rows)
        invalid += sum(not box_ok(*map(float, r[1:5])) for r in rows)
        backgrounds += not rows
    counts = {name: per_class.pop(c, 0) for c, name in names.items()}
    counts.update({f"unknown id {c}": n for c, n in per_class.items()})  # ids missing from data.yaml
    return {"images": len(images), "instances": sum(counts.values()), "backgrounds": backgrounds,
            "invalid_boxes": invalid, "per_class": counts}


def dhash(path, size=16):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return np.packbits(img[:, 1:] > img[:, :-1]).tobytes()


def copies_of_train(images):
    train = {h for h in map(dhash, images["train"]) if h}
    return {s: sum(dhash(f) in train for f in images[s]) for s in ("val", "test") if s in images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/combined/data.yaml", help="data.yaml or dataset folder")
    ap.add_argument("--dedup", action="store_true", help="count val/test images that are copies of train images")
    ap.add_argument("--out", help="default: results/dataset_stats_<dataset folder>.json")
    args = ap.parse_args()
    data = Path(args.data)
    args.out = args.out or f"results/dataset_stats_{(data if data.is_dir() else data.parent).name}.json"

    splits, names = load_dataset(args.data)
    images = {s: list_images(path) for s, path in splits.items()}
    stats = {s: count_split(files, names) for s, files in images.items()}

    classes = list(stats["train"]["per_class"])
    print(f"{'split':<7}{'images':>8}{'instances':>11}{'empty':>7}{'bad boxes':>11}" + "".join(f"{c:>15}" for c in classes))
    for s, v in stats.items():
        row = "".join(f"{v['per_class'].get(c, 0):>15}" for c in classes)
        print(f"{s:<7}{v['images']:>8}{v['instances']:>11}{v['backgrounds']:>7}{v['invalid_boxes']:>11}{row}")
    stats["total_images"] = sum(v["images"] for v in stats.values())
    print(f"total images: {stats['total_images']}")

    if args.dedup:
        stats["copies_of_train_images"] = copies_of_train(images)
        print(f"images that also appear in train: {stats['copies_of_train_images']}")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
