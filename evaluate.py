"""Validate a model on a dataset split and save the metrics to results/.

    python evaluate.py                    # 6-class model, combined dataset, test split
    python evaluate.py --split val
    python evaluate.py --model models/fruit-3class.pt --data datasets/combined-3class/data.yaml
    python evaluate.py --model models/fruit-6class_openvino_model --device intel:cpu
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fruit-6class.pt")
    ap.add_argument("--data", default="datasets/combined/data.yaml")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    tag = data.parent.name if data.stem == "data" else f"{data.parent.name}-{data.stem.removeprefix('data_')}"
    name = f"{Path(args.model).stem}_{tag}_{args.split}"
    model = YOLO(args.model, task="detect")
    seen = {}
    model.add_callback("on_val_end", lambda v: seen.update(images=v.seen))
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device,
                  project="runs/val", name=name, exist_ok=True)

    report = {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "device": args.device,
        "images": seen.get("images"),
        "instances": int(m.nt_per_class.sum()),
        "precision": round(m.box.mp, 4),
        "recall": round(m.box.mr, 4),
        "mAP50": round(m.box.map50, 4),
        "mAP50-95": round(m.box.map, 4),
        "per_class": m.summary(decimals=4),
        "ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
    }
    out = Path("results") / f"eval_{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda o: o.item()))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
