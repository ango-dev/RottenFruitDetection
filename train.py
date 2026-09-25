"""Train YOLOv12n. The defaults are the settings used for models/fruit-6class.pt.

    python train.py                                                    # 6 classes, combined dataset
    python train.py --data datasets/combined-3class/data.yaml --name fruit-3class   # 3 classes
    python train.py --resume runs/detect/fruit-6class/weights/last.pt
"""

import argparse

from ultralytics import YOLO

# Ultralytics' default detection augmentation, written out so it's visible here
AUGMENT = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # hue / saturation / brightness jitter
    translate=0.1, scale=0.5, fliplr=0.5,
    mosaic=1.0, close_mosaic=10,  # mosaic is turned off for the last 10 epochs
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/combined/data.yaml")
    ap.add_argument("--model", default="yolo12n.pt", help="starting weights (COCO-pretrained YOLOv12n)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--workers", type=int, default=8,
                    help="data-loader processes; more is faster, but on Windows each costs 1-2 GB of memory")
    ap.add_argument("--name", default="fruit-6class", help="run folder under runs/detect/")
    ap.add_argument("--resume", metavar="LAST_PT", help="continue an interrupted run from its last.pt")
    args = ap.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                    workers=args.workers, name=args.name, **AUGMENT)
    print(f"best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
