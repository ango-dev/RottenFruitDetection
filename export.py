"""Export weights to OpenVINO IR for Intel CPU / GPU / NPU inference.

    python export.py --weights models/fruit-6class.pt

The NPU needs a static input shape, so the model is exported at a fixed 1x3x640x640.
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/fruit-6class.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    path = YOLO(args.weights).export(format="openvino", imgsz=args.imgsz, dynamic=False, batch=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
