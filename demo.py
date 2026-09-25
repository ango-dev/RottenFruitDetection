"""Webcam fresh/rotten fruit detection with OpenVINO, on the Intel NPU by default.

    python demo.py                              # 6-class model, webcam 0, NPU
    python demo.py --device intel:gpu           # or intel:cpu
    python demo.py --model models/fruit-3class_openvino_model   # apple / banana / orange only
    python demo.py --model models/fruit-6class.pt --device cpu   # no Intel hardware
    python demo.py --save-dir detections        # also save crops + detections.csv
    python demo.py --benchmark 300              # time inference on a still image, no camera

Press q to quit.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import openvino as ov
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS


def check_device(device):
    """Fail early if an OpenVINO device is missing. Ultralytics would otherwise
    fall back to AUTO with only a warning, and you'd be benchmarking the wrong chip."""
    core = ov.Core()
    info = {"host_cpu": core.get_property("CPU", "FULL_DEVICE_NAME").strip()}
    if str(device).startswith("intel"):
        name = device.split(":")[1].upper()
        if name not in core.available_devices:
            raise SystemExit(f"OpenVINO device {name} not found (available: {core.available_devices})")
        info["device_name"] = core.get_property(name, "FULL_DEVICE_NAME").strip()
    elif str(device) != "cpu" and torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
    return info


def execution_devices(model):
    compiled = getattr(model.predictor.model, "ov_compiled_model", None)
    return list(compiled.get_property("EXECUTION_DEVICES")) if compiled else None


def run(model, args):
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {args.source}")

    log = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(save_dir / "detections.csv", "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2", "crop"])

    n, fps, start, prev = 0, 0.0, None, None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]

            now = time.perf_counter()
            if prev is None:
                start = now  # the first frame includes model compilation, so timing starts after it
            else:
                fps = 1 / (now - prev) if fps == 0 else 0.9 * fps + 0.1 / (now - prev)
            prev = now

            if log:
                boxes = zip(r.boxes.xyxy.int().tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist())
                for i, ((x1, y1, x2, y2), conf, cls) in enumerate(boxes):
                    crop = f"{n:06d}_{i}_{r.names[cls]}.jpg"
                    if x2 > x1 and y2 > y1:
                        cv2.imwrite(str(save_dir / crop), frame[y1:y2, x1:x2])
                    log.writerow([n, r.names[cls], round(conf, 3), x1, y1, x2, y2, crop])
            n += 1

            if not args.no_show:
                out = r.plot()
                cv2.putText(out, f"{fps:.1f} FPS  {r.speed['inference']:.1f} ms", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("fruit detection", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log_file.close()
    if n > 1:
        print(f"{n} frames, {(n - 1) / (prev - start):.1f} FPS average end-to-end")


def benchmark(model, args, info):
    img = cv2.resize(cv2.imread(args.image or str(ASSETS / "bus.jpg")), (640, 480))  # webcam-sized frame
    for _ in range(10):  # warm-up
        model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)

    speeds, t0 = [], time.perf_counter()
    for _ in range(args.benchmark):
        speeds.append(model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0].speed)
    wall = time.perf_counter() - t0

    ms = {k: round(sum(s[k] for s in speeds) / len(speeds), 2) for k in ("preprocess", "inference", "postprocess")}
    report = {
        "model": args.model,
        "device": args.device,
        **info,
        "execution_devices": execution_devices(model),
        "runs": args.benchmark,
        "mean_ms": ms,
        "inference_fps": round(1000 / ms["inference"], 1),
        "pipeline_fps": round(args.benchmark / wall, 1),  # pre + inference + NMS, no camera or display
    }
    print(json.dumps(report, indent=2))
    tag = str(args.device).replace(":", "_")
    out = Path("results") / f"benchmark_{'cuda' + tag if tag.isdigit() else tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/fruit-6class_openvino_model")
    ap.add_argument("--device", default="intel:npu", help="intel:npu / intel:gpu / intel:cpu, or 0 / cpu for .pt")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-dir", help="save detection crops and detections.csv here")
    ap.add_argument("--no-show", action="store_true", help="don't open a preview window")
    ap.add_argument("--benchmark", type=int, metavar="N", help="time N inferences on a still image and exit")
    ap.add_argument("--image", help="image to use for --benchmark (default: a bundled sample)")
    args = ap.parse_args()

    info = check_device(args.device)
    print(f"{args.model} on {info.get('device_name', args.device)}")
    model = YOLO(args.model, task="detect")
    if args.benchmark:
        benchmark(model, args, info)
    else:
        run(model, args)


if __name__ == "__main__":
    main()
