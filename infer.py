"""Inference code: run a model on images and save the annotated image and a JSON file per image.

    python infer.py samples/test_1.jpg                                  # fine-tuned YOLO-World-S (default)
    python infer.py photo.jpg --weights weights/yolov8s_custom.pt       # closed-set baseline
    python infer.py photo.jpg --weights yolov8s-worldv2.pt --classes "giraffe, lion, safari jeep"
    python infer.py folder_of_images/ --conf 0.3

Pipeline: input image -> letterbox to 640 (Ultralytics LetterBox) -> model -> NMS -> boxes in original pixels.
Attribution: written by our team; it calls Ultralytics for the model and the NMS.
"""
import argparse
import json
from pathlib import Path

import cv2

import yw_common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="image file or folder")
    ap.add_argument("--weights", default=str(C.TRAINED_WEIGHTS["yolo_world_s"]))
    ap.add_argument("--classes", default=None, help="comma-separated prompts (YOLO-World models only)")
    ap.add_argument("--conf", type=float, default=C.CONF_DEMO)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else None
    model, is_world = C.load_model(args.weights, classes)
    src = Path(args.source)
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) \
        if src.is_dir() else [src]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"model: {args.weights} (sha256 {C.file_sha256(args.weights) if Path(args.weights).exists() else 'n/a'}) | "
          f"open vocabulary: {is_world} | classes: {list(model.names.values())}")

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print("cannot read", f); continue
        _, pre = C.letterbox_preview(img)
        dets, annotated, speed = C.run_inference(model, img, args.conf, args.iou)
        cv2.imwrite(str(out / f"{f.stem}_pred.jpg"), annotated)
        record = dict(image=str(f), weights=args.weights, preprocessing=pre, speed_ms=speed, detections=dets)
        (out / f"{f.stem}_pred.json").write_text(json.dumps(record, indent=1))
        summary = ", ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in dets) or "no detections"
        print(f"{f.name}: {summary}  ({speed['inference']:.1f} ms)")
    print("saved results to", out.resolve())


if __name__ == "__main__":
    main()
