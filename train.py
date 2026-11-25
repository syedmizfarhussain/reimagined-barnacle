"""Training code: fine-tune YOLO-World-S and train the closed-set YOLOv8-S baseline on the custom dataset.

Both models use the SAME data, epochs, image size, batch size, seed and default augmentations, so the
comparison is fair. The best checkpoint (chosen on the val split) is copied to weights/.

    python train.py                       # both models, 30 epochs
    python train.py --models yolo_world_s --epochs 50
    python train.py --epochs 1 --fraction 0.05   # quick check that training works

Attribution: written by our team; the training loop itself is the Ultralytics trainer
(WorldTrainer for YOLO-World, DetectionTrainer for YOLOv8), called through model.train().
"""
import argparse
import json
import shutil
import time

import yw_common as C

START_WEIGHTS = {"yolo_world_s": "yolov8s-worldv2.pt", "yolov8s": "yolov8s.pt"}

# Optimizer recipe of each model. Everything else (data, epochs, image size, batch, seed, augmentation) is shared.
# YOLO-World: the official fine-tuning config (yolo_world_v2_s_..._finetune_coco.py: AdamW, base_lr 2e-4,
# weight_decay 0.05). We also turn off the bias warm-up: Ultralytics starts biases at lr 0.1, which damaged the
# pretrained text-matching head (bias / logit scale) in our tests. With the Ultralytics defaults (AdamW lr 0.00125)
# the test mAP fell from 56.3 (zero-shot) to 33.5 after 5 epochs.
# YOLOv8: the standard Ultralytics recipe (optimizer="auto").
RECIPES = {
    "yolo_world_s": dict(optimizer="AdamW", lr0=2e-4, weight_decay=0.05, warmup_bias_lr=0.0),
    "yolov8s": dict(),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(START_WEIGHTS), choices=list(START_WEIGHTS))
    ap.add_argument("--data", default=None, help="Ultralytics data.yaml (default: download the custom dataset)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0, help="part of the training set to use (1.0 = all)")
    args = ap.parse_args()

    from ultralytics import YOLO, YOLOWorld
    C.set_seed()
    data_yaml = args.data or C.prepare_custom_dataset()
    C.copy_demo_samples(data_yaml)
    C.WEIGHTS_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)

    summary = {}
    for kind in args.models:
        model = YOLOWorld(START_WEIGHTS[kind]) if kind == "yolo_world_s" else YOLO(START_WEIGHTS[kind])
        t0 = time.time()
        model.train(data=str(data_yaml), epochs=args.epochs, imgsz=C.IMG_SIZE, batch=args.batch, seed=C.SEED,
                    deterministic=True, device=C.get_device(), workers=args.workers, fraction=args.fraction,
                    project=str(C.RESULTS_DIR / "train_runs"), name=kind, exist_ok=True, plots=True,
                    **RECIPES[kind])
        minutes = (time.time() - t0) / 60
        dst = C.TRAINED_WEIGHTS[kind]
        shutil.copy(model.trainer.best, dst)
        summary[kind] = dict(start_weights=START_WEIGHTS[kind], recipe=RECIPES[kind], epochs=args.epochs, batch=args.batch,
                             fraction=args.fraction, train_minutes=round(minutes, 2), hardware=C.hardware_name(),
                             run_dir=str(model.trainer.save_dir), weights=str(dst), sha256_12=C.file_sha256(dst))
        print(f"\n{kind}: {minutes:.1f} min -> {dst}")

    out = C.RESULTS_DIR / "train_summary.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**old, **summary}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
