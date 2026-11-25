"""Evaluation code.

  custom  : trained checkpoints + zero-shot YOLO-World-S on the custom TEST split (never used in training)
  coco    : zero-shot COCO val2017 subset with the official pycocotools evaluation, compared with the
            numbers reported by Ultralytics for the same v2 weights
  speed   : latency / FPS / parameters of YOLO-World S, M and L (batch 1, 640 px)

    python evaluate.py custom
    python evaluate.py coco --sizes s m l --images 500
    python evaluate.py speed --sizes s m l

Tables are written to results/*.csv. The prompt study and LVIS evaluation are in the notebook
(Sections 9b and 10). Attribution: written by our team; it calls Ultralytics, pycocotools and lvis.
"""
import argparse

import cv2
import pandas as pd

import yw_common as C


def eval_custom(args):
    data_yaml = C.prepare_custom_dataset()
    rows = []
    zs, _ = C.load_model("yolov8s-worldv2.pt", C.CUSTOM["names"])
    candidates = [("YOLO-World-S zero-shot (no training)", zs)]
    for kind, label in [("yolo_world_s", "YOLO-World-S fine-tuned"), ("yolov8s", "YOLOv8-S fine-tuned (closed set)")]:
        if C.TRAINED_WEIGHTS[kind].exists():
            candidates.append((label, C.load_model(C.TRAINED_WEIGHTS[kind])[0]))
        else:
            print(f"skipping {label}: {C.TRAINED_WEIGHTS[kind]} not found (run train.py first)")
    for label, model in candidates:
        summary, per_cls = C.ultra_val(model, data_yaml, "test", name=label.split()[0] + "_" + str(len(rows)))
        rows.append(dict(model=label, params_M=round(C.count_params(model), 2), **summary,
                         **{f"AP {n}": per_cls.get(n) for n in C.CUSTOM["names"]}))
    df = pd.DataFrame(rows).round(2)
    df.to_csv(C.RESULTS_DIR / "custom_test_comparison.csv", index=False)
    print(df.to_string(index=False))


def eval_coco(args):
    ann_json, items = C.prepare_coco_subset(args.images)
    reported = {r["weights"]: r for r in C.ULTRALYTICS_COCO_ZERO_SHOT}
    rows = []
    for size in args.sizes:
        w = f"yolov8{size}-worldv2"
        model, _ = C.load_model(w + ".pt")
        ours, _ = C.coco_zero_shot_eval(model, ann_json, items)
        rep = reported.get(w, {})
        rows.append(dict(model=w, images=len(items), AP_ours=ours["AP"], AP_reported=rep.get("AP"),
                         AP50_ours=ours["AP50"], AP50_reported=rep.get("AP50"),
                         AP75_ours=ours["AP75"], AP75_reported=rep.get("AP75"),
                         APs=ours["APs"], APm=ours["APm"], APl=ours["APl"]))
        rows[-1]["AP_gap"] = rows[-1]["AP_ours"] - (rep.get("AP") or float("nan"))
    df = pd.DataFrame(rows).round(2)
    df.to_csv(C.RESULTS_DIR / "coco_zero_shot_vs_reported.csv", index=False)
    print(df.to_string(index=False))
    print("reported = Ultralytics docs, full val2017; ours = our seeded subset (see README).")


def eval_speed(args):
    ann_json, items = C.prepare_coco_subset(max(args.images, 100))
    imgs = [cv2.imread(str(p)) for p, _ in items[:100]]
    vocab = C.CUSTOM["names"] if args.custom_vocab else _coco_names(ann_json)
    paper = {r["model"][-1].lower(): r for r in C.PAPER_LVIS_TABLE2}
    rows = []
    for size in args.sizes:
        model, _ = C.load_model(f"yolov8{size}-worldv2.pt", vocab)    # vocabulary set once, before timing
        sp = C.measure_speed(model, imgs)
        rows.append(dict(model=f"YOLO-World-{size.upper()} v2", hardware=C.hardware_name(),
                         params_M=round(C.count_params(model), 2), **sp,
                         paper_fps_v100=paper.get(size, {}).get("fps_v100")))
    df = pd.DataFrame(rows).round(2)
    df.to_csv(C.RESULTS_DIR / "speed.csv", index=False)
    print(df.to_string(index=False))


def _coco_names(ann_json):
    """The 80 COCO class names in category-id order (the official YOLO-World COCO texts)."""
    import json
    cats = sorted(json.load(open(ann_json))["categories"], key=lambda c: c["id"])
    return [c["name"] for c in cats]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["custom", "coco", "speed"])
    ap.add_argument("--sizes", nargs="+", default=["s", "m", "l"], choices=["s", "m", "l", "x"])
    ap.add_argument("--images", type=int, default=500, help="COCO subset size")
    ap.add_argument("--custom-vocab", action="store_true", help="speed test with the 4 custom classes instead of 80")
    args = ap.parse_args()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    {"custom": eval_custom, "coco": eval_coco, "speed": eval_speed}[args.task](args)


if __name__ == "__main__":
    main()
