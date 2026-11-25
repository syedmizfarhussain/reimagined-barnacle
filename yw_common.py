"""Shared code for the scripts (train.py, evaluate.py, infer.py) and the Streamlit app (app.py).

Code attribution
----------------
* Written by our team: everything in this file (dataset download and layout handling, COCO subset
  sampling, conversion of predictions to the COCO result format, evaluation wrappers, timing code).
* Library code we call but did not write: Ultralytics (YOLO-World / YOLOv8 models, training loop,
  letterbox, NMS, validator), pycocotools (official COCO evaluation), lvis (official LVIS evaluation).
* The YOLO-World weights (yolov8{s,m,l}-worldv2.pt) are the released weights of the authors,
  distributed by Ultralytics. We do not re-run the pre-training.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import yaml
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------- paths and settings
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "samples"

SEED = 42
IMG_SIZE = 640
CONF_EVAL = 0.001          # standard COCO evaluation setting
MAX_DET = 300
CONF_DEMO = 0.25           # normal score limit for the app and the speed test

# Our custom dataset (see the notebook, Section 3)
CUSTOM = dict(
    name="african-wildlife",
    url="https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip",
    names=["buffalo", "elephant", "rhino", "zebra"],
)

# The 80 COCO class names (official YOLO-World COCO texts). Used as the default open vocabulary in the app.
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Extra animal names for the app's default open vocabulary. In our tests, adding them made the pretrained
# model label a polar bear and a hyena correctly (they were "bear" and "sheep" with COCO names only).
EXTRA_ANIMALS = [
    "polar bear", "brown bear", "butterfly", "moth", "bee", "tiger", "lion", "leopard", "cheetah", "hyena",
    "wolf", "fox", "deer", "monkey", "gorilla", "kangaroo", "camel", "hippopotamus", "crocodile", "snake",
    "turtle", "frog", "eagle", "owl", "penguin", "dolphin", "whale", "shark", "fish", "rabbit", "squirrel",
    "goat", "pig", "donkey", "panda", "koala", "rhino", "buffalo",
]

# Trained weights produced by train.py (and by Section 12 of the notebook)
TRAINED_WEIGHTS = {
    "yolo_world_s": WEIGHTS_DIR / "yolo_world_s_custom.pt",   # YOLO-World-S fine-tuned on the custom data
    "yolov8s": WEIGHTS_DIR / "yolov8s_custom.pt",             # closed-set YOLOv8-S baseline
}

# ----------------------------------------------------------------------------- reported numbers
# Original authors' results, copied from the sources (NOT our results).
# Cheng et al., "YOLO-World", CVPR 2024, Table 2: zero-shot LVIS minival, Fixed AP,
# FPS on one NVIDIA V100 without TensorRT (re-parameterized model). v1 weights.
PAPER_LVIS_TABLE2 = [
    dict(model="YOLO-World-S", params_M=13, pretrain="O365,GoldG", fps_v100=74.1, AP=26.2, APr=19.1, APc=23.6, APf=29.8),
    dict(model="YOLO-World-M", params_M=29, pretrain="O365,GoldG", fps_v100=58.1, AP=31.0, APr=23.8, APc=29.2, APf=33.9),
    dict(model="YOLO-World-L", params_M=48, pretrain="O365,GoldG", fps_v100=52.0, AP=35.0, APr=27.1, APc=32.8, APf=38.3),
]
# Ultralytics documentation (docs.ultralytics.com/models/yolo-world), "Zero-shot Transfer on COCO Dataset",
# for the v2 weights that we use.
ULTRALYTICS_COCO_ZERO_SHOT = [
    dict(model="YOLO-World-S v2", weights="yolov8s-worldv2", AP=37.7, AP50=52.2, AP75=41.0),
    dict(model="YOLO-World-M v2", weights="yolov8m-worldv2", AP=43.0, AP50=58.4, AP75=46.8),
    dict(model="YOLO-World-L v2", weights="yolov8l-worldv2", AP=45.8, AP50=61.3, AP75=49.8),
]


# ----------------------------------------------------------------------------- small helpers
def get_device():
    """GPU number 0 if a CUDA GPU exists, otherwise "cpu" (the format Ultralytics wants)."""
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def hardware_name():
    """GPU name, or a readable CPU name (Windows reports "Intel64 Family 6 ..." otherwise)."""
    import platform
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        for junk in ["(R)", "(TM)", "12th Gen "]:
            name = name.replace(junk, "")
        return " ".join(name.split()) + " (CPU)"
    except (ImportError, OSError):
        return platform.processor() or "CPU"


def set_seed(seed=SEED):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def quiet():
    """Hide the long text that pycocotools and lvis print."""
    return contextlib.redirect_stdout(io.StringIO())


def file_sha256(path, n=12):
    """Short fingerprint of a weights file. The app shows it to prove which model made the prediction."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def download_file(url, dest, chunk=1 << 20):
    """Download url to dest once. A ".part" file is used so a broken download is never used."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name,
                                        disable=total < 5_000_000) as bar:
            for part in r.iter_content(chunk):
                f.write(part); bar.update(len(part))
    tmp.rename(dest)
    return dest


PROMPT_TEMPLATES = {"plain": "{}", "photo": "a photo of a {}", "object": "a {} object"}


def build_prompts(class_names, template="plain", overrides=None):
    """Class names -> prompts, e.g. "a photo of a {}". overrides replaces single classes: {"rhino": "rhinoceros"}."""
    t = PROMPT_TEMPLATES.get(template, template)
    overrides = overrides or {}
    return [overrides.get(n, t.format(n)) for n in class_names]


# ----------------------------------------------------------------------------- models
def load_model(weights, classes=None):
    """Load any of our checkpoints. Returns (model, is_open_vocabulary).

    YOLO-World checkpoints (pretrained or fine-tuned) accept a new vocabulary through set_classes();
    this runs the CLIP text encoder once (prompt then detect). A closed-set YOLOv8 has a fixed class list."""
    from ultralytics import YOLO, YOLOWorld
    weights = str(weights)
    model = YOLO(weights)
    is_world = type(model.model).__name__ == "WorldModel"
    if is_world and not isinstance(model, YOLOWorld):
        model = YOLOWorld(weights)
    if classes:
        if not is_world:
            raise ValueError("This is a closed-set YOLOv8 model: its classes are fixed by training.")
        if list(classes) != list(model.names.values()):   # same as the trained classes -> no need to run CLIP
            model.set_classes(list(classes))
    return model, is_world


def count_params(model):
    return sum(p.numel() for p in model.model.parameters()) / 1e6


# ----------------------------------------------------------------------------- custom dataset
def find_split_root(folder):
    """The folder that holds images/train (Ultralytics layout) inside the extracted zip, or None."""
    folder = Path(folder)
    if not folder.exists():
        return None
    for d in [folder, *sorted(p for p in folder.rglob("*") if p.is_dir())]:
        if (d / "images" / "train").is_dir():
            return d
    return None


def prepare_custom_dataset(out_dir=DATA_DIR / "custom"):
    """Download and unzip the custom dataset once, and write a data.yaml with absolute paths."""
    out_dir = Path(out_dir)
    if find_split_root(out_dir) is None:
        z = download_file(CUSTOM["url"], DATA_DIR / f"{CUSTOM['name']}.zip")
        with zipfile.ZipFile(z) as f:
            f.extractall(out_dir)
    root = find_split_root(out_dir)
    if root is None:
        raise FileNotFoundError(f"no images/train folder found in {out_dir}")
    cfg = {"path": str(root.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test" if (root / "images" / "test").is_dir() else "images/val",
           "names": {i: n for i, n in enumerate(CUSTOM["names"])}}
    data_yaml = out_dir / "data_abs.yaml"
    with open(data_yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return data_yaml


def copy_demo_samples(data_yaml, n=8, seed=SEED):
    """Copy a few TEST images (never used in training) into samples/ for the live demo."""
    import shutil
    cfg = yaml.safe_load(open(data_yaml))
    test_dir = Path(cfg["path"]) / cfg["test"]
    imgs = sorted(p for p in test_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(seed).shuffle(imgs)
    SAMPLES_DIR.mkdir(exist_ok=True)
    for p in imgs[:n]:
        shutil.copy(p, SAMPLES_DIR / f"test_{p.name}")
    return sorted(SAMPLES_DIR.glob("*"))


# ----------------------------------------------------------------------------- COCO subset
def prepare_coco_subset(n_images=500, seed=SEED, min_box_side=2.0, workers=16):
    """Download the val2017 annotation file and the SAME seeded random subset as the notebook (Section 2).
    Returns (ann_json path, list of (image path, image id))."""
    from concurrent.futures import ThreadPoolExecutor
    coco_dir = DATA_DIR / "coco"
    ann_zip = download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                            coco_dir / "annotations_trainval2017.zip")
    ann_json = coco_dir / "annotations" / "instances_val2017.json"
    if not ann_json.exists():
        with zipfile.ZipFile(ann_zip) as z:
            z.extract("annotations/instances_val2017.json", coco_dir)
    coco = json.load(open(ann_json))
    has_box = defaultdict(bool)
    for a in coco["annotations"]:              # same image filter as the notebook loader
        x, y, w, h = a["bbox"]
        if not a.get("iscrowd", 0) and w >= min_box_side and h >= min_box_side:
            has_box[a["image_id"]] = True
    info = {im["id"]: im for im in coco["images"]}
    ids = sorted(has_box)
    random.Random(seed).shuffle(ids)
    ids = ids[:n_images]
    img_dir = coco_dir / "val2017"

    def fetch(i):
        url = info[i]["coco_url"]
        try:
            return download_file(url, img_dir / url.split("/")[-1]), i
        except Exception:
            return None, i
    with ThreadPoolExecutor(workers) as ex:
        items = [(p, i) for p, i in tqdm(ex.map(fetch, ids), total=len(ids), desc="COCO images") if p]
    return ann_json, items


def coco_zero_shot_eval(model, ann_json, items):
    """Official COCO box evaluation (pycocotools) of an open-vocabulary model on our image subset.
    The model gets the 80 COCO class names as prompts. Returns (summary in %, AP per class in %)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with quiet():
        gt = COCO(str(ann_json))
    cat_ids = sorted(gt.getCatIds())
    names = [gt.cats[c]["name"] for c in cat_ids]
    model.set_classes(names)
    dets = []
    paths = [str(p) for p, _ in items]
    id_of = {str(p): i for p, i in items}
    for k in tqdm(range(0, len(paths), 16), desc="COCO predict", leave=False):
        for p, r in zip(paths[k:k + 16], model.predict(paths[k:k + 16], conf=CONF_EVAL, iou=0.7, max_det=MAX_DET,
                                                        imgsz=IMG_SIZE, device=get_device(), verbose=False)):
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append(dict(image_id=id_of[p], category_id=cat_ids[int(c)], score=round(s, 5),
                                 bbox=[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]))
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR100"]
    if not dets:
        return dict.fromkeys(keys, 0.0), {}
    with quiet():
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.params.imgIds = sorted(id_of.values())
        E.evaluate(); E.accumulate(); E.summarize()
    s = np.where(E.stats < 0, np.nan, E.stats * 100)
    summary = dict(zip(keys, [s[0], s[1], s[2], s[3], s[4], s[5], s[8]]))
    prec = E.eval["precision"][:, :, :, 0, -1]
    per_class = {names[k]: float(prec[:, :, k][prec[:, :, k] > -1].mean() * 100)
                 for k in range(len(names)) if (prec[:, :, k] > -1).any()}
    return summary, per_class


# ----------------------------------------------------------------------------- custom evaluation and speed
def ultra_val(model, data_yaml, split="test", name="val"):
    """Ultralytics evaluation on one split. Returns (summary in %, AP50-95 per class name in %)."""
    names = yaml.safe_load(open(data_yaml))["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    m = model.val(data=str(data_yaml), split=split, imgsz=IMG_SIZE, batch=8, conf=CONF_EVAL, iou=0.7,
                  device=get_device(), plots=False, verbose=False, workers=0,
                  project=str(RESULTS_DIR / "val_runs"), name=name, exist_ok=True)
    summary = dict(mAP50_95=m.box.map * 100, mAP50=m.box.map50 * 100, mAP75=m.box.map75 * 100,
                   precision=m.box.mp * 100, recall=m.box.mr * 100)
    per_class = {names[int(c)]: float(ap) * 100 for c, ap in zip(m.ap_class_index, m.box.ap)}
    return summary, per_class


def measure_speed(model, images, warmup=10, half=False):
    """Time per image at batch size 1 on images already in memory (BGR arrays), in ms."""
    import torch
    cuda = torch.cuda.is_available()
    kw = dict(imgsz=IMG_SIZE, conf=CONF_DEMO, device=get_device(), verbose=False)
    if half:
        kw["half"] = True
    for im in (images * warmup)[:warmup]:
        model.predict(im, **kw)
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    pre, inf, post, wall = [], [], [], []
    for im in images:
        t0 = time.perf_counter()
        r = model.predict(im, **kw)[0]
        if cuda:
            torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    return dict(pre_ms=float(np.mean(pre)), infer_ms=float(np.mean(inf)), post_ms=float(np.mean(post)),
                total_ms=float(np.mean(wall)), fps_model=1000 / float(np.mean(inf)),
                peak_gpu_mem_MB=torch.cuda.max_memory_allocated() / 2**20 if cuda else None)


# ----------------------------------------------------------------------------- inference (used by infer.py and app.py)
def letterbox_preview(img_bgr, imgsz=IMG_SIZE):
    """The image exactly as the model sees it: Ultralytics LetterBox with the settings predict() uses
    for one image (keep shape, scale the long side to imgsz, pad to a multiple of 32 with grey 114)."""
    from ultralytics.data.augment import LetterBox
    h, w = img_bgr.shape[:2]
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=32)
    out = lb(image=img_bgr)
    r = min(imgsz / h, imgsz / w)
    return out, dict(original_hw=(h, w), scale=round(r, 4), model_input_hw=out.shape[:2])


def run_inference(model, img_bgr, conf=CONF_DEMO, iou=0.7, imgsz=IMG_SIZE):
    """Input image (BGR array) -> list of detections + annotated image (BGR) + timing (ms)."""
    r = model.predict(img_bgr, conf=conf, iou=iou, imgsz=imgsz, device=get_device(), verbose=False)[0]
    dets = [dict(class_id=int(c), class_name=r.names[int(c)], confidence=round(float(s), 4),
                 box_xyxy=[round(v, 1) for v in b])
            for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
    return dets, r.plot(), dict(r.speed)
