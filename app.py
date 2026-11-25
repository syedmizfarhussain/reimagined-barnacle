"""Deployment: Streamlit web app for live inference.

    streamlit run app.py

Shows every stage of the pipeline: Input -> Preprocessing -> Trained Model -> Prediction -> Output.
Inputs: file upload, webcam photo, image URL, or a held-out test image from samples/.
Attribution: written by our team; the model, letterbox and NMS come from Ultralytics.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

import yw_common as C

st.set_page_config(page_title="YOLO-World Detector", page_icon="🔎", layout="wide")

MODEL_CHOICES = {
    "YOLO-World-S fine-tuned (our trained model)": str(C.TRAINED_WEIGHTS["yolo_world_s"]),
    "YOLOv8-S fine-tuned (closed-set baseline)": str(C.TRAINED_WEIGHTS["yolov8s"]),
    "YOLO-World-S pretrained (open vocabulary)": "yolov8s-worldv2.pt",
}


@st.cache_resource(show_spinner="Loading model ...")
def get_model(weights, classes):
    """Cached per (weights, vocabulary). set_classes() encodes the prompts once (prompt then detect)."""
    model, is_world = C.load_model(weights, list(classes) if classes else None)
    sha = C.file_sha256(weights) if Path(weights).exists() else "downloaded by Ultralytics"
    return model, is_world, sha, C.count_params(model)


def read_input():
    """Return the input image as a BGR array (or None) and a short description of where it came from."""
    tab_up, tab_cam, tab_url, tab_sample = st.tabs(["Upload", "Camera", "Image URL", "Test samples"])
    with tab_up:
        f = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if f is not None:
            return Image.open(f).convert("RGB"), f"upload: {f.name}"
    with tab_cam:
        cam = st.camera_input("Take a photo")
        if cam is not None:
            return Image.open(cam).convert("RGB"), "camera"
    with tab_url:
        url = st.text_input("Image URL (http/https)")
        if url:
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "yolo-world-demo"})
                r.raise_for_status()
                return Image.open(io.BytesIO(r.content)).convert("RGB"), f"url: {url}"
            except Exception as e:
                st.error(f"Could not load the image: {e}")
    with tab_sample:
        samples = sorted(p for p in C.SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if C.SAMPLES_DIR.exists() else []
        if samples:
            pick = st.selectbox("Held-out test image (never used in training)", ["-"] + [p.name for p in samples])
            if pick != "-":
                return Image.open(C.SAMPLES_DIR / pick).convert("RGB"), f"sample: {pick}"
        else:
            st.info("No samples yet: train.py copies some test images into samples/.")
    return None, None


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
available = {k: v for k, v in MODEL_CHOICES.items() if not v.endswith("_custom.pt") or Path(v).exists()}
missing = [k for k in MODEL_CHOICES if k not in available]
choice = st.sidebar.radio("Model", list(available))
weights = available[choice]
if missing:
    st.sidebar.caption("Not found (run train.py): " + "; ".join(missing))

# Default vocabulary: the fine-tuned models know our 4 animals; the pretrained model gets a broad list
# (80 COCO classes + the 2 custom animals that are not in COCO), so everyday photos are recognised.
if "pretrained" in choice:
    default_vocab = ", ".join(C.COCO_NAMES + [n for n in C.EXTRA_ANIMALS if n not in C.COCO_NAMES])
else:
    default_vocab = ", ".join(C.CUSTOM["names"])
vocab_text = st.sidebar.text_area(
    "Classes to detect (YOLO-World only, comma-separated)", value=default_vocab, height=150,
    key=f"vocab_{choice}",          # each model keeps its own list
    help="Open vocabulary: the model only looks for the names in this list. Type any object names. "
         "The fine-tuned model works best on its 4 trained classes.",
    disabled="YOLOv8" in choice)
st.sidebar.caption(f"{len([c for c in vocab_text.split(',') if c.strip()])} classes in the vocabulary")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, C.CONF_DEMO, 0.05)
iou = st.sidebar.slider("NMS IoU threshold", 0.3, 0.9, 0.7, 0.05)

classes = None
if "YOLOv8" not in choice:
    classes = tuple(c.strip() for c in vocab_text.split(",") if c.strip()) or tuple(C.CUSTOM["names"])

# ----------------------------------------------------------------------------- page
st.title("YOLO-World: open-vocabulary object detection")
st.caption("Cheng et al., CVPR 2024 — reproduction and deployment. "
           "Pipeline: Input → Preprocessing → Trained Model → Prediction → Output")

st.header("1. Input")
pil_img, source = read_input()
if pil_img is None:
    st.stop()
img_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

model, is_world, sha, params = get_model(weights, classes)
lb_img, pre = C.letterbox_preview(img_bgr)
dets, annotated, speed = C.run_inference(model, img_bgr, conf, iou)

c1, c2 = st.columns(2)
with c1:
    st.image(pil_img, caption=f"{source} — {pre['original_hw'][1]}×{pre['original_hw'][0]} px", width="stretch")
with c2:
    st.header("2. Preprocessing")
    st.image(cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB),
             caption=f"Letterbox: scale {pre['scale']}, padded to {pre['model_input_hw'][1]}×{pre['model_input_hw'][0]} "
                     f"(grey 114), RGB, pixels / 255", width="stretch")

st.header("3. Trained model")
m1, m2, m3, m4 = st.columns([2.6, 1.1, 1.6, 2.2])
m1.metric("Model", choice.split(" (")[0])
m2.metric("Parameters", f"{params:.1f} M")
m3.metric("Weights SHA-256", sha[:8])
m4.metric("Device", C.hardware_name().replace("Intel Core ", "").replace(" (CPU)", " CPU"))
st.caption(f"Weights file: `{weights}` · vocabulary ({len(model.names)} classes): {', '.join(model.names.values())}")

st.header("4. Prediction")
t1, t2, t3 = st.columns(3)
t1.metric("Pre-processing", f"{speed['preprocess']:.1f} ms")
t2.metric("Model inference", f"{speed['inference']:.1f} ms")
t3.metric("NMS / post-processing", f"{speed['postprocess']:.1f} ms")
if dets:
    st.dataframe(pd.DataFrame(dets), width="stretch", hide_index=True)
else:
    st.warning("No objects above the confidence threshold. Try a lower threshold or other class names.")

st.header("5. Output")
st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"{len(dets)} detections", width="stretch")
ok, png = cv2.imencode(".png", annotated)
d1, d2 = st.columns(2)
d1.download_button("Download annotated image", png.tobytes(), "prediction.png", "image/png")
d2.download_button("Download detections (JSON)",
                   json.dumps(dict(source=source, weights=weights, sha256_12=sha, conf=conf, iou=iou,
                                   preprocessing=pre, speed_ms=speed, detections=dets), indent=1),
                   "prediction.json", "application/json")
