# YOLO-World: Reproduction, Experiments and Deployment

**Paper:** T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," CVPR 2024 (arXiv:2401.17270).
**Group:** S. Mujtaba Kazmi (36397) · Syed Mizfar Hussain (36251) · Ayaan Hussain (36223)

> **Links** (also on page 1 of the report and in the LMS comment):
> - Code repository: https://github.com/syedmizfarhussain/reimagined-barnacle
> - Trained weights: https://github.com/syedmizfarhussain/reimagined-barnacle/tree/main/weights (`yolo_world_s_custom.pt`, `yolov8s_custom.pt`)

## 1. What is in this folder

| File | Purpose |
|---|---|
| `YOLO World Experiments.ipynb` | The complete record of the project, run in Colab: data pipeline and EDA (Sections 1–7, from Deliverable 2), custom dataset EDA (3b), zero-shot COCO/LVIS reproduction (9), comparison with the paper (11b), prompt study (10), S/M/L speed study (11), fine-tuning vs. YOLOv8 (12), results summary (13). **All numbers in the report come from this notebook.** |
| `train.py` | **Training:** fine-tunes YOLO-World-S and trains the closed-set YOLOv8-S baseline with the same settings, then copies the best checkpoints to `weights/` |
| `evaluate.py` | **Evaluation:** `custom` (test split), `coco` (zero-shot, official pycocotools, compared with reported numbers), `speed` (S/M/L latency) |
| `infer.py` | **Inference** from the command line (image or folder → annotated image + JSON) |
| `app.py` | **Deployment:** Streamlit web app for live inference |
| `yw_common.py` | Shared code used by the scripts and the app |
| `requirements.txt` | Dependencies |
| `weights/` | Trained checkpoints (created by `train.py` or by notebook Section 12) |
| `samples/` | Held-out **test** images for the demo (never used in training) |
| `results/` | CSV/JSON result tables written by the scripts and the notebook |

## 2. Setup

Python 3.10 or newer. A GPU is recommended for training; inference and the app also run on CPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps lvis==0.5.3     # only for the LVIS evaluation in the notebook
```

The model weights `yolov8{s,m,l}-worldv2.pt` and `yolov8s.pt` download automatically the first time (from Ultralytics). The datasets download automatically too:
- custom data: *African Wildlife* (Ultralytics, ~1.5k images, classes buffalo, elephant, rhino, zebra; train/val/test),
- COCO val2017 annotations (241 MB) + a seeded random subset of 500 images,
- LVIS v1 val annotations + 500 images (notebook only).

## 3. Running the experiments

**Full record (recommended): Google Colab.** Open `YOLO World Experiments.ipynb`, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. Set `CFG["QUICK_TEST"] = True` first to check every cell in a few minutes. Results go to `data/results/` (CSV + `summary.md` + `run_info.json`), and the trained weights are copied to `weights/`. In Colab, download `weights/`, `samples/` and `data/results/` from the *Files* panel (or copy them to Google Drive) and put them into this folder. The app and `evaluate.py custom` then use exactly the weights behind the reported numbers.

**Scripts (same settings as the notebook):**

```bash
python train.py                         # fine-tune YOLO-World-S + train YOLOv8-S (30 epochs each)
python evaluate.py custom               # test-split comparison: zero-shot vs fine-tuned vs YOLOv8
python evaluate.py coco --sizes s m l   # zero-shot COCO (500-image subset) vs. reported numbers
python evaluate.py speed --sizes s m l  # latency / FPS / parameters
python infer.py samples/ --out outputs  # predictions for the demo images
```

Fixed settings: seed 42, image size 640, score threshold 0.001 and at most 300 boxes for mAP, 0.25 for the demo and the speed test, batch size 1 for timing.

## 4. Launching the app (deployment)

**Windows, easiest:** double-click **`Start_App.bat`**. The first time, it installs everything into a `.venv` folder (about 10 minutes, needs internet). After that, it starts the app and opens your browser. Keep the black window open while you use the app.

**Any system:**
```bash
streamlit run app.py
```

Open the printed URL (default http://localhost:8501). The page shows the whole pipeline **Input → Preprocessing → Trained Model → Prediction → Output**:

1. **Input:** upload an image, take a webcam photo, paste an image URL, or pick a held-out test sample.
2. **Preprocessing:** the letterboxed image the network receives (scale factor, padded size, grey 114 padding).
3. **Trained model:** which checkpoint is used, its parameter count and the SHA-256 fingerprint of the weights file. The fingerprint shows the prediction comes from our trained model.
4. **Prediction:** detections table (class, confidence, box in original pixels) and timing for pre-processing, inference and NMS.
5. **Output:** annotated image, downloadable as PNG, and the detections as JSON.

Models in the sidebar:
- **YOLO-World-S fine-tuned** (our trained model, `weights/yolo_world_s_custom.pt`). It is still open-vocabulary, so the class list can be edited.
- **YOLOv8-S fine-tuned** (closed-set baseline, `weights/yolov8s_custom.pt`). Its 4 classes are fixed.
- **YOLO-World-S pretrained** (zero-shot). Type any object names, e.g. `giraffe, lion, safari jeep`.

**Input/output format.** Input is any RGB image (JPG/PNG/BMP/WEBP, any size). Output is a list of `{class_id, class_name, confidence, box_xyxy}` with boxes in original-image pixels (`x1, y1, x2, y2`), plus the annotated image.

**Before the demo:** start the app during the previous group's viva. Load each model once so the weights are cached, then check one sample image. Weights and samples are local files, but the *Image URL* tab needs internet.

## 5. Code attribution

| Part | Status | Details |
|---|---|---|
| Data loading, record format, COCO/LVIS subset sampling, YOLO-format reader/exporter (notebook §2–3, `yw_common.py`) | **Written by our team** | |
| EDA and data checks (notebook §2c, §3b) | **Written by our team** | |
| Letterbox, mosaic, random affine, HSV, flip (notebook §4) | **Adapted** | Re-implemented in NumPy/OpenCV, following YOLOv5/YOLOv8 (Ultralytics). Parameter values are from the official YOLO-World config `yolo_world_v2_s_..._finetune_coco.py` |
| `random_load_text` (notebook §4b) | **Adapted** | Same behaviour as `RandomLoadText` in the official `AILab-CVC/YOLO-World` repository, written without MMCV |
| CLIP text encoding and caching (notebook §6) | **Written by our team** using the Hugging Face `transformers` CLIP model `openai/clip-vit-base-patch32` (reused) |
| Evaluation wrappers, prompt study, S/M/L timing, comparison tables, results summary (notebook §8–13, `evaluate.py`) | **Written by our team** | |
| Official COCO and LVIS metrics | **Reused** | `pycocotools`, `lvis` packages |
| YOLO-World and YOLOv8 networks, training loop, NMS, validator | **Reused** | `ultralytics` library (`YOLOWorld`, `YOLO`, `model.train`, `model.val`) |
| Pretrained weights `yolov8{s,m,l}-worldv2.pt` | **Reused** | Authors' released weights via Ultralytics. We did not re-run the pre-training (Objects365/GoldG/CC3M is far beyond our compute) |
| Streamlit app, CLI scripts (`app.py`, `train.py`, `evaluate.py`, `infer.py`) | **Written by our team** | |

**Results attribution.** Numbers marked *reported*/*paper* in the notebook and in `results/` are copied from the original sources: paper Table 2 (zero-shot LVIS minival, V100), and the Ultralytics docs (zero-shot COCO for the v2 weights). All other numbers are measured by us and printed by the code.

## 6. Notes and limitations

- COCO and LVIS are evaluated on 500-image seeded subsets (compute limit), not the full splits, so the gap to the reported numbers includes subset noise.
- The paper reports LVIS *minival* with *Fixed AP* and v1 weights; we use the LVIS v1 *val* subset, standard AP and v2 weights. Section 11b of the notebook explains each difference.
- Speed depends on the GPU: the paper uses a V100, Colab gives a T4. `run_info.json` records our hardware.
- Each model is trained once (one seed).
