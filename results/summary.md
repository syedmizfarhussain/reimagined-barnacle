# YOLO-World experiments: results

Run info: date=2026-11-25 15:39, python=3.13.16, torch=2.14.1+cu130, ultralytics=8.4.218, hardware=Tesla T4, img_size=640, conf_eval=0.001, conf_speed=0.25, max_det=300, coco_images=500, lvis_images=500, custom_dataset=african-wildlife, epochs=30, quick_test=False, train_recipes={'yolo_world_s': {'optimizer': 'AdamW', 'lr0': 0.0002, 'weight_decay': 0.05, 'warmup_bias_lr': 0.0}, 'yolov8s': {}}

## Key findings

- Zero-shot COCO (500 images), YOLO-World-S: AP 36.9, AP50 50.3, AP small/medium/large 20.7 / 39.9 / 56.0.
- Reproduction gap, zero-shot COCO AP (ours on the subset minus reported): YOLO-World-S -0.8; YOLO-World-M -1.4; YOLO-World-L -1.5.
- Reproduction gap, zero-shot LVIS AP of the S model (ours minus paper Table 2): -6.2.
- Zero-shot LVIS (500 images): AP 20.0, APr 10.0, APc 15.0, APf 21.6.
- Prompt wording on COCO: AP ranges from 33.6 to 36.9 (spread 3.3 points); best variant: 'plain'.
- Prompt wording on the custom test split: mAP50-95 from 53.5 to 56.2; best variant: 'plain'.
- S/M/L on Tesla T4: YOLO-World-S AP 36.9 at 12.6 ms (79 FPS); YOLO-World-M AP 41.6 at 20.6 ms (49 FPS); YOLO-World-L AP 44.3 at 36.6 ms (27 FPS).
- Custom test split mAP50-95: YOLO-World-S zero-shot 56.2; YOLO-World-S fine-tuned 84.8; YOLOv8-S fine-tuned (closed set) 83.8.
- COCO AP of YOLO-World-S before / after fine tuning: 36.9 / 11.3.

## coco_zero_shot

| model | prompts | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---|---|---|---|---|---|---|
| YOLO-World-S v2 | plain | 36.87 | 50.26 | 39.26 | 20.67 | 39.92 | 55.97 | 55.63 |

## coco_per_class

| class | AP | boxes_in_subset | boxes_in_val2017 |
|---|---|---|---|
| person | 57.75 | 1148 | 10766 |
| bicycle | 25.30 | 34 | 314 |
| car | 42.10 | 217 | 1918 |
| motorcycle | 35.22 | 26 | 367 |
| airplane | 78.48 | 26 | 143 |
| bus | 72.39 | 31 | 283 |
| train | 70.24 | 25 | 190 |
| truck | 33.80 | 47 | 414 |
| boat | 12.95 | 42 | 424 |
| traffic light | 17.99 | 78 | 630 |
| fire hydrant | 68.38 | 5 | 101 |
| stop sign | 32.88 | 8 | 75 |
| parking meter | 17.14 | 19 | 60 |
| bench | 24.54 | 38 | 411 |
| bird | 20.38 | 45 | 420 |
| cat | 66.17 | 25 | 202 |
| dog | 44.40 | 14 | 218 |
| horse | 43.57 | 43 | 272 |
| sheep | 54.32 | 59 | 354 |
| cow | 60.25 | 12 | 372 |
| elephant | 65.87 | 19 | 252 |
| bear | 28.84 | 9 | 71 |
| zebra | 76.03 | 12 | 266 |
| giraffe | 81.82 | 22 | 232 |
| backpack | 13.64 | 45 | 371 |
| umbrella | 41.17 | 26 | 405 |
| handbag | 12.37 | 58 | 540 |
| tie | 42.08 | 20 | 251 |
| suitcase | 31.92 | 35 | 299 |
| frisbee | 53.98 | 8 | 115 |
| skis | 15.44 | 24 | 233 |
| snowboard | 23.20 | 10 | 66 |
| sports ball | 23.34 | 29 | 259 |
| kite | 22.89 | 19 | 327 |
| baseball bat | 14.40 | 7 | 145 |
| baseball glove | 47.49 | 15 | 148 |
| skateboard | 65.56 | 23 | 179 |
| surfboard | 59.80 | 24 | 265 |
| tennis racket | 51.24 | 15 | 225 |
| bottle | 35.38 | 73 | 1013 |
| wine glass | 12.48 | 32 | 341 |
| cup | 48.79 | 99 | 895 |
| fork | 41.00 | 16 | 215 |
| knife | 16.53 | 22 | 325 |
| spoon | 11.92 | 26 | 253 |
| bowl | 42.73 | 77 | 622 |
| banana | 11.66 | 54 | 370 |
| apple | 11.93 | 13 | 236 |
| sandwich | 34.11 | 9 | 177 |
| orange | 45.31 | 36 | 285 |
| broccoli | 16.60 | 30 | 312 |
| carrot | 23.73 | 38 | 365 |
| hot dog | 59.93 | 5 | 125 |
| pizza | 27.89 | 34 | 284 |
| donut | 49.10 | 11 | 328 |
| cake | 28.68 | 19 | 310 |
| chair | 28.83 | 126 | 1771 |
| couch | 43.49 | 22 | 261 |
| potted plant | 25.03 | 42 | 342 |
| bed | 37.43 | 15 | 163 |
| dining table | 24.57 | 70 | 694 |
| toilet | 40.46 | 23 | 179 |
| tv | 52.94 | 23 | 288 |
| laptop | 75.79 | 17 | 231 |
| mouse | 65.69 | 11 | 106 |
| remote | 6.11 | 32 | 282 |
| keyboard | 55.97 | 13 | 153 |
| cell phone | 24.46 | 11 | 261 |
| microwave | 41.05 | 3 | 55 |
| oven | 9.45 | 14 | 143 |
| toaster | 0.00 | 1 | 9 |
| sink | 50.52 | 17 | 225 |
| refrigerator | 43.90 | 18 | 126 |
| book | 11.14 | 156 | 1128 |
| clock | 47.39 | 20 | 267 |
| vase | 38.11 | 27 | 274 |
| scissors | 9.65 | 8 | 36 |
| teddy bear | 40.11 | 26 | 190 |
| hair drier | 0.00 | 1 | 11 |
| toothbrush | 10.56 | 5 | 57 |

## lvis_zero_shot

| model | prompts | AP | AP50 | AP75 | APs | APm | APl | APr | APc | APf | AR@300 | ARs@300 | ARm@300 | ARl@300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World-S v2 | plain | 20.03 | 29.39 | 20.85 | 8.77 | 26.24 | 43.11 | 10.00 | 15.01 | 21.62 | 23.64 | 9.75 | 31.59 | 47.72 |

## prompt_sensitivity_coco

| variant | example | text_cosine_to_plain | AP | AP50 | AP75 | APs | APm | APl | AR100 | dAP_vs_plain |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | dining table | 1.00 | 36.87 | 50.26 | 39.26 | 20.67 | 39.92 | 55.98 | 55.63 | 0.00 |
| photo | a photo of a dining table | 0.94 | 33.58 | 46.10 | 35.35 | 16.98 | 36.41 | 55.26 | 53.86 | -3.29 |
| object | a dining table object | 0.94 | 34.01 | 46.15 | 36.20 | 16.83 | 36.40 | 56.56 | 53.76 | -2.86 |
| synonyms | table | 0.99 | 36.56 | 49.86 | 38.72 | 20.38 | 39.59 | 56.09 | 55.63 | -0.32 |

## prompt_synonyms_per_class

| class | new prompt | boxes_in_subset | AP plain | AP synonym | change |
|---|---|---|---|---|---|
| hair drier | hair dryer | 1 | 0.00 | 0.00 | 0.00 |
| dining table | table | 70 | 24.57 | 20.34 | -4.23 |
| potted plant | house plant | 42 | 25.03 | 22.75 | -2.29 |
| tv | television | 23 | 52.94 | 44.46 | -8.47 |
| cell phone | mobile phone | 11 | 24.46 | 20.69 | -3.77 |
| sports ball | ball | 29 | 23.34 | 21.36 | -1.98 |
| mouse | computer mouse | 11 | 65.69 | 62.75 | -2.95 |
| remote | remote control | 32 | 6.11 | 5.41 | -0.70 |

## prompt_sensitivity_custom

| variant | prompts | mAP50_95 | mAP50 | mAP75 | precision | recall | AP buffalo | AP elephant | AP rhino | AP zebra |
|---|---|---|---|---|---|---|---|---|---|---|
| plain | buffalo | elephant | rhino | zebra | 56.25 | 65.27 | 62.37 | 76.42 | 62.15 | 50.74 | 58.54 | 31.93 | 83.80 |
| photo | a photo of a buffalo | a photo of a elephant | a photo of a rhino | a photo of a zebra | 54.83 | 64.77 | 61.28 | 45.63 | 79.89 | 40.88 | 60.05 | 42.48 | 75.91 |
| specific | african buffalo | african elephant | rhinoceros | zebra | 53.52 | 62.56 | 59.28 | 48.78 | 71.91 | 49.91 | 46.05 | 34.87 | 83.26 |

## model_size_speed

| model | size | params_M | AP | AP50 | AP75 | APs | pre_ms | infer_ms | post_ms | total_ms | fps_model | fps_end_to_end | peak_gpu_mem_MB | infer_ms_fp16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World-S v2 | s | 12.76 | 36.87 | 50.26 | 39.26 | 20.67 | 0.92 | 12.58 | 1.41 | 15.34 | 79.48 | 65.18 | 447.39 | 16.11 |
| YOLO-World-M v2 | m | 28.38 | 41.60 | 55.14 | 44.51 | 24.40 | 0.97 | 20.57 | 1.58 | 23.62 | 48.61 | 42.33 | 595.21 | 16.30 |
| YOLO-World-L v2 | l | 46.83 | 44.31 | 58.99 | 47.01 | 26.88 | 0.90 | 36.60 | 1.40 | 39.37 | 27.32 | 25.40 | 648.23 | 20.51 |

## reproduction_coco

| model | our_AP | our_AP50 | our_AP75 | rep_AP | rep_AP50 | rep_AP75 | AP_gap |
|---|---|---|---|---|---|---|---|
| YOLO-World-S v2 | 36.87 | 50.26 | 39.26 | 37.70 | 52.20 | 41.00 | -0.83 |
| YOLO-World-M v2 | 41.60 | 55.14 | 44.51 | 43.00 | 58.40 | 46.80 | -1.40 |
| YOLO-World-L v2 | 44.31 | 58.99 | 47.01 | 45.80 | 61.30 | 49.80 | -1.49 |

## reproduction_speed

| model | params_M | infer_ms | fps_model | paper_params_M | paper_fps_v100 | hardware |
|---|---|---|---|---|---|---|
| YOLO-World-S v2 | 12.76 | 12.58 | 79.48 | 13 | 74.10 | Tesla T4 |
| YOLO-World-M v2 | 28.38 | 20.57 | 48.61 | 29 | 58.10 | Tesla T4 |
| YOLO-World-L v2 | 46.83 | 36.60 | 27.32 | 48 | 52.00 | Tesla T4 |

## reproduction_lvis

| metric | ours_v2_val_subset | paper_v1_minival | gap |
|---|---|---|---|
| AP | 20.03 | 26.20 | -6.17 |
| APr | 10.00 | 19.10 | -9.10 |
| APc | 15.01 | 23.60 | -8.59 |
| APf | 21.62 | 29.80 | -8.18 |

## finetune_comparison

| model | train_min | params_M | mAP50_95 | mAP50 | mAP75 | precision | recall | infer_ms | AP buffalo | AP elephant | AP rhino | AP zebra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO-World-S zero-shot | 0.00 | 12.76 | 56.25 | 65.27 | 62.37 | 76.42 | 62.15 | 11.52 | 50.74 | 58.54 | 31.93 | 83.80 |
| YOLO-World-S fine-tuned | 14.21 | 12.76 | 84.82 | 97.14 | 93.59 | 93.72 | 93.90 | 13.57 | 84.40 | 82.10 | 90.63 | 82.14 |
| YOLOv8-S fine-tuned (closed set) | 18.53 | 11.14 | 83.75 | 98.40 | 91.62 | 95.32 | 94.61 | 8.37 | 86.73 | 79.06 | 92.65 | 76.57 |

## finetune_per_class

| class | YOLO-World-S zero-shot | YOLO-World-S fine-tuned | YOLOv8-S fine-tuned (closed set) |
|---|---|---|---|
| buffalo (outside COCO) | 50.74 | 84.40 | 86.73 |
| elephant (in COCO) | 58.54 | 82.10 | 79.06 |
| rhino (outside COCO) | 31.93 | 90.63 | 92.65 |
| zebra (in COCO) | 83.80 | 82.14 | 76.57 |

## forgetting_coco

| model | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---|---|---|---|---|---|
| YOLO-World-S before fine tuning | 36.87 | 50.26 | 39.26 | 20.67 | 39.92 | 55.97 | 55.63 |
| YOLO-World-S after fine tuning | 11.26 | 17.40 | 11.64 | 6.68 | 13.69 | 20.78 | 30.22 |