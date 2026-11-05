# ML Object Detection Investigation

This repository contains a benchmarking framework for comparing several object detection model families on FPV / aerial imagery.

The project evaluates both **detection accuracy** and **real-time inference performance** using a common three-class evaluation space:

- Person
- Car
- Airplane

The model families currently included are:

- YOLO26
- RF-DETR
- RT-DETRv2
- D-FINE

## Project Goals

The purpose of this project is to compare pretrained object detection models under a consistent evaluation methodology and investigate the tradeoff between detection accuracy and real-time performance.

The benchmarks focus particularly on small objects in FPV / aerial imagery, where objects may occupy only a small portion of the source frame.

Two primary measurements are collected:

### Accuracy

Accuracy is evaluated using a manually reviewed FPV image dataset and COCO-style mean Average Precision metrics.

The primary accuracy metric is mAP50-95.

Accuracy inference uses a low confidence threshold so the full precision-recall curve can be evaluated.

### Latency

Latency is evaluated using recorded 1920x1080 video played back at its native frame rate.

The latency benchmark uses a latest-frame-wins architecture. If inference cannot keep up with the source frame rate, stale frames are discarded rather than queued.

Metrics include:

- Prediction latency
- End-to-end latency
- P95 and P99 latency
- Effective inference FPS
- Frames processed
- Frames dropped
- Drop percentage
- Frame deadline misses

## Model Families

### YOLO26

YOLO26 models are evaluated using Ultralytics.

The benchmark supports:

- YOLO26n
- YOLO26s
- YOLO26m
- YOLO26l
- YOLO26x

Full-frame inference preserves source aspect ratio using rectangular inference.

### RF-DETR

RF-DETR models include:

- RF-DETR Nano
- RF-DETR Small
- RF-DETR Medium
- RF-DETR Large

Both standard full-frame inference and tiled inference are supported.

### RT-DETRv2

RT-DETRv2 models include:

- R18
- R34
- R50
- R101

Models are loaded from Hugging Face using the Transformers library.

Both standard full-frame inference and tiled inference are supported.

### D-FINE

D-FINE models include:

- Nano
- Small
- Medium
- Large
- XLarge

Models are loaded from Hugging Face using the Transformers library.

Both standard full-frame inference and tiled inference are supported.

## Tiled Inference

Transformer models are also evaluated using overlapping tiled inference to preserve small-object detail.

The standard tiled configuration is:

- Tile size: 640x640
- Overlap: 20%
- Tile batch size: 8
- Class-aware NMS IoU: 0.50

For a 1920x1080 frame, this produces eight overlapping tiles.

Tile detections are translated back into full-frame coordinates and merged using class-aware Non-Maximum Suppression.

## Repository Structure

```text
ML-Object-Detection-Investigation/
│
├── D_FINE/
│   ├── accuracy.py
│   ├── accuracy_tiled_batched.py
│   ├── latency.py
│   ├── latency_tiled_batched.py
│   ├── experiment_config.py
│   ├── dfine_viewer.py
│   ├── dfine_obs_smooth.py
│   └── run_all.py
│
├── RF_Detr/
│   ├── accuracy.py
│   ├── accuracy_tiled_batched.py
│   ├── latency.py
│   ├── latency_tiled_batched.py
│   ├── experiment_config.py
│   ├── rfdetr_viewer.py
│   ├── rfdetr_obs_smooth.py
│   └── run_all.py
│
├── RT_DETRv2/
│   ├── accuracy.py
│   ├── accuracy_tiled_batched.py
│   ├── latency.py
│   ├── latency_tiled_batched.py
│   ├── experiment_config.py
│   ├── rtdetrv2_viewer.py
│   ├── rtdetrv2_obs_smooth.py
│   └── run_all.py
│
└── Yolo/
    ├── accuracy.py
    ├── latency.py
    ├── experiment_config.py
    ├── yolo_viewer.py
    ├── yolo_obs_smooth.py
    └── run_all.py
```

## Local Data

The evaluation dataset and benchmark video are not included in this repository.

Each benchmark folder expects the following local structure when applicable:

```text
accuracy_test/
├── images/
└── labels/

videos/
└── 30fps.mp4
```

The label files use YOLO-format normalized bounding boxes.

The evaluation uses the following original COCO class IDs:

```text
0 = person
2 = car
4 = airplane
```

Annotations belonging to classes outside the three-class evaluation set are ignored.

## Installation

A single Python environment can be used for all model families.

See requirements.txt for information about required libraries. 

## Viewers

Each model family includes a standalone viewer tool for visually inspecting detections on the benchmark video.

## Live / OBS Demonstrations

Each model family also includes an OBS / live-capture demonstration.

These scripts use the OBS virual webcam tool to monitor a live video feed. This can be true live footage or a video passed through as a source.

The capture device index in the script may need to be changed depending on your system and the OBS Virtual Camera configuration.
