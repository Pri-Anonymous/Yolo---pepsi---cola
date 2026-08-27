# Pepsi vs Coca-Cola Detector (YOLO11 Instance Segmentation)

A custom computer vision model that detects and segments Pepsi and Coca-Cola cans in real time using a fine-tuned YOLO11 segmentation model, trained on a self-collected and self-annotated dataset.

## Demo

Real-time detection running on a live webcam feed, drawing segmentation masks and confidence scores around detected cans.

*(Add your demo GIF/video here — see "Adding Your Demo" section below)*

## Overview

This project covers the full computer vision pipeline end-to-end:

- **Data collection** — gathered images of Pepsi and Coca-Cola cans
- **Annotation** — labeled using Roboflow (instance segmentation polygons)
- **Dataset** — 512 images total (360 train / 101 valid / 51 test), 2 classes
- **Training** — fine-tuned YOLO11n-seg (Ultralytics) on the custom dataset
- **Evaluation** — tracked precision, recall, mAP50, and mAP50-95 across training
- **Deployment** — real-time inference on live webcam feed using OpenCV

## Results

| Metric | Score |
|---|---|
| Box mAP50 | ~0.85+ |
| Mask mAP50 | ~0.85+ |
| Classes | `pepsi`, `cocacola` |
| Model | YOLO11n-seg |
| Training epochs | 60 |
| Image size | 480x480 |

## Tech Stack

- **Ultralytics YOLO11** — model architecture & training
- **Roboflow** — dataset annotation and management
- **PyTorch** (CUDA) — training backend
- **OpenCV** — real-time webcam inference and visualization
- **Python 3.12**, Conda environment

## Project Structure

```
yolo-pepsi-coke/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml              # dataset config for YOLO
├── train.py                # training script
├── predictions.py          # real-time webcam inference script
├── runs/                   # training outputs, weights, metrics (generated after training)
└── README.md
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/yolo-pepsi-coke.git
   cd yolo-pepsi-coke
   ```

2. **Create and activate an environment**
   ```bash
   conda create -n python_yolo_sols python=3.12
   conda activate python_yolo_sols
   ```

3. **Install dependencies**
   ```bash
   pip install ultralytics opencv-python
   ```

## Training

```bash
python train.py
```

Trains a YOLO11n-seg model on the dataset defined in `data.yaml`. Trained weights are saved to `runs/segment/<run-name>/weights/best.pt`.

## Running Inference

**On an image or folder:**
```python
from ultralytics import YOLO

model = YOLO("runs/segment/<run-name>/weights/best.pt")
model.predict(source="path/to/image_or_folder", show=True, save=True, conf=0.5)
```

**On a live webcam feed:**
```bash
python predictions.py
```
Opens your webcam and draws real-time bounding boxes/masks around detected Pepsi and Coca-Cola cans. Press `q` to quit.

## Dataset

The dataset was collected and annotated manually via [Roboflow](https://roboflow.com), using instance segmentation polygons rather than plain bounding boxes for more precise object boundaries.

> **Note:** Due to size, the dataset itself is not included directly in this repo. See `data.yaml` for the expected folder structure, or link your own Roboflow export.

## Limitations & Future Work

- Trained on a relatively small dataset (512 images) — generalizes well to similar conditions but can produce false positives on visually similar objects (e.g., certain patterned/colored items) that weren't represented in training
- Planned improvements:
  - Add "negative" background images with no annotations to reduce false positives
  - Expand dataset with more angles, lighting conditions, and backgrounds
  - Experiment with YOLO11s-seg / YOLO11m-seg for improved accuracy

## Adding Your Demo

To embed a video/gif in this README on GitHub:
```markdown
![demo](demo.gif)
```
Place `demo.gif` in your repo root (or a `docs/` or `assets/` folder) and adjust the path accordingly. GitHub renders gifs inline automatically; `.mp4`/`.avi` files won't autoplay in the README, so convert your `.avi` demo to a `.gif` first (e.g. using [ezgif.com](https://ezgif.com) or `ffmpeg`).

## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset annotation tooling
