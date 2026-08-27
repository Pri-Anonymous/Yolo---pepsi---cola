<img width="480" height="360" alt="demo" src="https://github.com/user-attachments/assets/2e8940b7-d18b-45ed-afef-9b61db0a60d6" />


# Pepsi vs Coca-Cola Detector (YOLO11 Instance Segmentation)

A custom computer vision model that detects and segments Pepsi and Coca-Cola cans in real time using a fine-tuned YOLO11 segmentation model, trained on a self-collected and self-annotated dataset.

## Demo

Real-time detection running on a live webcam feed, drawing segmentation masks and confidence scores around detected cans.

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

The dataset was collected and annotated manually via [Roboflow](https://roboflow.com), using instance segmentation polygons rather than plain bounding boxes for more precise object boundaries.

## Limitations & Future Work

- Trained on a relatively small dataset (512 images) — generalizes well to similar conditions but can produce false positives on visually similar objects (e.g., certain patterned/colored items) that weren't represented in training
- Planned improvements:
  - Add "negative" background images with no annotations to reduce false positives
  - Expand dataset with more angles, lighting conditions, and backgrounds
  - Experiment with YOLO11s-seg / YOLO11m-seg for improved accuracy

```markdown
[demo]<img width="480" height="360" alt="demo" src="https://github.com/user-attachments/assets/41ad2e50-2d88-4e69-b70f-fa0ac0ae9ec3" />


```
## Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset annotation tooling
