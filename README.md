# Construction Site Safety Detection with YOLO

This project implements a YOLO-based system for detecting safety violations on construction sites, such as workers not wearing hard hats or safety harnesses.

## Features
- Real-time detection of safety equipment
- Classification of safety violations
- Video and image processing capabilities
- Training pipeline for custom safety detection models

## Requirements
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO

## Installation
```bash
pip install torch torchvision opencv-python ultralytics
```

## Usage
1. Prepare your dataset with safety equipment annotations
2. Train the model using the training script
3. Run inference on images or videos