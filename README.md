# Construction Safety Detection with YOLO - 100-200 Classes

This project implements a YOLO-based system for detecting 100-200 different dangerous behaviors and safety violations in construction sites. The system can identify various safety hazards including workers not wearing protective equipment, unsafe positioning, dangerous actions, and environmental hazards.

## Features

- Detection of 100-200 different dangerous behaviors and safety violations
- Support for custom YOLOv8/v9 models
- Comprehensive data preparation tools
- Training pipeline with validation
- Real-time detection capabilities
- Post-processing for safety analysis

## Requirements

- Python 3.7+
- PyTorch
- YOLOv8/v9 (Ultralytics)
- OpenCV
- NumPy
- Pandas

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Prepare your dataset with 100-200 classes
2. Run the training script:
   ```bash
   python src/train_safety_model.py
   ```

### Inference

```bash
python src/detection.py --input path/to/input --output path/to/output
```

## Dataset Structure

Your dataset should follow the YOLO format:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```