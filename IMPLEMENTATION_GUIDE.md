# Implementation Guide: Construction Site Safety Detection with YOLO

This guide provides a comprehensive overview of how to implement a YOLO-based system for detecting safety violations on construction sites, such as workers not wearing hard hats or safety harnesses.

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Training](#model-training)
5. [Inference and Deployment](#inference-and-deployment)
6. [Safety Violation Detection Logic](#safety-violation-detection-logic)
7. [Evaluation and Testing](#evaluation-and-testing)

## Overview

The construction site safety detection system uses YOLO (You Only Look Once) object detection to identify workers and their safety equipment in real-time. The system can detect violations such as:
- Workers not wearing hard hats
- Workers not wearing safety vests
- Workers not wearing safety harnesses (especially important for high-altitude work)

## System Architecture

The system consists of several components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Data    │───▶│  YOLO Detector   │───▶│  Violation      │
│ (Images/Videos) │    │  (Person/Safety   │    │  Analyzer       │
└─────────────────┘    │   Equipment)      │    └─────────────────┘
                       └──────────────────┘
                                │
                       ┌──────────────────┐
                       │  Output Results  │
                       │ (Visualizations, │
                       │   Alerts, etc.)  │
                       └──────────────────┘
```

## Dataset Preparation

### Required Dataset Structure
```
safety_dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Classes Definition
The model should detect the following classes:
- `0`: person
- `1`: hard_hat
- `2`: safety_vest
- `3`: safety_harness
- `4`: safety_goggles
- `5`: gloves
- `6`: boots

### Annotation Format
Labels should be in YOLO format (.txt files):
```
class_id center_x center_y width height
```
All values are normalized between 0 and 1.

### Data Collection Tips
- Collect images from various construction sites
- Include different lighting conditions (daylight, artificial light, shadows)
- Capture different angles and distances
- Ensure diverse worker appearances and clothing
- Include both compliant and non-compliant scenarios

## Model Training

### Using the Training Script
```bash
python src/train_safety_model.py
```

### Training Parameters
- **Model Architecture**: Start with `yolov8n.pt` for faster training, or `yolov8m.pt` for better accuracy
- **Image Size**: 640x640 pixels (default)
- **Batch Size**: 16 (adjust based on GPU memory)
- **Epochs**: 100-300 (monitor validation metrics to avoid overfitting)

### Training Best Practices
1. Use data augmentation to improve model generalization
2. Monitor validation loss to prevent overfitting
3. Use transfer learning from a pre-trained model
4. Fine-tune hyperparameters based on validation results

## Inference and Deployment

### Real-time Detection
```bash
python -m src.detection --input webcam
```

### Image Detection
```bash
python -m src.detection --input path/to/image.jpg --output output.jpg
```

### Video Detection
```bash
python -m src.detection --input path/to/video.mp4 --output output.mp4
```

### Custom Model Usage
```bash
python -m src.detection --input webcam --model path/to/trained_model.pt
```

## Safety Violation Detection Logic

The system identifies safety violations by analyzing the spatial relationships between detected objects:

### Hard Hat Violation Detection
1. Detect all `person` objects
2. Detect all `hard_hat` objects
3. For each person, check if there's a hard hat in the head region
4. If no hard hat is detected near a person's head, flag as violation

### Safety Harness Violation Detection
1. Detect `person` objects in potentially dangerous areas (height, near edges)
2. Check for `safety_harness` objects associated with these persons
3. Flag violations if required equipment is missing

### Algorithm Details
```python
# Pseudo-code for violation detection
for person in detected_persons:
    head_region = calculate_head_region(person_bbox)
    has_hard_hat = False
    
    for hard_hat in detected_hard_hats:
        if bbox_in_head_region(hard_hat, head_region):
            has_hard_hat = True
            break
    
    if not has_hard_hat:
        flag_violation("missing_hard_hat", person)
```

## Evaluation and Testing

### Metrics
- **Precision**: Percentage of correctly identified violations
- **Recall**: Percentage of actual violations that were detected
- **mAP (mean Average Precision)**: Overall detection accuracy

### Testing Scenarios
1. **Daylight conditions**: Standard outdoor lighting
2. **Low light conditions**: Dawn, dusk, or artificial lighting
3. **Different weather**: Sunny, cloudy, rainy
4. **Various construction sites**: Building sites, road work, industrial sites
5. **Different worker postures**: Standing, crouching, climbing

### Performance Optimization
- Use appropriate model size for your hardware
- Consider quantization for edge deployment
- Optimize preprocessing pipeline
- Use GPU acceleration where available

## Implementation Considerations

### Hardware Requirements
- **Training**: GPU with >=8GB VRAM recommended
- **Inference**: Can run on CPU, but GPU provides better real-time performance

### Privacy and Ethics
- Ensure compliance with local privacy regulations
- Anonymize data where possible
- Obtain necessary permissions for site recordings

### Accuracy Improvements
- Collect more diverse training data
- Use domain-specific data augmentation
- Implement ensemble methods
- Regular model updates with new data

## Deployment Options

### Edge Deployment
- Deploy on local hardware at construction sites
- Use NVIDIA Jetson or similar edge AI platforms
- Implement local alert systems

### Cloud-Based Processing
- Stream video to cloud for processing
- Store and analyze violation data
- Generate reports and analytics

### Hybrid Approach
- Local processing for real-time alerts
- Cloud processing for analytics and model updates

This implementation provides a robust foundation for detecting safety violations on construction sites using YOLO-based object detection technology.