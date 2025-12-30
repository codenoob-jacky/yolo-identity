# Training YOLO Model for 100-200 Construction Safety Violations

## Overview

This project provides a complete solution for training a YOLO model to detect 100-200 different dangerous behaviors and safety violations in construction sites. The system can identify various safety hazards including workers not wearing protective equipment, unsafe positioning, dangerous actions, and environmental hazards.

## Key Components

### 1. Training Script (`src/train_safety_model.py`)
- Supports training with 100-200 classes
- Configurable parameters for batch size, epochs, image size
- Automatic dataset validation
- Comprehensive class name generation for safety violations

### 2. Data Preparation Utilities (`utils/data_preparation.py`)
- Create proper dataset directory structure
- Split datasets into train/val/test sets
- Generate configuration files with 100-200 safety violation classes
- Validate label formats and class distributions
- Analyze dataset completeness

### 3. Sample Dataset Creator (`create_sample_dataset.py`)
- Generate dummy dataset for testing
- Create proper YOLO format labels
- Demonstrate dataset structure

### 4. Installation Script (`install.sh`)
- Install all necessary dependencies
- Handle both CPU and GPU environments
- Include packages for large-scale training

### 5. Implementation Guide (`IMPLEMENTATION_GUIDE.md`)
- Comprehensive documentation for training 100-200 classes
- Best practices for large-scale object detection
- Performance optimization strategies

## Training Process

### Step 1: Environment Setup
```bash
chmod +x install.sh
./install.sh
```

### Step 2: Dataset Preparation
```bash
python utils/data_preparation.py
# Select option 1 to create dataset structure
# Select option 3 to generate config with safety classes
```

### Step 3: Create Sample Dataset (Optional)
```bash
python create_sample_dataset.py
```

### Step 4: Training
```bash
python src/train_safety_model.py --dataset_path /path/to/dataset --num_classes 150 --epochs 200 --batch_size 8
```

## Dataset Structure

Your dataset should follow this structure:
```
dataset/
├── dataset_config.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Key Features for 100-200 Classes

1. **Comprehensive Class Names**: Generated from real safety violation categories
2. **Memory Management**: Configurable batch sizes and caching options
3. **Validation Tools**: Built-in dataset validation and error checking
4. **Scalability**: Designed to handle large numbers of classes efficiently
5. **Flexible Architecture**: Support for different YOLO model sizes (n, m, l, x)

## Safety Violation Categories

The system can detect various types of violations:

- Personal Protective Equipment (PPE) violations
- Unsafe positioning and work practices
- Equipment misuse
- Environmental hazards
- Crane and lifting operation violations
- Excavation and structural safety issues
- Chemical and electrical hazards
- Confined space violations
- And many more specific construction safety violations

## Performance Considerations

- Use more powerful models (YOLOv8x) for better accuracy with many classes
- Reduce batch size if memory is limited
- Increase training epochs for convergence with many classes
- Use mixed precision training to save memory
- Consider distributed training for very large datasets

## Next Steps

1. Prepare your real dataset with 100-200 safety violation classes
2. Use the data preparation tools to organize your data
3. Start with a smaller number of classes to validate the pipeline
4. Gradually increase the number of classes as needed
5. Fine-tune hyperparameters based on your specific requirements
6. Deploy the trained model for real-time safety monitoring

This solution provides a robust foundation for training YOLO models to detect 100-200 different safety violations in construction environments.