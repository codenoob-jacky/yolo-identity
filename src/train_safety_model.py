"""
Training script for custom safety detection model
This script trains a YOLO model to detect safety equipment like hard hats, safety vests, etc.
"""

import torch
import yaml
from ultralytics import YOLO
import os
from pathlib import Path


def create_dataset_config(dataset_path: str, num_classes: int, class_names: list):
    """
    Create a YAML configuration file for the dataset
    
    Args:
        dataset_path: Path to the dataset directory
        num_classes: Number of classes in the dataset
        class_names: List of class names
    """
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',  # Optional
        'nc': num_classes,
        'names': class_names
    }
    
    config_path = os.path.join(dataset_path, 'dataset.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Dataset configuration saved to {config_path}")
    return config_path


def train_safety_model(
    dataset_config: str,
    model_architecture: str = 'yolov8n.pt',  # Model architecture to use
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    save_dir: str = './runs/train_safety'
):
    """
    Train a YOLO model for safety equipment detection
    
    Args:
        dataset_config: Path to dataset configuration YAML file
        model_architecture: Pretrained model to start from
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size for training
        save_dir: Directory to save training results
    """
    # Load a model
    model = YOLO(model_architecture)
    
    # Train the model
    results = model.train(
        data=dataset_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        save_dir=save_dir,
        name='safety_detection',
        exist_ok=True
    )
    
    print("Training completed!")
    return model


def validate_model(model_path: str, dataset_config: str):
    """
    Validate the trained model
    
    Args:
        model_path: Path to the trained model
        dataset_config: Path to dataset configuration YAML file
    """
    # Load model
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(data=dataset_config)
    print(f"Validation results: {metrics}")
    
    return metrics


def main():
    print("Safety Detection Model Training")
    print("=" * 40)
    
    # Define safety classes
    safety_classes = [
        'person',           # 0
        'hard_hat',         # 1
        'safety_vest',      # 2
        'safety_harness',   # 3
        'safety_goggles',   # 4
        'gloves',           # 5
        'boots'             # 6
    ]
    
    # Create dataset configuration
    # Note: You'll need to prepare your dataset in the expected format
    dataset_path = input("Enter path to your dataset directory: ").strip()
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        print("Please prepare your dataset with the following structure:")
        print("""
        dataset/
        ├── dataset.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
        """)
        return
    
    config_path = create_dataset_config(dataset_path, len(safety_classes), safety_classes)
    
    # Get training parameters
    epochs = int(input("Enter number of epochs (default 100): ") or "100")
    imgsz = int(input("Enter image size (default 640): ") or "640")
    batch_size = int(input("Enter batch size (default 16): ") or "16")
    
    # Train the model
    print("\nStarting training...")
    model = train_safety_model(
        dataset_config=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size
    )
    
    print("Training completed successfully!")
    print("Model saved and ready for inference.")


if __name__ == "__main__":
    main()