#!/usr/bin/env python3
"""
Sample script to create a dummy dataset for training a YOLO model 
to detect 100-200 construction safety violations
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml
from utils.data_preparation import generate_comprehensive_class_names


def create_sample_images_and_labels(dataset_path: str, num_classes: int = 100, num_samples: int = 50):
    """
    Create sample images and labels for demonstration purposes
    
    Args:
        dataset_path: Path to create the dataset
        num_classes: Number of classes to create
        num_samples: Number of sample images to create
    """
    print(f"Creating sample dataset with {num_classes} classes and {num_samples} images...")
    
    # Create directory structure
    dirs = {
        'images_train': os.path.join(dataset_path, 'images', 'train'),
        'images_val': os.path.join(dataset_path, 'images', 'val'),
        'images_test': os.path.join(dataset_path, 'images', 'test'),
        'labels_train': os.path.join(dataset_path, 'labels', 'train'),
        'labels_val': os.path.join(dataset_path, 'labels', 'val'),
        'labels_test': os.path.join(dataset_path, 'labels', 'test')
    }
    
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    # Generate class names
    class_names = generate_comprehensive_class_names(num_classes)
    
    # Split samples between train/val/test (70/20/10)
    train_samples = int(num_samples * 0.7)
    val_samples = int(num_samples * 0.2)
    test_samples = num_samples - train_samples - val_samples
    
    print(f"Splitting samples: {train_samples} train, {val_samples} val, {test_samples} test")
    
    # Create sample data for each split
    splits = [
        ('train', train_samples, dirs['images_train'], dirs['labels_train']),
        ('val', val_samples, dirs['images_val'], dirs['labels_train']),
        ('test', test_samples, dirs['images_test'], dirs['labels_test'])
    ]
    
    for split_name, split_count, img_dir, lbl_dir in splits:
        print(f"Creating {split_name} split with {split_count} samples...")
        
        for i in range(split_count):
            # Create a sample image with random shapes
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Add some shapes to simulate construction scene
            for _ in range(np.random.randint(3, 8)):
                # Random shape (rectangle, circle, or line)
                shape_type = np.random.choice(['rect', 'circle', 'line'])
                
                if shape_type == 'rect':
                    pt1 = (np.random.randint(0, 600), np.random.randint(0, 600))
                    pt2 = (pt1[0] + np.random.randint(20, 100), pt1[1] + np.random.randint(20, 100))
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.rectangle(img, pt1, pt2, color, -1)
                elif shape_type == 'circle':
                    center = (np.random.randint(50, 590), np.random.randint(50, 590))
                    radius = np.random.randint(10, 50)
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.circle(img, center, radius, color, -1)
                else:  # line
                    pt1 = (np.random.randint(0, 640), np.random.randint(0, 640))
                    pt2 = (np.random.randint(0, 640), np.random.randint(0, 640))
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    thickness = np.random.randint(1, 5)
                    cv2.line(img, pt1, pt2, color, thickness)
            
            # Save image
            img_path = os.path.join(img_dir, f"sample_{split_name}_{i:04d}.jpg")
            cv2.imwrite(img_path, img)
            
            # Create corresponding label file with random annotations
            label_path = os.path.join(lbl_dir, f"sample_{split_name}_{i:04d}.txt")
            
            with open(label_path, 'w') as f:
                # Add random number of annotations (1 to 5)
                num_annotations = np.random.randint(1, 6)
                
                for _ in range(num_annotations):
                    # Random class ID
                    class_id = np.random.randint(0, num_classes)
                    
                    # Random bounding box (normalized coordinates)
                    x_center = np.random.uniform(0.1, 0.9)
                    y_center = np.random.uniform(0.1, 0.9)
                    width = np.random.uniform(0.05, 0.3)
                    height = np.random.uniform(0.05, 0.3)
                    
                    # Ensure the box stays within bounds
                    x_center = min(max(x_center, width/2), 1 - width/2)
                    y_center = min(max(y_center, height/2), 1 - height/2)
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Create dataset configuration file
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': num_classes,
        'names': class_names
    }
    
    config_path = os.path.join(dataset_path, 'dataset_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Dataset created successfully at: {dataset_path}")
    print(f"Configuration saved to: {config_path}")
    print(f"Total classes: {num_classes}")
    print(f"Total images: {num_samples}")
    print(f"Example class names: {class_names[:10]}...")  # Show first 10 class names


def main():
    print("Sample Dataset Creation for 100-200 Class Safety Detection")
    print("="*60)
    
    dataset_path = input("Enter path to create sample dataset: ").strip()
    if not dataset_path:
        dataset_path = "./sample_dataset"
    
    num_classes = int(input("Enter number of classes (default 100): ").strip() or "100")
    num_samples = int(input("Enter number of sample images (default 50): ").strip() or "50")
    
    create_sample_images_and_labels(dataset_path, num_classes, num_samples)
    
    print("\nDataset creation completed!")
    print(f"You can now use this dataset to test the training script:")
    print(f"python src/train_safety_model.py --dataset_path {dataset_path} --num_classes {num_classes}")


if __name__ == "__main__":
    main()