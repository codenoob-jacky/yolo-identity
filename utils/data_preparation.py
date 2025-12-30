"""
Data preparation utilities for construction site safety detection
This script helps organize and prepare datasets for training YOLO models
"""

import os
import shutil
import random
from pathlib import Path
import yaml
from typing import List, Tuple


def create_dataset_structure(base_path: str):
    """
    Create the required directory structure for YOLO training
    
    Args:
        base_path: Base path for the dataset
    """
    directories = [
        os.path.join(base_path, 'images', 'train'),
        os.path.join(base_path, 'images', 'val'),
        os.path.join(base_path, 'images', 'test'),
        os.path.join(base_path, 'labels', 'train'),
        os.path.join(base_path, 'labels', 'val'),
        os.path.join(base_path, 'labels', 'test')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def split_dataset(
    images_path: str, 
    labels_path: str, 
    output_path: str, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.2, 
    test_ratio: float = 0.1
):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        images_path: Path to images directory
        labels_path: Path to labels directory
        output_path: Base path for output dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
    
    # Verify image-label pairs exist
    image_names = {os.path.splitext(f)[0] for f in image_files}
    label_names = {os.path.splitext(f)[0] for f in label_files}
    common_names = image_names.intersection(label_names)
    
    if len(common_names) == 0:
        raise ValueError("No matching image-label pairs found!")
    
    print(f"Found {len(common_names)} image-label pairs")
    
    # Shuffle and split the data
    names_list = list(common_names)
    random.shuffle(names_list)
    
    n_total = len(names_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_names = set(names_list[:n_train])
    val_names = set(names_list[n_train:n_train+n_val])
    test_names = set(names_list[n_train+n_val:])
    
    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")
    
    # Create dataset structure
    create_dataset_structure(output_path)
    
    # Copy files to respective directories
    splits = [
        (train_names, 'train'),
        (val_names, 'val'),
        (test_names, 'test')
    ]
    
    for split_names, split_type in splits:
        img_dst = os.path.join(output_path, 'images', split_type)
        lbl_dst = os.path.join(output_path, 'labels', split_type)
        
        for name in split_names:
            # Copy image
            for img_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                img_src = os.path.join(images_path, name + img_ext)
                if os.path.exists(img_src):
                    shutil.copy2(img_src, os.path.join(img_dst, name + img_ext))
                    break
            
            # Copy label
            lbl_src = os.path.join(labels_path, name + '.txt')
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, os.path.join(lbl_dst, name + '.txt'))


def create_coco_to_yolo_conversion():
    """
    Create utility function to convert COCO format annotations to YOLO format
    """
    pass  # Implementation would go here


def create_label_visualization():
    """
    Create utility to visualize YOLO format labels on images
    """
    pass  # Implementation would go here


def generate_sample_config(dataset_path: str, class_names: List[str]):
    """
    Generate a sample dataset configuration file
    
    Args:
        dataset_path: Path to the dataset
        class_names: List of class names
    """
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    config_path = os.path.join(dataset_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {config_path}")
    return config_path


def main():
    print("Construction Site Safety Dataset Preparation Tool")
    print("=" * 50)
    
    action = input("Select action:\n1. Create dataset structure\n2. Split existing dataset\n3. Generate config file\nChoice (1/2/3): ").strip()
    
    if action == "1":
        base_path = input("Enter base path for dataset: ").strip()
        create_dataset_structure(base_path)
        print("Dataset structure created successfully!")
        
    elif action == "2":
        images_path = input("Enter path to images directory: ").strip()
        labels_path = input("Enter path to labels directory: ").strip()
        output_path = input("Enter output base path: ").strip()
        
        train_ratio = float(input("Enter train ratio (default 0.7): ") or "0.7")
        val_ratio = float(input("Enter validation ratio (default 0.2): ") or "0.2")
        test_ratio = float(input("Enter test ratio (default 0.1): ") or "0.1")
        
        split_dataset(images_path, labels_path, output_path, train_ratio, val_ratio, test_ratio)
        print("Dataset split completed!")
        
    elif action == "3":
        dataset_path = input("Enter dataset path: ").strip()
        print("Enter class names (one per line, empty line to finish):")
        class_names = []
        while True:
            name = input("> ").strip()
            if not name:
                break
            class_names.append(name)
        
        if class_names:
            generate_sample_config(dataset_path, class_names)
            print("Configuration file generated!")
        else:
            print("No class names provided.")
    
    else:
        print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()