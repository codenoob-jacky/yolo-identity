"""
Data preparation utilities for construction site safety detection with 100-200 classes
This script helps organize and prepare datasets for training YOLO models to detect
multiple dangerous behaviors and safety violations
"""

import os
import shutil
import random
import json
from pathlib import Path
import yaml
from typing import List, Tuple
import cv2
import numpy as np
from collections import Counter


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


def analyze_class_distribution(labels_path: str, num_classes: int) -> dict:
    """
    Analyze the distribution of classes in the dataset
    
    Args:
        labels_path: Path to the directory containing label files
        num_classes: Number of classes in the dataset
        
    Returns:
        Dictionary with class distribution statistics
    """
    class_counts = Counter()
    
    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue
            
        with open(os.path.join(labels_path, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                    except ValueError:
                        continue
    
    # Ensure all classes are represented in the result
    result = {}
    for i in range(num_classes):
        result[i] = class_counts.get(i, 0)
    
    return result


def validate_labels_format(labels_path: str, num_classes: int) -> dict:
    """
    Validate YOLO format labels in the specified path
    
    Args:
        labels_path: Path to the directory containing label files
        num_classes: Number of expected classes
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    total_labels = 0
    
    for label_file in os.listdir(labels_path):
        if not label_file.endswith('.txt'):
            continue
            
        label_file_path = os.path.join(labels_path, label_file)
        
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            total_labels += 1
            
            if len(parts) != 5:
                errors.append(f"{label_file}:{line_num} - Invalid format: {line.strip()}")
                continue
                
            try:
                class_id, x_center, y_center, width, height = parts
                class_id = int(class_id)
                x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                
                # Check if class ID is valid
                if class_id < 0 or class_id >= num_classes:
                    errors.append(f"{label_file}:{line_num} - Invalid class ID: {class_id} (expected 0-{num_classes-1})")
                
                # Check if coordinates are normalized (0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    errors.append(f"{label_file}:{line_num} - Coordinates not normalized: {line.strip()}")
                    
            except ValueError:
                errors.append(f"{label_file}:{line_num} - Invalid numeric values: {line.strip()}")
    
    return {
        'total_labels': total_labels,
        'errors': errors,
        'is_valid': len(errors) == 0
    }


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


def generate_comprehensive_class_names(num_classes: int) -> List[str]:
    """
    Generate comprehensive class names for construction safety violations
    
    Args:
        num_classes: Number of classes to generate
        
    Returns:
        List of class names
    """
    # Base safety violation categories
    base_classes = [
        "no_hard_hat", "no_safety_vest", "no_safety_harness", "no_safety_goggles", 
        "no_safety_gloves", "unsafe_posture", "dangerous_position", "no_safety_shoes",
        "equipment_misuse", "unsafe_climbing", "unprotected_edge", "no_hard_hat_area",
        "unauthorized_equipment", "improper_lifting", "unsecured_materials", 
        "unsafe_excavation", "no_lockout_tagout", "slipping_hazard", "falling_objects",
        "electrical_hazard", "chemical_exposure", "fire_hazard", "noise_hazard",
        "confined_space", "unsafe_scaffolding", "unstable_surface", "improper_ventilation",
        "no_gas_monitor", "unsafe_hot_work", "unprotected_wiring", "inadequate_lighting",
        "blocked_entrance", "missing_guardrails", "unsafe_ladder", "overhead_work",
        "unsafe_material_storage", "no_first_aid", "untrained_worker", "unsafe_tool_use",
        "poor_housekeeping", "weather_hazard", "unsafe_transport", "unprotected_openings",
        "unsafe_welding", "hazardous_materials", "unsafe_excavation", "no_spill_containment",
        "improper_stacking", "unsafe_loading", "unprotected_rotating", "unsafe_pressure",
        "no_ventilation", "unsafe_confined_space", "unprotected_electrical", "unsafe_flammable",
        "no_fall_protection", "unsafe_excavation", "improper_chemical", "unsafe_pressure_vessel",
        "unprotected_moving_parts", "unsafe_ventilation", "no_eye_protection", "no_hand_protection",
        "no_foot_protection", "no_head_protection", "no_respiratory_protection", "no_hearing_protection",
        "unsafe_height", "unsafe_excavation_depth", "no_confined_space_entry", "improper_lockout",
        "unsafe_chemical_storage", "no_gas_detection", "unsafe_welding_area", "improper_ventilation",
        "no_hot_work_permit", "unsafe_excavation_shoring", "improper_scaffolding", "unsafe_lifting_equipment",
        "no_crane_clearance", "unsafe_rigging", "improper_load_balance", "no_spotter",
        "unsafe_crane_operation", "improper_hooking", "unsafe_swing_radius", "no_wind_monitoring",
        "unsafe_tower_crane", "improper_outrigger", "no_crane_inspection", "unsafe_lifting_practice",
        "improper_rigging_angle", "no_crane_manual", "unsafe_load_test", "improper_hook_protection",
        "no_crane_operator_cert", "unsafe_crane_assembly", "improper_crane_maintenance", "no_crane_load_chart",
        "unsafe_crane_dismantle", "improper_crane_relocation", "no_crane_weather_limit", "unsafe_crane_repair",
        "improper_crane_operating", "no_crane_communication", "unsafe_crane_load", "improper_crane_setup",
        "no_crane_operator_training", "unsafe_crane_ground_condition", "improper_crane_access", "no_crane_inspection_record",
        "unsafe_crane_electrical", "improper_crane_control", "no_crane_emergency_stop", "unsafe_crane_limit_switch",
        "improper_crane_brake", "no_crane_cable_inspection", "unsafe_crane_hook", "improper_crane_wire_rope",
        "no_crane_safety_factor", "unsafe_crane_load_path", "improper_crane_rotation", "no_crane_load_monitor",
        "unsafe_crane_counterweight", "improper_crane_jib", "no_crane_operating_manual", "unsafe_crane_lifting_angle",
        "improper_crane_attachment", "no_crane_operating_procedure", "unsafe_crane_lifting_zone", "improper_crane_operation_sequence",
        "no_crane_operating_limit", "unsafe_crane_load_distribution", "improper_crane_hook_block", "no_crane_operating_checklist",
        "unsafe_crane_operating_environment", "improper_crane_operating_condition", "no_crane_operating_permission", "unsafe_crane_operating_proximity",
        "improper_crane_operating_height", "no_crane_operating_clearance", "unsafe_crane_operating_surface", "improper_crane_operating_support",
        "no_crane_operating_stability", "unsafe_crane_operating_ground", "improper_crane_operating_foundation", "no_crane_operating_access_control",
        "unsafe_crane_operating_signage", "improper_crane_operating_lighting", "no_crane_operating_communication_equipment", "unsafe_crane_operating_weather",
        "improper_crane_operating_temperature", "no_crane_operating_wind_speed", "unsafe_crane_operating_precipitation", "improper_crane_operating_visibility",
        "no_crane_operating_obstruction", "unsafe_crane_operating_power_line", "improper_crane_operating_structural", "no_crane_operating_adjacent",
        "unsafe_crane_operating_traffic", "improper_crane_operating_pedestrian", "no_crane_operating_emergency", "unsafe_crane_operating_rescue",
        "improper_crane_operating_evacuation", "no_crane_operating_medical", "unsafe_crane_operating_first_aid", "improper_crane_operating_fire",
        "no_crane_operating_safety_equipment", "unsafe_crane_operating_ppe", "improper_crane_operating_protection", "no_crane_operating_isolation",
        "unsafe_crane_operating_barrier", "improper_crane_operating_warning", "no_crane_operating_alert", "unsafe_crane_operating_notification",
        "improper_crane_operating_procedure", "no_crane_operating_instruction", "unsafe_crane_operating_guidance", "improper_crane_operating_direction",
        "no_crane_operating_supervision", "unsafe_crane_operating_monitoring", "improper_crane_operating_observation", "no_crane_operating_inspection",
        "unsafe_crane_operating_testing", "improper_crane_operating_calibration", "no_crane_operating_maintenance", "unsafe_crane_operating_repair",
        "improper_crane_operating_service", "no_crane_operating_lubrication", "unsafe_crane_operating_adjustment", "improper_crane_operating_tuning",
        "no_crane_operating_modification", "unsafe_crane_operating_upgrade", "improper_crane_operating_retrofit", "no_crane_operating_replacement",
        "unsafe_crane_operating_installation", "improper_crane_operating_assembly", "no_crane_operating_dismantle", "unsafe_crane_operating_disposal",
        "improper_crane_operating_recycling", "no_crane_operating_salvage", "unsafe_crane_operating_scrap", "improper_crane_operating_reuse"
    ]
    
    # If we have more classes than base categories, extend with numbered variants
    if num_classes > len(base_classes):
        for i in range(len(base_classes), num_classes):
            base_classes.append(f"dangerous_behavior_{i:03d}")
    
    return base_classes[:num_classes]


def main():
    print("Construction Site Safety Dataset Preparation Tool (100-200 Classes)")
    print("=" * 60)
    
    action = input("Select action:\n1. Create dataset structure\n2. Split existing dataset\n3. Generate config file with safety classes\n4. Analyze class distribution\n5. Validate label format\nChoice (1/2/3/4/5): ").strip()
    
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
        num_classes = int(input("Enter number of classes (100-200): ") or "100")
        
        class_names = generate_comprehensive_class_names(num_classes)
        
        if class_names:
            generate_sample_config(dataset_path, class_names)
            print(f"Configuration file generated with {len(class_names)} safety violation classes!")
        else:
            print("No class names provided.")
    
    elif action == "4":
        labels_path = input("Enter path to labels directory: ").strip()
        num_classes = int(input("Enter number of classes: ") or "100")
        
        distribution = analyze_class_distribution(labels_path, num_classes)
        print("Class Distribution:")
        for class_id, count in distribution.items():
            print(f"  Class {class_id}: {count} instances")
    
    elif action == "5":
        labels_path = input("Enter path to labels directory: ").strip()
        num_classes = int(input("Enter number of classes: ") or "100")
        
        validation_result = validate_labels_format(labels_path, num_classes)
        print(f"Total labels processed: {validation_result['total_labels']}")
        print(f"Validation status: {'VALID' if validation_result['is_valid'] else 'INVALID'}")
        
        if validation_result['errors']:
            print(f"Found {len(validation_result['errors'])} errors:")
            for error in validation_result['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(validation_result['errors']) > 10:
                print(f"  ... and {len(validation_result['errors']) - 10} more errors")
    else:
        print("Invalid choice. Please select 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()