"""
Training script for YOLO-based construction safety detection model
Supports 100-200 different dangerous behaviors and safety violations
"""
import os
import yaml
import argparse
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pathlib import Path


def create_dataset_config(num_classes, class_names, dataset_path):
    """
    Create a YAML configuration file for the dataset
    
    Args:
        num_classes (int): Number of classes to detect
        class_names (list): List of class names
        dataset_path (str): Path to the dataset
    """
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
    
    print(f"Dataset configuration saved to {config_path}")
    return config_path


def train_model(config_path, img_size=640, epochs=100, batch_size=16, model_type='yolov8n.pt'):
    """
    Train the YOLO model with the specified configuration
    
    Args:
        config_path (str): Path to the dataset configuration file
        img_size (int): Image size for training
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_type (str): Pretrained model type to use
    """
    # Load a model
    model = YOLO(model_type)  # Load the specified YOLO model
    
    # Train the model
    results = model.train(
        data=config_path,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        save_period=10,  # Save checkpoint every 10 epochs
        cache='ram',     # Cache dataset in RAM for faster training
        device=0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        plots=True,      # Generate training plots
        verbose=True
    )
    
    print("Training completed!")
    return model


def validate_dataset_structure(dataset_path):
    """
    Validate that the dataset follows the required structure
    
    Args:
        dataset_path (str): Path to the dataset
    """
    required_dirs = [
        os.path.join(dataset_path, 'images', 'train'),
        os.path.join(dataset_path, 'images', 'val'),
        os.path.join(dataset_path, 'images', 'test'),
        os.path.join(dataset_path, 'labels', 'train'),
        os.path.join(dataset_path, 'labels', 'val'),
        os.path.join(dataset_path, 'labels', 'test')
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required directory does not exist: {dir_path}")
    
    print("Dataset structure validation passed!")


def create_sample_class_names(num_classes):
    """
    Create sample class names for construction safety violations
    
    Args:
        num_classes (int): Number of classes to create
    """
    # Base safety violation categories
    categories = [
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
    
    # If we have more classes than categories, extend with numbered variants
    if num_classes > len(categories):
        for i in range(len(categories), num_classes):
            categories.append(f"dangerous_behavior_{i:03d}")
    
    return categories[:num_classes]


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for construction safety detection')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of classes to detect (default: 100)')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for training (default: 640)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--model_type', type=str, default='yolov8n.pt',
                        help='Pretrained model type (default: yolov8n.pt)')
    
    args = parser.parse_args()
    
    print(f"Starting training for {args.num_classes} safety violation classes...")
    
    # Validate dataset structure
    validate_dataset_structure(args.dataset_path)
    
    # Create class names
    class_names = create_sample_class_names(args.num_classes)
    print(f"Created {len(class_names)} class names")
    
    # Create dataset configuration
    config_path = create_dataset_config(args.num_classes, class_names, args.dataset_path)
    
    # Train the model
    model = train_model(
        config_path=config_path,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type
    )
    
    print("Model training completed successfully!")
    print(f"Model saved and ready for inference")


if __name__ == "__main__":
    main()