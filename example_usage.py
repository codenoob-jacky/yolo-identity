"""
Example usage of the Construction Site Safety Detection System
This script demonstrates how to use the trained model for inference
"""

import cv2
import numpy as np
from src.detection import SafetyDetector
import os


def example_image_detection():
    """Example of detecting safety violations in an image"""
    print("Example: Image Detection")
    print("-" * 30)
    
    # Initialize detector with a pre-trained model
    # For demonstration, we'll use a standard YOLOv8 model
    # In practice, you'd use your trained safety detection model
    detector = SafetyDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Create a sample image with simulated detections (for demonstration)
    # In real usage, you would load an actual image
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some simulated objects for demonstration
    cv2.rectangle(sample_img, (100, 100), (200, 300), (0, 255, 0), 2)  # Person
    cv2.putText(sample_img, 'person', (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.rectangle(sample_img, (150, 80), (180, 110), (0, 0, 255), 2)  # Hard hat
    cv2.putText(sample_img, 'hard_hat', (150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    print("Processing sample image...")
    annotated_img, violations = detector.detect_safety_violations(sample_img)
    
    print(f"Found {len(violations)} violations in sample image")
    for violation in violations:
        print(f"  - {violation['message']}")
    
    # Save the annotated image
    output_path = os.path.join("output", "annotated_sample.jpg")
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(output_path, annotated_img)
    print(f"Annotated image saved to {output_path}")


def example_video_detection():
    """Example of detecting safety violations in a video"""
    print("\nExample: Video Detection")
    print("-" * 30)
    
    # Initialize detector
    detector = SafetyDetector(model_path='yolov8n.pt', confidence=0.5)
    
    # Note: For actual usage, you would provide a path to a real video file
    print("To run video detection, use:")
    print("  python -m src.detection --input path/to/video.mp4 --output output_video.mp4")
    

def example_realtime_detection():
    """Example of real-time safety detection"""
    print("\nExample: Real-time Detection")
    print("-" * 30)
    
    # Initialize detector
    detector = SafetyDetector(model_path='yolov8n.pt', confidence=0.5)
    
    print("To run real-time detection from webcam, use:")
    print("  python -m src.detection --input webcam")
    

def main():
    print("Construction Site Safety Detection - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_image_detection()
    example_video_detection()
    example_realtime_detection()
    
    print("\nFor full functionality, you will need to:")
    print("1. Train a custom model with safety equipment data")
    print("2. Use the trained model in the detector")
    print("3. Run inference on real construction site images/videos")
    
    print("\nTo train a model:")
    print("  python src/train_safety_model.py")
    
    print("\nTo run inference on an image:")
    print("  python src/detection.py --input path/to/image.jpg --output output.jpg")
    
    print("\nTo run inference on a video:")
    print("  python src/detection.py --input path/to/video.mp4 --output output.mp4")
    
    print("\nTo run real-time detection:")
    print("  python src/detection.py --input webcam")


if __name__ == "__main__":
    main()