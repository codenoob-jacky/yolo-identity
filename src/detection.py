"""
Construction Site Safety Detection using YOLO
Detects safety violations like not wearing hard hats, safety harnesses, etc.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
from typing import List, Tuple, Dict
import argparse


class SafetyDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the safety detector
        
        Args:
            model_path: Path to the YOLO model
            confidence: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_names = {0: 'person', 1: 'hard_hat', 2: 'safety_vest', 3: 'safety_harness'}  # Example classes
        self.colors = {
            'person': (0, 255, 0),      # Green
            'hard_hat': (0, 0, 255),    # Red
            'safety_vest': (255, 0, 0), # Blue
            'safety_harness': (255, 255, 0) # Cyan
        }
        
    def detect_safety_violations(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect safety violations in an image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (annotated_image, violations_list)
        """
        results = self.model(image, conf=self.confidence)
        
        annotated_img = image.copy()
        violations = []
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.class_names.get(cls, f'unknown_{cls}')
                    
                    # Draw bounding box
                    color = self.colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(annotated_img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Analyze safety violations
        violations = self.analyze_violations(results, annotated_img.shape)
        
        return annotated_img, violations
    
    def analyze_violations(self, results, img_shape) -> List[Dict]:
        """
        Analyze detection results to identify safety violations
        
        Args:
            results: YOLO detection results
            img_shape: Shape of the input image
            
        Returns:
            List of violations detected
        """
        violations = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Group detections by class
                detections = {}
                for box in boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if cls not in detections:
                        detections[cls] = []
                    detections[cls].append({
                        'bbox': [x1, y1, x2, y2],
                        'conf': conf
                    })
                
                # Check for safety violations
                # For example: detect people without hard hats
                persons = detections.get(0, [])  # Assuming class 0 is 'person'
                hard_hats = detections.get(1, [])  # Assuming class 1 is 'hard_hat'
                
                for person in persons:
                    person_bbox = person['bbox']
                    person_center = ((person_bbox[0] + person_bbox[2]) // 2, 
                                   (person_bbox[1] + person_bbox[3]) // 2)
                    
                    # Check if there's a hard hat near the person's head
                    has_hard_hat = False
                    for hard_hat in hard_hats:
                        hat_bbox = hard_hat['bbox']
                        hat_center = ((hat_bbox[0] + hat_bbox[2]) // 2, 
                                    (hat_bbox[1] + hat_bbox[3]) // 2)
                        
                        # Simple distance check - in a real implementation, 
                        # you'd want more sophisticated spatial analysis
                        distance = np.sqrt((person_center[0] - hat_center[0])**2 + 
                                         (person_center[1] - hat_center[1])**2)
                        
                        # If hard hat is close to person's head region
                        head_region = person_bbox[1] + (person_bbox[3] - person_bbox[1]) * 0.3
                        if hat_center[1] < head_region and distance < 100:
                            has_hard_hat = True
                            break
                    
                    if not has_hard_hat:
                        violations.append({
                            'type': 'missing_hard_hat',
                            'person_bbox': person_bbox,
                            'confidence': person['conf'],
                            'message': 'Person not wearing hard hat'
                        })
        
        return violations
    
    def detect_from_image(self, image_path: str, output_path: str = None):
        """
        Detect safety violations in an image file
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        annotated_img, violations = self.detect_safety_violations(image)
        
        print(f"Detected {len(violations)} safety violations:")
        for violation in violations:
            print(f"  - {violation['message']} (confidence: {violation['confidence']:.2f})")
        
        if output_path:
            cv2.imwrite(output_path, annotated_img)
            print(f"Output saved to {output_path}")
        
        return annotated_img, violations
    
    def detect_from_video(self, video_path: str, output_path: str = None):
        """
        Detect safety violations in a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_violations = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, violations = self.detect_safety_violations(frame)
            total_violations += len(violations)
            
            if output_path:
                out.write(annotated_frame)
            
            # Display frame with detections (optional)
            cv2.imshow('Safety Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processed {frame_count} frames, total violations: {total_violations}")
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Video processing complete. Total violations detected: {total_violations}")
    
    def detect_realtime(self):
        """
        Perform real-time safety detection using webcam
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise ValueError("Could not open webcam")
        
        print("Starting real-time safety detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame, violations = self.detect_safety_violations(frame)
            
            # Display violation count on frame
            cv2.putText(annotated_frame, f'Violations: {len(violations)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Real-time Safety Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Construction Site Safety Detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='Path to YOLO model')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image/video path or "webcam" for real-time detection')
    parser.add_argument('--output', type=str, 
                       help='Output path for annotated image/video')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    detector = SafetyDetector(model_path=args.model, confidence=args.conf)
    
    if args.input.lower() == 'webcam':
        detector.detect_realtime()
    elif args.input.lower().endswith(('.jpg', '.jpeg', '.png')):
        detector.detect_from_image(args.input, args.output)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detector.detect_from_video(args.input, args.output)
    else:
        print("Unsupported input format. Use image file, video file, or 'webcam'.")


if __name__ == "__main__":
    main()