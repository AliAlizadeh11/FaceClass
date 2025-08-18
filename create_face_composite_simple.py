#!/usr/bin/env python3
"""
Simple Face Detection and Composite Creation Script
==================================================

This script processes keyframes to:
1. Detect all faces using OpenCV and Haar cascade
2. Crop and align each detected face
3. Create a composite image with all faces arranged in a grid

Usage:
    python create_face_composite_simple.py
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFaceDetector:
    """Simple face detection using OpenCV methods."""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV methods."""
        all_faces = []
        
        # Method 1: Haar Cascade (most reliable for this use case)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Multiple detection passes for better coverage
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(30, 30),
                maxSize=(300, 300)
            )
            
            for (x, y, w, h) in faces:
                all_faces.append({
                    'bbox': [x, y, x + w, y + h],
                    'confidence': 0.8,
                    'method': 'Haar Cascade'
                })
                
        except Exception as e:
            logger.error(f"Haar cascade detection error: {e}")
        
        # Method 2: Try to load OpenCV DNN model if available
        try:
            model_path = "models/face_detection/opencv_face_detector_uint8.pb"
            config_path = "models/face_detection/opencv_face_detector.pbtxt"
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                net = cv2.dnn.readNet(model_path, config_path)
                
                # Prepare input blob
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                net.setInput(blob)
                detections = net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.confidence_threshold:
                        # Get bounding box coordinates
                        x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                        y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                        x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                        y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                        
                        all_faces.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'method': 'OpenCV DNN'
                        })
                        
        except Exception as e:
            logger.debug(f"OpenCV DNN detection error: {e}")
        
        # Remove duplicate detections
        unique_faces = self._remove_duplicates(all_faces)
        
        # Assign IDs
        for i, face in enumerate(unique_faces):
            face['face_id'] = i + 1
        
        return unique_faces
    
    def _remove_duplicates(self, faces: List[Dict]) -> List[Dict]:
        """Remove duplicate face detections using IoU."""
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        
        for face in faces:
            is_duplicate = False
            
            for unique_face in unique_faces:
                if self._calculate_iou(face['bbox'], unique_face['bbox']) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class FaceProcessor:
    """Process detected faces for cropping, alignment, and composite creation."""
    
    def __init__(self, target_size: Tuple[int, int] = (150, 150)):
        self.target_size = target_size
        
    def crop_and_align_face(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Crop and align a face from the frame."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Check if bbox is valid
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract face region
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Resize to target size
            face_resized = cv2.resize(face_region, self.target_size)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            return None
    
    def create_composite_grid(self, faces: List[np.ndarray], max_faces_per_row: int = 6) -> np.ndarray:
        """Create a grid composite of all faces."""
        if not faces:
            logger.warning("No faces to create composite")
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Calculate grid dimensions
        num_faces = len(faces)
        num_rows = math.ceil(num_faces / max_faces_per_row)
        num_cols = min(num_faces, max_faces_per_row)
        
        # Calculate composite dimensions
        face_height, face_width = self.target_size
        composite_height = num_rows * face_height
        composite_width = num_cols * face_width
        
        # Create composite canvas with white background
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255
        
        # Place faces in grid
        for i, face in enumerate(faces):
            row = i // max_faces_per_row
            col = i % max_faces_per_row
            
            y_start = row * face_height
            x_start = col * face_width
            
            composite[y_start:y_start + face_height, x_start:x_start + face_width] = face
        
        return composite

def process_keyframes():
    """Process keyframes to detect faces and create composite."""
    # Look for keyframes in data/frames
    frames_dir = Path("data/frames")
    
    if not frames_dir.exists():
        logger.error(f"Frames directory does not exist: {frames_dir}")
        return
    
    # Find keyframe files
    keyframe_files = [f for f in frames_dir.iterdir() 
                     if f.name.startswith('keyframe_') and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    if not keyframe_files:
        logger.error("No keyframe files found")
        return
    
    logger.info(f"Found {len(keyframe_files)} keyframe files")
    
    # Initialize components
    detector = SimpleFaceDetector(confidence_threshold=0.3)
    processor = FaceProcessor(target_size=(150, 150))
    
    all_detected_faces = []
    frame_face_counts = {}
    
    # Process each keyframe
    for keyframe_file in sorted(keyframe_files):
        try:
            logger.info(f"Processing keyframe: {keyframe_file.name}")
            
            # Read frame
            frame = cv2.imread(str(keyframe_file))
            if frame is None:
                logger.warning(f"Could not read keyframe: {keyframe_file.name}")
                continue
            
            # Detect faces
            faces = detector.detect_faces(frame)
            frame_face_counts[keyframe_file.name] = len(faces)
            
            logger.info(f"Detected {len(faces)} faces in {keyframe_file.name}")
            
            # Crop and align each face
            for face_info in faces:
                face_cropped = processor.crop_and_align_face(frame, face_info['bbox'])
                if face_cropped is not None:
                    # Add metadata to face
                    face_with_meta = {
                        'image': face_cropped,
                        'source_frame': keyframe_file.name,
                        'bbox': face_info['bbox'],
                        'confidence': face_info['confidence'],
                        'method': face_info['method'],
                        'face_id': face_info['face_id']
                    }
                    all_detected_faces.append(face_with_meta)
            
        except Exception as e:
            logger.error(f"Error processing keyframe {keyframe_file.name}: {e}")
            continue
    
    logger.info(f"Total faces detected across all keyframes: {len(all_detected_faces)}")
    
    if not all_detected_faces:
        logger.warning("No faces detected in any keyframes")
        return
    
    # Create composite image
    try:
        # Create grid composite
        face_images = [face['image'] for face in all_detected_faces]
        grid_composite = processor.create_composite_grid(face_images, max_faces_per_row=6)
        
        # Save composite
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "keyframe_faces_composite.jpg"
        cv2.imwrite(str(output_path), grid_composite)
        logger.info(f"Composite saved to: {output_path}")
        
        # Save detailed report
        report_path = output_dir / "keyframe_faces_report.txt"
        with open(report_path, 'w') as f:
            f.write("Keyframe Face Detection Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Keyframes Processed: {len(keyframe_files)}\n")
            f.write(f"Total Faces Detected: {len(all_detected_faces)}\n\n")
            
            f.write("Keyframe Analysis:\n")
            f.write("-" * 20 + "\n")
            for frame_name, face_count in frame_face_counts.items():
                f.write(f"{frame_name}: {face_count} faces\n")
            
            f.write(f"\nFace Details:\n")
            f.write("-" * 15 + "\n")
            for i, face in enumerate(all_detected_faces):
                f.write(f"Face {i+1}:\n")
                f.write(f"  Source: {face['source_frame']}\n")
                f.write(f"  BBox: {face['bbox']}\n")
                f.write(f"  Confidence: {face['confidence']:.3f}\n")
                f.write(f"  Method: {face['method']}\n\n")
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        # Display summary
        print(f"\n{'='*60}")
        print(f"KEYFRAME FACE DETECTION COMPLETED")
        print(f"{'='*60}")
        print(f"Keyframes processed: {len(keyframe_files)}")
        print(f"Total faces detected: {len(all_detected_faces)}")
        print(f"Composite image: {output_path}")
        print(f"Detailed report: {report_path}")
        print(f"{'='*60}")
        
        # Also save individual face crops for inspection
        faces_dir = output_dir / "individual_faces"
        faces_dir.mkdir(exist_ok=True)
        
        for i, face in enumerate(all_detected_faces):
            face_path = faces_dir / f"face_{i+1:03d}_{face['source_frame']}"
            cv2.imwrite(str(face_path), face['image'])
        
        logger.info(f"Individual face crops saved to: {faces_dir}")
        
    except Exception as e:
        logger.error(f"Error creating composite: {e}")
        raise

def main():
    """Main function."""
    try:
        logger.info("Starting keyframe face detection and composite creation...")
        process_keyframes()
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
