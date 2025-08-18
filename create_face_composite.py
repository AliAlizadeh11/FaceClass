#!/usr/bin/env python3
"""
Face Detection and Composite Creation Script
============================================

This script processes video frames to:
1. Detect all faces using multiple detection methods
2. Crop and align each detected face
3. Create a composite image with all faces arranged in a grid

Usage:
    python create_face_composite.py [--frames-dir data/frames] [--output output/composite_faces.jpg]
"""

import cv2
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import math
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for MediaPipe availability
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe is available")
except ImportError:
    logger.warning("MediaPipe not available, using fallback methods")

class FaceDetector:
    """Enhanced face detection using multiple methods."""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.detection_methods_used = []
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using multiple methods for maximum coverage."""
        all_faces = []
        
        # Method 1: OpenCV DNN (most reliable)
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
                        self.detection_methods_used.append('OpenCV DNN')
                        
            else:
                raise FileNotFoundError("DNN model files not found")
                
        except Exception as e:
            logger.debug(f"OpenCV DNN detection error: {e}")
            pass
        
        # Method 2: Haar Cascade
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=4, 
                minSize=(20, 20)
            )
            
            for (x, y, w, h) in faces:
                # Check for duplicates
                is_duplicate = False
                for existing_face in all_faces:
                    existing_bbox = existing_face['bbox']
                    overlap_x = max(0, min(x + w, existing_bbox[2]) - max(x, existing_bbox[0]))
                    overlap_y = max(0, min(y + h, existing_bbox[3]) - max(y, existing_bbox[1]))
                    overlap_area = overlap_x * overlap_y
                    face_area = w * h
                    
                    if overlap_area > face_area * 0.5:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_faces.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.8,
                        'method': 'Haar Cascade'
                    })
                    self.detection_methods_used.append('Haar Cascade')
                    
        except Exception as e:
            logger.debug(f"Haar cascade detection error: {e}")
            pass
        
        # Method 3: MediaPipe (if available)
        if MEDIAPIPE_AVAILABLE:
            try:
                with mp_face_detection.FaceDetection(
                    model_selection=1, 
                    min_detection_confidence=self.confidence_threshold
                ) as face_detection:
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            
                            x1 = int(bboxC.xmin * iw)
                            y1 = int(bboxC.ymin * ih)
                            x2 = int((bboxC.xmin + bboxC.width) * iw)
                            y2 = int((bboxC.ymin + bboxC.height) * ih)
                            
                            # Ensure coordinates are within frame bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(iw, x2)
                            y2 = min(ih, y2)
                            
                            # Check for duplicates
                            is_duplicate = False
                            for existing_face in all_faces:
                                existing_bbox = existing_face['bbox']
                                overlap_x = max(0, min(x2, existing_bbox[2]) - max(x1, existing_bbox[0]))
                                overlap_y = max(0, min(y2, existing_bbox[3]) - max(y1, existing_bbox[1]))
                                overlap_area = overlap_x * overlap_y
                                face_area = (x2 - x1) * (y2 - y1)
                                
                                if overlap_area > face_area * 0.5:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                confidence = detection.score[0]
                                all_faces.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(confidence),
                                    'method': 'MediaPipe'
                                })
                                self.detection_methods_used.append('MediaPipe')
                    
            except Exception as e:
                logger.debug(f"MediaPipe detection error: {e}")
                pass
        
        # Remove duplicates and assign IDs
        final_faces = []
        for i, face in enumerate(all_faces):
            face['face_id'] = i + 1
            final_faces.append(face)
        
        return final_faces

class FaceProcessor:
    """Process detected faces for cropping, alignment, and composite creation."""
    
    def __init__(self, target_size: Tuple[int, int] = (200, 200)):
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
    
    def create_composite_grid(self, faces: List[np.ndarray], max_faces_per_row: int = 5) -> np.ndarray:
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
        
        # Create composite canvas
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Place faces in grid
        for i, face in enumerate(faces):
            row = i // max_faces_per_row
            col = i % max_faces_per_row
            
            y_start = row * face_height
            x_start = col * face_width
            
            composite[y_start:y_start + face_height, x_start:x_start + face_width] = face
        
        return composite
    
    def create_collage_layout(self, faces: List[np.ndarray]) -> np.ndarray:
        """Create a more artistic collage layout."""
        if not faces:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Sort faces by size (area) for better layout
        face_areas = [(i, face.shape[0] * face.shape[1]) for i, face in enumerate(faces)]
        face_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Create a larger canvas for collage
        canvas_size = max(800, len(faces) * 100)
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Place largest faces first in center
        center_x, center_y = canvas_size // 2, canvas_size // 2
        
        for i, (face_idx, _) in enumerate(face_areas):
            face = faces[face_idx]
            
            # Calculate position (spiral-like placement)
            angle = i * 2 * math.pi / len(faces)
            radius = 50 + (i * 20)
            
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            
            # Ensure face fits in canvas
            face_h, face_w = face.shape[:2]
            x = max(0, min(x, canvas_size - face_w))
            y = max(0, min(y, canvas_size - face_h))
            
            # Place face
            canvas[y:y + face_h, x:x + face_w] = face
        
        return canvas

def process_frames_directory(frames_dir: str, output_path: str, max_faces_per_row: int = 5):
    """Process all frames in a directory and create face composite."""
    frames_path = Path(frames_dir)
    
    if not frames_path.exists():
        logger.error(f"Frames directory does not exist: {frames_dir}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    frame_files = [f for f in frames_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not frame_files:
        logger.error(f"No image files found in {frames_dir}")
        return
    
    logger.info(f"Found {len(frame_files)} frame files")
    
    # Initialize components
    detector = FaceDetector(confidence_threshold=0.3)
    processor = FaceProcessor(target_size=(200, 200))
    
    all_detected_faces = []
    frame_face_counts = {}
    
    # Process each frame
    for frame_file in frame_files:
        try:
            logger.info(f"Processing frame: {frame_file.name}")
            
            # Read frame
            frame = cv2.imread(str(frame_file))
            if frame is None:
                logger.warning(f"Could not read frame: {frame_file.name}")
                continue
            
            # Detect faces
            faces = detector.detect_faces(frame)
            frame_face_counts[frame_file.name] = len(faces)
            
            logger.info(f"Detected {len(faces)} faces in {frame_file.name}")
            
            # Crop and align each face
            for face_info in faces:
                face_cropped = processor.crop_and_align_face(frame, face_info['bbox'])
                if face_cropped is not None:
                    # Add metadata to face
                    face_with_meta = {
                        'image': face_cropped,
                        'source_frame': frame_file.name,
                        'bbox': face_info['bbox'],
                        'confidence': face_info['confidence'],
                        'method': face_info['method'],
                        'face_id': face_info['face_id']
                    }
                    all_detected_faces.append(face_with_meta)
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_file.name}: {e}")
            continue
    
    logger.info(f"Total faces detected across all frames: {len(all_detected_faces)}")
    
    if not all_detected_faces:
        logger.warning("No faces detected in any frames")
        return
    
    # Create composite images
    try:
        # Create grid composite
        face_images = [face['image'] for face in all_detected_faces]
        grid_composite = processor.create_composite_grid(face_images, max_faces_per_row)
        
        # Create collage composite
        collage_composite = processor.create_collage_layout(face_images)
        
        # Save composites
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save grid composite
        grid_output = output_path.parent / f"{output_path.stem}_grid{output_path.suffix}"
        cv2.imwrite(str(grid_output), grid_composite)
        logger.info(f"Grid composite saved to: {grid_output}")
        
        # Save collage composite
        collage_output = output_path.parent / f"{output_path.stem}_collage{output_path.suffix}"
        cv2.imwrite(str(collage_output), collage_composite)
        logger.info(f"Collage composite saved to: {collage_output}")
        
        # Save detailed report
        report_path = output_path.parent / f"{output_path.stem}_report.txt"
        with open(report_path, 'w') as f:
            f.write("Face Detection and Composite Creation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Frames Directory: {frames_dir}\n")
            f.write(f"Total Frames Processed: {len(frame_files)}\n")
            f.write(f"Total Faces Detected: {len(all_detected_faces)}\n")
            f.write(f"Detection Methods Used: {', '.join(set(detector.detection_methods_used))}\n\n")
            
            f.write("Frame-by-Frame Analysis:\n")
            f.write("-" * 30 + "\n")
            for frame_name, face_count in frame_face_counts.items():
                f.write(f"{frame_name}: {face_count} faces\n")
            
            f.write(f"\nFace Details:\n")
            f.write("-" * 20 + "\n")
            for i, face in enumerate(all_detected_faces):
                f.write(f"Face {i+1}:\n")
                f.write(f"  Source: {face['source_frame']}\n")
                f.write(f"  BBox: {face['bbox']}\n")
                f.write(f"  Confidence: {face['confidence']:.3f}\n")
                f.write(f"  Method: {face['method']}\n\n")
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        # Display summary
        print(f"\n{'='*60}")
        print(f"FACE DETECTION AND COMPOSITE CREATION COMPLETED")
        print(f"{'='*60}")
        print(f"Frames processed: {len(frame_files)}")
        print(f"Total faces detected: {len(all_detected_faces)}")
        print(f"Detection methods: {', '.join(set(detector.detection_methods_used))}")
        print(f"Grid composite: {grid_output}")
        print(f"Collage composite: {collage_output}")
        print(f"Detailed report: {report_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error creating composite: {e}")
        raise

def main():
    """Main function to run the face detection and composite creation."""
    parser = argparse.ArgumentParser(description='Detect faces in video frames and create composite images')
    parser.add_argument('--frames-dir', default='data/frames', 
                       help='Directory containing video frames (default: data/frames)')
    parser.add_argument('--output', default='output/composite_faces.jpg',
                       help='Output path for composite image (default: output/composite_faces.jpg)')
    parser.add_argument('--max-faces-per-row', type=int, default=5,
                       help='Maximum faces per row in grid layout (default: 5)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Face detection confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting face detection and composite creation...")
        process_frames_directory(args.frames_dir, args.output, args.max_faces_per_row)
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
