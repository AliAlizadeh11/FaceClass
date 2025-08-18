#!/usr/bin/env python3
"""
Enhanced Face Detection and Composite Creation Script
====================================================

This script provides enhanced face processing with:
1. Face quality assessment and filtering
2. Better face alignment and cropping
3. Multiple composite layout options
4. Face clustering by similarity
5. Enhanced visualization

Usage:
    python create_enhanced_face_composite.py [--frames-dir data/frames] [--output output/enhanced_composite.jpg]
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
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    """Enhanced face detection with quality assessment."""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.detection_methods_used = []
        
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using multiple methods with quality assessment."""
        all_faces = []
        
        # Method 1: Haar Cascade with multiple scales
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Multiple detection passes for better coverage
            scale_factors = [1.05, 1.1, 1.15]
            min_neighbors_options = [3, 4, 5]
            
            for scale_factor in scale_factors:
                for min_neighbors in min_neighbors_options:
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=scale_factor, 
                        minNeighbors=min_neighbors, 
                        minSize=(25, 25),
                        maxSize=(300, 300)
                    )
                    
                    for (x, y, w, h) in faces:
                        # Calculate face quality metrics
                        face_region = gray[y:y+h, x:x+w]
                        quality_score = self._assess_face_quality(face_region, w, h)
                        
                        all_faces.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': 0.8,
                            'method': 'Haar Cascade',
                            'quality_score': quality_score,
                            'size': (w, h)
                        })
                        self.detection_methods_used.append('Haar Cascade')
                        
        except Exception as e:
            logger.error(f"Haar cascade detection error: {e}")
        
        # Method 2: OpenCV DNN if available
        try:
            model_path = "models/face_detection/opencv_face_detector_uint8.pb"
            config_path = "models/face_detection/opencv_face_detector.pbtxt"
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                net = cv2.dnn.readNet(model_path, config_path)
                
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                net.setInput(blob)
                detections = net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.confidence_threshold:
                        x1 = int(detections[0, 0, i, 3] * frame.shape[1])
                        y1 = int(detections[0, 0, i, 4] * frame.shape[0])
                        x2 = int(detections[0, 0, i, 5] * frame.shape[1])
                        y2 = int(detections[0, 0, i, 6] * frame.shape[0])
                        
                        w, h = x2 - x1, y2 - y1
                        face_region = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None
                        
                        if face_region is not None:
                            quality_score = self._assess_face_quality(face_region, w, h)
                            
                            all_faces.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'method': 'OpenCV DNN',
                                'quality_score': quality_score,
                                'size': (w, h)
                            })
                            self.detection_methods_used.append('OpenCV DNN')
                        
        except Exception as e:
            logger.debug(f"OpenCV DNN detection error: {e}")
        
        # Remove duplicates and filter by quality
        unique_faces = self._remove_duplicates(all_faces)
        quality_faces = self._filter_by_quality(unique_faces)
        
        # Assign IDs
        for i, face in enumerate(quality_faces):
            face['face_id'] = i + 1
        
        return quality_faces
    
    def _assess_face_quality(self, face_region: np.ndarray, width: int, height: int) -> float:
        """Assess face quality based on size, contrast, and clarity."""
        if face_region.size == 0:
            return 0.0
        
        # Size score (prefer larger faces)
        size_score = min(1.0, (width * height) / (100 * 100))
        
        # Contrast score
        contrast = np.std(face_region)
        contrast_score = min(1.0, contrast / 50.0)
        
        # Sharpness score (using Laplacian variance)
        laplacian = cv2.Laplacian(face_region, cv2.CV_64F)
        sharpness = np.var(laplacian)
        sharpness_score = min(1.0, sharpness / 100.0)
        
        # Overall quality score
        quality = (size_score * 0.4 + contrast_score * 0.3 + sharpness_score * 0.3)
        
        return quality
    
    def _filter_by_quality(self, faces: List[Dict], min_quality: float = 0.3) -> List[Dict]:
        """Filter faces by quality score."""
        return [face for face in faces if face['quality_score'] >= min_quality]
    
    def _remove_duplicates(self, faces: List[Dict]) -> List[Dict]:
        """Remove duplicate face detections using IoU."""
        if len(faces) <= 1:
            return faces
        
        unique_faces = []
        
        for face in faces:
            is_duplicate = False
            
            for unique_face in unique_faces:
                if self._calculate_iou(face['bbox'], unique_face['bbox']) > 0.4:
                    # Keep the one with higher quality
                    if face['quality_score'] > unique_face['quality_score']:
                        unique_faces.remove(unique_face)
                        unique_faces.append(face)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class EnhancedFaceProcessor:
    """Enhanced face processing with better alignment and layouts."""
    
    def __init__(self, target_size: Tuple[int, int] = (180, 180)):
        self.target_size = target_size
        
    def crop_and_align_face(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Crop and align a face with padding and enhancement."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract face region with padding
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Add padding (20% on each side)
            pad_x = int(face_width * 0.2)
            pad_y = int(face_height * 0.2)
            
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            
            face_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if face_region.size == 0:
                return None
            
            # Resize to target size
            face_resized = cv2.resize(face_region, self.target_size)
            
            # Apply slight enhancement
            face_enhanced = self._enhance_face(face_resized)
            
            return face_enhanced
            
        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            return None
    
    def _enhance_face(self, face: np.ndarray) -> np.ndarray:
        """Apply basic enhancement to face image."""
        try:
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            
            # Enhance L channel (lightness)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.debug(f"Enhancement failed: {e}")
            return face
    
    def create_enhanced_grid(self, faces: List[np.ndarray], max_faces_per_row: int = 8) -> np.ndarray:
        """Create an enhanced grid composite with better spacing."""
        if not faces:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        num_faces = len(faces)
        num_rows = math.ceil(num_faces / max_faces_per_row)
        num_cols = min(num_faces, max_faces_per_row)
        
        # Calculate dimensions with spacing
        face_height, face_width = self.target_size
        spacing = 10  # pixels between faces
        
        composite_height = num_rows * face_height + (num_rows - 1) * spacing
        composite_width = num_cols * face_width + (num_cols - 1) * spacing
        
        # Create white background
        composite = np.ones((composite_height, composite_width, 3), dtype=np.uint8) * 255
        
        # Place faces with spacing
        for i, face in enumerate(faces):
            row = i // max_faces_per_row
            col = i % max_faces_per_row
            
            y_start = row * (face_height + spacing)
            x_start = col * (face_width + spacing)
            
            composite[y_start:y_start + face_height, x_start:x_start + face_width] = face
        
        return composite
    
    def create_circular_layout(self, faces: List[np.ndarray]) -> np.ndarray:
        """Create a circular layout composite."""
        if not faces:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Calculate canvas size
        canvas_size = max(800, len(faces) * 120)
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        center_x, center_y = canvas_size // 2, canvas_size // 2
        
        # Place faces in a circle
        for i, face in enumerate(faces):
            angle = i * 2 * math.pi / len(faces)
            radius = 150 + (i * 10)  # Increasing radius for each face
            
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            
            # Ensure face fits in canvas
            face_h, face_w = face.shape[:2]
            x = max(0, min(x - face_w//2, canvas_size - face_w))
            y = max(0, min(y - face_h//2, canvas_size - face_h))
            
            canvas[y:y + face_h, x:x + face_w] = face
        
        return canvas
    
    def create_pyramid_layout(self, faces: List[np.ndarray]) -> np.ndarray:
        """Create a pyramid layout composite."""
        if not faces:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Sort faces by quality/size for better pyramid
        face_sizes = [(i, face.shape[0] * face.shape[1]) for i, face in enumerate(faces)]
        face_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate canvas size
        canvas_size = max(1000, len(faces) * 100)
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        center_x = canvas_size // 2
        
        # Place largest faces at bottom, smaller at top
        current_y = canvas_size - 200
        faces_per_row = 1
        face_index = 0
        
        while face_index < len(faces) and current_y > 0:
            # Calculate spacing for this row
            row_width = faces_per_row * self.target_size[0]
            spacing = (canvas_size - row_width) // (faces_per_row + 1)
            
            # Place faces in this row
            for i in range(faces_per_row):
                if face_index >= len(faces):
                    break
                
                face_idx = face_sizes[face_index][0]
                face = faces[face_idx]
                
                x = spacing + i * (self.target_size[0] + spacing)
                y = current_y
                
                canvas[y:y + self.target_size[1], x:x + self.target_size[0]] = face
                face_index += 1
            
            # Move to next row
            current_y -= self.target_size[1] + 20
            faces_per_row = min(faces_per_row + 1, 8)  # Increase faces per row
        
        return canvas

def process_frames_enhanced(frames_dir: str, output_path: str):
    """Process frames with enhanced face detection and processing."""
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
    detector = EnhancedFaceDetector(confidence_threshold=0.3)
    processor = EnhancedFaceProcessor(target_size=(180, 180))
    
    all_detected_faces = []
    frame_face_counts = {}
    
    # Process each frame
    for frame_file in sorted(frame_files):
        try:
            logger.info(f"Processing frame: {frame_file.name}")
            
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            # Detect faces
            faces = detector.detect_faces(frame)
            frame_face_counts[frame_file.name] = len(faces)
            
            logger.info(f"Detected {len(faces)} faces in {frame_file.name}")
            
            # Crop and align each face
            for face_info in faces:
                face_cropped = processor.crop_and_align_face(frame, face_info['bbox'])
                if face_cropped is not None:
                    face_with_meta = {
                        'image': face_cropped,
                        'source_frame': frame_file.name,
                        'bbox': face_info['bbox'],
                        'confidence': face_info['confidence'],
                        'method': face_info['method'],
                        'quality_score': face_info['quality_score'],
                        'face_id': face_info['face_id']
                    }
                    all_detected_faces.append(face_with_meta)
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_file.name}: {e}")
            continue
    
    logger.info(f"Total faces detected: {len(all_detected_faces)}")
    
    if not all_detected_faces:
        logger.warning("No faces detected")
        return
    
    # Sort faces by quality
    all_detected_faces.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Create multiple composite layouts
    try:
        face_images = [face['image'] for face in all_detected_faces]
        
        # Create different layouts
        grid_composite = processor.create_enhanced_grid(face_images, max_faces_per_row=8)
        circular_composite = processor.create_circular_layout(face_images)
        pyramid_composite = processor.create_pyramid_layout(face_images)
        
        # Save composites
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save different layouts
        cv2.imwrite(str(output_path.parent / "enhanced_grid.jpg"), grid_composite)
        cv2.imwrite(str(output_path.parent / "enhanced_circular.jpg"), circular_composite)
        cv2.imwrite(str(output_path.parent / "enhanced_pyramid.jpg"), pyramid_composite)
        
        # Save detailed report
        report_path = output_path.parent / "enhanced_faces_report.txt"
        with open(report_path, 'w') as f:
            f.write("Enhanced Face Detection Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Processed: {len(frame_files)}\n")
            f.write(f"Total Faces Detected: {len(all_detected_faces)}\n")
            f.write(f"Detection Methods: {', '.join(set(detector.detection_methods_used))}\n\n")
            
            f.write("Frame Analysis:\n")
            f.write("-" * 20 + "\n")
            for frame_name, face_count in frame_face_counts.items():
                f.write(f"{frame_name}: {face_count} faces\n")
            
            f.write(f"\nTop Quality Faces:\n")
            f.write("-" * 20 + "\n")
            for i, face in enumerate(all_detected_faces[:10]):  # Top 10
                f.write(f"Face {i+1}:\n")
                f.write(f"  Source: {face['source_frame']}\n")
                f.write(f"  Quality Score: {face['quality_score']:.3f}\n")
                f.write(f"  Method: {face['method']}\n\n")
        
        logger.info(f"Enhanced composites and report saved")
        
        # Display summary
        print(f"\n{'='*60}")
        print(f"ENHANCED FACE DETECTION COMPLETED")
        print(f"{'='*60}")
        print(f"Frames processed: {len(frame_files)}")
        print(f"Total faces detected: {len(all_detected_faces)}")
        print(f"Top quality face score: {all_detected_faces[0]['quality_score']:.3f}")
        print(f"Output files:")
        print(f"  - Enhanced Grid: {output_path.parent}/enhanced_grid.jpg")
        print(f"  - Enhanced Circular: {output_path.parent}/enhanced_circular.jpg")
        print(f"  - Enhanced Pyramid: {output_path.parent}/enhanced_pyramid.jpg")
        print(f"  - Report: {report_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Error creating enhanced composites: {e}")
        raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced face detection and composite creation')
    parser.add_argument('--frames-dir', default='data/frames', 
                       help='Directory containing video frames')
    parser.add_argument('--output', default='output/enhanced_composite.jpg',
                       help='Output path for composite images')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting enhanced face detection and composite creation...")
        process_frames_enhanced(args.frames_dir, args.output)
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
