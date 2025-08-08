#!/usr/bin/env python3
"""
Test script to verify face detection is working on video frames.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import Config
from detection.face_tracker import FaceTracker

def test_face_detection():
    """Test face detection on video frames."""
    print("Testing face detection...")
    
    # Load configuration
    config = Config()
    
    # Initialize face tracker
    face_tracker = FaceTracker(config)
    
    # Test on a few video frames
    frames_dir = Path("data/frames")
    if not frames_dir.exists():
        print("No frames directory found. Please run frame extraction first.")
        return
    
    frame_files = sorted(frames_dir.glob("*.jpg"))[:3]  # Test first 3 frames
    
    for frame_file in frame_files:
        print(f"\nTesting frame: {frame_file.name}")
        
        # Load frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Failed to load frame: {frame_file}")
            continue
        
        print(f"Frame shape: {frame.shape}")
        
        # Detect faces
        detections = face_tracker.detect_faces(frame)
        print(f"Detected {len(detections)} faces")
        
        # Draw detections on frame
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {detection['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated frame
        output_path = f"test_output_{frame_file.stem}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Saved annotated frame: {output_path}")
        
        # Show detection details
        for i, detection in enumerate(detections):
            print(f"  Face {i+1}: bbox={detection['bbox']}, confidence={detection['confidence']:.2f}")

if __name__ == "__main__":
    test_face_detection() 