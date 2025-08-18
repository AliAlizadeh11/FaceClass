#!/usr/bin/env python3
"""
Test Website Upload Process
This script simulates the website upload process to verify that bounding boxes are displayed correctly.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.video_processor import VideoProcessor
from services.visualization import VisualizationService
from config import Config

def test_website_upload_process():
    """Test the complete website upload and processing workflow."""
    print("🧪 Testing Website Upload Process")
    print("=" * 50)
    
    # Initialize configuration and services
    config = Config()
    video_processor = VideoProcessor(config.config)
    visualizer = VisualizationService(config.config)
    
    print("✓ Services initialized")
    
    # Create a test video with faces
    test_video_path = create_test_video_with_faces()
    print(f"✓ Test video created: {test_video_path}")
    
    # Simulate website upload processing
    print("\n📤 Simulating website upload processing...")
    
    # Process video (same as website upload route)
    results = video_processor.process_video(
        str(test_video_path),
        save_annotated_video=True,
        save_results=True
    )
    
    if 'error' in results:
        print(f"❌ Processing failed: {results['error']}")
        return
    
    print("✓ Video processing completed")
    print(f"  - Total frames: {results['video_info']['total_frames']}")
    print(f"  - Total detections: {results['processing_stats']['total_detections']}")
    print(f"  - Processing time: {results['processing_time']:.2f}s")
    
    # Check if annotated video was created
    annotated_video_path = results.get('annotated_video_path')
    if annotated_video_path:
        print(f"✓ Annotated video created: {annotated_video_path}")
        
        # Verify bounding boxes in annotated video
        verify_bounding_boxes_in_video(annotated_video_path)
    else:
        print("❌ No annotated video created")
    
    # Check individual annotated frames
    check_annotated_frames(results)
    
    print(f"\n✅ Website upload test completed!")

def create_test_video_with_faces():
    """Create a test video with simulated faces for testing."""
    output_path = Path('test_website_video.mp4')
    
    # Video properties
    width, height = 640, 480
    fps = 10
    duration = 3  # 3 seconds
    total_frames = duration * fps
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Face positions that will move slightly over time
    base_faces = [
        {'x': 100, 'y': 100, 'w': 80, 'h': 100, 'color': (255, 255, 255)},
        {'x': 300, 'y': 150, 'w': 90, 'h': 110, 'color': (200, 200, 255)},
        {'x': 500, 'y': 120, 'w': 85, 'h': 105, 'color': (255, 200, 200)}
    ]
    
    for frame_id in range(total_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # Add moving faces
        for i, face in enumerate(base_faces):
            # Add slight movement
            movement_x = int(np.sin(frame_id * 0.1 + i) * 3)
            movement_y = int(np.cos(frame_id * 0.05 + i) * 2)
            
            x = face['x'] + movement_x
            y = face['y'] + movement_y
            w, h = face['w'], face['h']
            
            # Draw face region
            cv2.rectangle(frame, (x, y), (x + w, y + h), face['color'], 2)
            
            # Add facial features
            cv2.circle(frame, (x + w//3, y + h//3), 5, (0, 0, 0), -1)  # Left eye
            cv2.circle(frame, (x + 2*w//3, y + h//3), 5, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(frame, (x + w//2, y + 2*h//3), (w//4, h//6), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_id:03d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        video_writer.write(frame)
    
    video_writer.release()
    return output_path

def verify_bounding_boxes_in_video(video_path):
    """Verify that bounding boxes are visible in the annotated video."""
    print(f"\n🔍 Verifying bounding boxes in annotated video...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open annotated video")
        return
    
    frame_count = 0
    frames_with_boxes = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if frame has colored rectangles (bounding boxes)
        # Look for green, yellow, or red rectangles
        has_boxes = False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Check for green (recognized attentive)
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        if cv2.countNonZero(green_mask) > 1000:  # Threshold for significant green areas
            has_boxes = True
        
        # Check for yellow (recognized inattentive)
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        if cv2.countNonZero(yellow_mask) > 1000:
            has_boxes = True
        
        # Check for red (unknown)
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        if cv2.countNonZero(red_mask1) > 1000 or cv2.countNonZero(red_mask2) > 1000:
            has_boxes = True
        
        if has_boxes:
            frames_with_boxes += 1
        
        frame_count += 1
        
        # Save sample frames for inspection
        if frame_count % 10 == 0:
            sample_path = f'test_website_frame_{frame_count:03d}.jpg'
            cv2.imwrite(sample_path, frame)
            print(f"  - Saved sample frame {frame_count}: {sample_path}")
    
    cap.release()
    
    print(f"  - Total frames: {frame_count}")
    print(f"  - Frames with bounding boxes: {frames_with_boxes}")
    print(f"  - Bounding box coverage: {frames_with_boxes/frame_count*100:.1f}%")
    
    if frames_with_boxes > 0:
        print("✓ Bounding boxes detected in annotated video")
    else:
        print("❌ No bounding boxes detected in annotated video")

def check_annotated_frames(results):
    """Check individual annotated frames for bounding boxes."""
    print(f"\n🖼️ Checking individual annotated frames...")
    
    # Look for annotated frame files
    output_dir = Path(results.get('output_directory', 'data/outputs'))
    annotated_frames = list(output_dir.glob('frame_*_annotated.jpg'))
    
    if not annotated_frames:
        print("  - No annotated frame files found")
        return
    
    print(f"  - Found {len(annotated_frames)} annotated frames")
    
    # Check first few frames
    for i, frame_path in enumerate(annotated_frames[:3]):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  - Could not read frame: {frame_path}")
            continue
        
        # Check for bounding boxes (similar to video verification)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        has_boxes = False
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        
        if (cv2.countNonZero(green_mask) > 1000 or 
            cv2.countNonZero(yellow_mask) > 1000 or
            cv2.countNonZero(red_mask1) > 1000 or
            cv2.countNonZero(red_mask2) > 1000):
            has_boxes = True
        
        status = "✓" if has_boxes else "❌"
        print(f"  {status} Frame {frame_path.name}: {'Has boxes' if has_boxes else 'No boxes'}")

def main():
    """Main function."""
    try:
        test_website_upload_process()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
