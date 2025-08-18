#!/usr/bin/env python3
"""
Test Video Processing with Bounding Boxes
This script tests the complete video processing pipeline to ensure bounding boxes are displayed in each frame.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.face_detection import FaceDetectionService
from services.visualization import VisualizationService
from frame_by_frame_analysis import FrameByFrameAnalyzer

def create_test_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a test video with simulated faces for testing.
    
    Args:
        output_path: Path to save the test video
        duration: Video duration in seconds
        fps: Frames per second
    """
    print(f"🎬 Creating test video: {duration}s at {fps} FPS")
    
    # Video properties
    width, height = 640, 480
    total_frames = duration * fps
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
    print(f"✓ Test video created: {output_path}")

def test_video_processing():
    """Test the complete video processing pipeline."""
    print("🧪 Testing Video Processing with Bounding Boxes")
    print("=" * 60)
    
    # Configuration
    config = {
        'face_detection': {
            'model': 'opencv',
            'confidence_threshold': 0.3,
            'nms_threshold': 0.4,
            'min_face_size': 20,
            'max_faces': 50
        },
        'paths': {
            'outputs': 'test_video_output',
            'processed_videos': 'test_video_output',
            'keyframes': 'test_video_output',
            'thumbnails': 'test_video_output'
        }
    }
    
    # Create output directory
    output_dir = Path(config['paths']['outputs'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test video
    test_video_path = output_dir / 'test_video.mp4'
    create_test_video(str(test_video_path), duration=3, fps=10)  # 3 seconds, 10 FPS for faster testing
    
    # Test 1: Direct face detection on video frames
    print("\n🔍 Test 1: Direct Face Detection on Video Frames")
    test_direct_detection(test_video_path, config, output_dir)
    
    # Test 2: Frame-by-frame analysis
    print("\n🎬 Test 2: Frame-by-Frame Analysis")
    test_frame_by_frame_analysis(test_video_path, config, output_dir)
    
    print(f"\n✅ All tests completed! Check output in: {output_dir}")

def test_direct_detection(video_path: str, config: dict, output_dir: Path):
    """Test direct face detection on video frames."""
    print("  - Testing direct face detection...")
    
    # Initialize services
    face_detector = FaceDetectionService(config)
    visualizer = VisualizationService(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    all_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = face_detector.detect_faces(frame)
        
        # Prepare for visualization
        viz_detections = []
        for i, detection in enumerate(detections):
            viz_detection = {
                'track_id': i + 1,
                'student_id': f'Frame_{frame_count}_Student_{i + 1}',
                'bbox': detection['bbox'],
                'confidence': detection.get('confidence', 0.8),
                'recognition_confidence': 0.9,
                'emotion': 'neutral',
                'is_attentive': True
            }
            viz_detections.append(viz_detection)
        
        # Create annotated frame
        annotated_frame = visualizer.annotate_frame(frame, viz_detections)
        
        # Save every 5th frame
        if frame_count % 5 == 0:
            frame_filename = output_dir / f'direct_detection_frame_{frame_count:03d}.jpg'
            cv2.imwrite(str(frame_filename), annotated_frame)
        
        all_detections.extend(detections)
        frame_count += 1
    
    cap.release()
    
    print(f"    ✓ Processed {frame_count} frames")
    print(f"    ✓ Total detections: {len(all_detections)}")
    print(f"    ✓ Sample frames saved")

def test_frame_by_frame_analysis(video_path: str, config: dict, output_dir: Path):
    """Test the frame-by-frame analysis system."""
    print("  - Testing frame-by-frame analysis...")
    
    try:
        # Initialize analyzer
        analyzer = FrameByFrameAnalyzer(config)
        
        # Run analysis
        results = analyzer.analyze_video_frame_by_frame(video_path, str(output_dir / 'frame_analysis'))
        
        print(f"    ✓ Analysis completed")
        print(f"    ✓ Processing time: {results['processing_info']['processing_time']:.2f}s")
        print(f"    ✓ Total tracks: {results['tracking_summary']['total_tracks']}")
        
    except Exception as e:
        print(f"    ❌ Frame-by-frame analysis failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    try:
        test_video_processing()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
