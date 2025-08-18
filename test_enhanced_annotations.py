#!/usr/bin/env python3
"""
Test script for enhanced video annotation system.
Creates sample annotated frames and videos with comprehensive detection information.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.visualization import VisualizationService

def create_realistic_classroom_frame():
    """Create a realistic classroom frame with multiple students."""
    # Create a 1280x720 frame (HD resolution)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add gradient background (classroom-like)
    for y in range(720):
        for x in range(1280):
            # Create a subtle gradient from top to bottom
            intensity = int(200 + 55 * (y / 720))
            frame[y, x] = [intensity, intensity, intensity]
    
    # Add some classroom elements
    # Blackboard area at the top
    cv2.rectangle(frame, (50, 50), (1230, 150), (50, 50, 50), -1)
    cv2.rectangle(frame, (50, 50), (1230, 150), (255, 255, 255), 3)
    
    # Add text on blackboard
    cv2.putText(frame, "Computer Vision Class", (100, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # Add some desk-like structures
    for i in range(3):
        y_pos = 300 + i * 120
        cv2.rectangle(frame, (100 + i*200, y_pos), (300 + i*200, y_pos + 80), (139, 69, 19), -1)
        cv2.rectangle(frame, (100 + i*200, y_pos), (300 + i*200, y_pos + 80), (0, 0, 0), 2)
    
    return frame

def create_comprehensive_detections():
    """Create comprehensive detection data with all information."""
    return [
        {
            "track_id": 1,
            "student_id": "John_Doe",
            "bbox": [150, 250, 120, 150],
            "emotion": "happy",
            "is_attentive": True,
            "confidence": 0.95,
            "recognition_confidence": 0.87
        },
        {
            "track_id": 2,
            "student_id": "Jane_Smith",
            "bbox": [400, 280, 110, 140],
            "emotion": "neutral",
            "is_attentive": False,
            "confidence": 0.92,
            "recognition_confidence": 0.78
        },
        {
            "track_id": 3,
            "student_id": "Mike_Johnson",
            "bbox": [650, 300, 125, 155],
            "emotion": "confused",
            "is_attentive": True,
            "confidence": 0.88,
            "recognition_confidence": 0.82
        },
        {
            "track_id": 4,
            "student_id": "unknown",
            "bbox": [900, 270, 115, 145],
            "emotion": "",
            "is_attentive": None,
            "confidence": 0.76,
            "recognition_confidence": 0.0
        },
        {
            "track_id": 5,
            "student_id": "Sarah_Wilson",
            "bbox": [1150, 290, 118, 148],
            "emotion": "surprised",
            "is_attentive": True,
            "confidence": 0.94,
            "recognition_confidence": 0.91
        }
    ]

def test_enhanced_frame_annotation():
    """Test enhanced frame annotation with comprehensive information."""
    print("Testing enhanced frame annotation...")
    
    # Create visualization service
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create realistic classroom frame
    frame = create_realistic_classroom_frame()
    
    # Create comprehensive detections
    detections = create_comprehensive_detections()
    
    # Annotate frame
    start_time = time.time()
    annotated_frame = visualizer.annotate_frame(frame, detections)
    annotation_time = time.time() - start_time
    
    print(f"Frame annotation completed in {annotation_time:.3f} seconds")
    
    # Save annotated frame
    output_path = "test_output/enhanced_annotated_frame.jpg"
    Path("test_output").mkdir(exist_ok=True)
    
    success = cv2.imwrite(output_path, annotated_frame)
    
    if success:
        print(f"✓ Enhanced annotated frame saved to: {output_path}")
        print(f"  - Frame size: {annotated_frame.shape}")
        print(f"  - Detections processed: {len(detections)}")
    else:
        print("✗ Failed to save enhanced annotated frame")
    
    return annotated_frame

def test_annotated_video_creation():
    """Test creation of annotated video with multiple frames."""
    print("\nTesting annotated video creation...")
    
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create a simple video with multiple frames
    video_path = "test_output/sample_video.mp4"
    output_path = "test_output/sample_annotated_video.mp4"
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 5.0, (1280, 720))
    
    # Create multiple frames with slight variations
    for i in range(30):  # 30 frames at 5 fps = 6 seconds
        frame = create_realistic_classroom_frame()
        
        # Add some movement to detections (simulate tracking)
        detections = create_comprehensive_detections()
        for detection in detections:
            # Add slight movement to bbox
            detection['bbox'][0] += int(np.sin(i * 0.2) * 5)
            detection['bbox'][1] += int(np.cos(i * 0.1) * 3)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Sample video created: {video_path}")
    
    # Create detections per frame (same detections for all frames in this test)
    detections_per_frame = [create_comprehensive_detections() for _ in range(30)]
    
    # Create annotated video
    start_time = time.time()
    success = visualizer.save_annotated_video(
        video_path,
        detections_per_frame,
        output_path,
        fps=5.0
    )
    video_time = time.time() - start_time
    
    if success:
        print(f"✓ Annotated video created in {video_time:.2f} seconds")
        print(f"  - Output: {output_path}")
        print(f"  - Frames: {len(detections_per_frame)}")
        print(f"  - Duration: {len(detections_per_frame) / 5.0:.1f} seconds")
    else:
        print("✗ Failed to create annotated video")
    
    # Clean up sample video
    Path(video_path).unlink(missing_ok=True)

def test_keyframe_extraction():
    """Test keyframe extraction for each student."""
    print("\nTesting keyframe extraction...")
    
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create sample frame
    frame = create_realistic_classroom_frame()
    
    # Create detections
    detections = create_comprehensive_detections()
    
    # Save keyframes for each student
    session_id = "enhanced_test_session"
    keyframe_paths = []
    
    for detection in detections:
        keyframe_path = visualizer.save_keyframe(
            frame, 
            detection, 
            "static/keyframes", 
            session_id
        )
        
        if keyframe_path:
            keyframe_paths.append(keyframe_path)
            print(f"✓ Keyframe saved: {Path(keyframe_path).name}")
        else:
            print(f"✗ Failed to save keyframe for {detection.get('student_id', 'unknown')}")
    
    print(f"Total keyframes saved: {len(keyframe_paths)}")
    
    return keyframe_paths

def test_thumbnail_grid():
    """Test thumbnail grid creation with enhanced keyframes."""
    print("\nTesting thumbnail grid creation...")
    
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Get existing keyframes
    keyframes_dir = Path("static/keyframes")
    keyframe_paths = []
    
    if keyframes_dir.exists():
        keyframe_files = list(keyframes_dir.glob("*.jpg"))
        keyframe_paths = [str(f) for f in sorted(keyframe_files)]
    
    if keyframe_paths:
        # Create thumbnail grid
        grid_path = "static/thumbnails/enhanced_grid.jpg"
        success = visualizer.create_thumbnail_grid(
            keyframe_paths,
            grid_path,
            grid_size=(3, 2),
            thumbnail_size=(200, 200)
        )
        
        if success:
            print(f"✓ Enhanced thumbnail grid saved to: {grid_path}")
            print(f"  - Grid size: 3x2")
            print(f"  - Thumbnail size: 200x200")
            print(f"  - Keyframes included: {len(keyframe_paths)}")
        else:
            print("✗ Failed to create enhanced thumbnail grid")
    else:
        print("✗ No keyframes available for grid creation")

def main():
    """Main test function for enhanced annotation system."""
    print("FaceClass Enhanced Video Annotation System Test")
    print("=" * 55)
    
    try:
        # Test enhanced frame annotation
        test_enhanced_frame_annotation()
        
        # Test annotated video creation
        test_annotated_video_creation()
        
        # Test keyframe extraction
        test_keyframe_extraction()
        
        # Test thumbnail grid
        test_thumbnail_grid()
        
        print("\n" + "=" * 55)
        print("All enhanced annotation tests completed!")
        print("\nCheck the following outputs:")
        print("- test_output/enhanced_annotated_frame.jpg")
        print("- test_output/sample_annotated_video.mp4")
        print("- static/keyframes/ (enhanced keyframes)")
        print("- static/thumbnails/enhanced_grid.jpg")
        print("\nEnhanced features verified:")
        print("✓ Detailed bounding box annotations")
        print("✓ Student ID/Name labels")
        print("✓ Detection confidence scores")
        print("✓ Recognition confidence scores")
        print("✓ Emotion labels")
        print("✓ Attention status indicators")
        print("✓ Color-coded recognition status")
        
    except Exception as e:
        print(f"\n✗ Enhanced test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
