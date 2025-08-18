#!/usr/bin/env python3
"""
Test script for the visualization service.
Creates sample annotated frames and videos to verify functionality.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.visualization import VisualizationService

def create_sample_frame():
    """Create a sample frame with some content."""
    # Create a 640x480 frame with gradient background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(480):
        for x in range(640):
            frame[y, x] = [
                int(255 * x / 640),  # Blue gradient
                int(255 * y / 480),  # Green gradient
                128  # Fixed red
            ]
    
    # Add some text
    cv2.putText(frame, "Sample Classroom Frame", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def create_sample_detections():
    """Create sample detection data."""
    return [
        {
            "track_id": 1,
            "student_id": "John_Doe",
            "bbox": [100, 100, 120, 150],
            "emotion": "happy",
            "is_attentive": True
        },
        {
            "track_id": 2,
            "student_id": "Jane_Smith",
            "bbox": [300, 150, 110, 140],
            "emotion": "neutral",
            "is_attentive": False
        },
        {
            "track_id": 3,
            "student_id": "unknown",
            "bbox": [500, 200, 100, 130],
            "emotion": "",
            "is_attentive": None
        }
    ]

def test_frame_annotation():
    """Test frame annotation functionality."""
    print("Testing frame annotation...")
    
    # Create visualization service
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create sample frame
    frame = create_sample_frame()
    
    # Create sample detections
    detections = create_sample_detections()
    
    # Annotate frame
    annotated_frame = visualizer.annotate_frame(frame, detections)
    
    # Save annotated frame
    output_path = "test_output/annotated_frame.jpg"
    Path("test_output").mkdir(exist_ok=True)
    
    success = cv2.imwrite(output_path, annotated_frame)
    
    if success:
        print(f"✓ Annotated frame saved to: {output_path}")
    else:
        print("✗ Failed to save annotated frame")
    
    return annotated_frame

def test_keyframe_saving():
    """Test keyframe saving functionality."""
    print("\nTesting keyframe saving...")
    
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create sample frame
    frame = create_sample_frame()
    
    # Create sample detection
    detection = {
        "track_id": 1,
        "student_id": "John_Doe",
        "bbox": [100, 100, 120, 150],
        "emotion": "happy",
        "is_attentive": True
    }
    
    # Save keyframe
    keyframe_path = visualizer.save_keyframe(
        frame, 
        detection, 
        "static/keyframes", 
        "test_session"
    )
    
    if keyframe_path:
        print(f"✓ Keyframe saved to: {keyframe_path}")
    else:
        print("✗ Failed to save keyframe")

def test_thumbnail_grid():
    """Test thumbnail grid creation."""
    print("\nTesting thumbnail grid creation...")
    
    config = {
        'paths': {
            'outputs': 'test_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create sample keyframes first
    frame = create_sample_frame()
    detections = create_sample_detections()
    
    keyframe_paths = []
    for i, detection in enumerate(detections):
        keyframe_path = visualizer.save_keyframe(
            frame, 
            detection, 
            "static/keyframes", 
            f"test_session_{i}"
        )
        if keyframe_path:
            keyframe_paths.append(keyframe_path)
    
    if keyframe_paths:
        # Create thumbnail grid
        grid_path = "static/thumbnails/test_grid.jpg"
        success = visualizer.create_thumbnail_grid(
            keyframe_paths,
            grid_path,
            grid_size=(2, 2),
            thumbnail_size=(150, 150)
        )
        
        if success:
            print(f"✓ Thumbnail grid saved to: {grid_path}")
        else:
            print("✗ Failed to create thumbnail grid")
    else:
        print("✗ No keyframes available for grid creation")

def main():
    """Main test function."""
    print("FaceClass Visualization Service Test")
    print("=" * 40)
    
    try:
        # Test frame annotation
        test_frame_annotation()
        
        # Test keyframe saving
        test_keyframe_saving()
        
        # Test thumbnail grid
        test_thumbnail_grid()
        
        print("\n" + "=" * 40)
        print("All tests completed!")
        print("\nCheck the following directories for output:")
        print("- test_output/annotated_frame.jpg")
        print("- static/keyframes/")
        print("- static/thumbnails/")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
