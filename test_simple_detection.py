#!/usr/bin/env python3
"""
Simple test for face detection and bounding box display
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.face_detection import FaceDetectionService
from services.visualization import VisualizationService

def main():
    """Test face detection and visualization."""
    print("🧪 Simple Face Detection Test")
    
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
            'outputs': 'test_simple_output'
        }
    }
    
    # Create output directory
    output_dir = Path(config['paths']['outputs'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize services
    print("🔧 Initializing services...")
    face_detector = FaceDetectionService(config)
    visualizer = VisualizationService(config)
    print("✓ Services initialized")
    
    # Create test image with simulated faces
    print("🖼️ Creating test image...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw some simulated face regions
    face_positions = [
        (100, 100, 80, 100),   # Left face
        (300, 150, 90, 110),   # Center face
        (500, 120, 85, 105)    # Right face
    ]
    
    for x, y, w, h in face_positions:
        # Draw face region (simplified)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        # Add some features
        cv2.circle(image, (x + w//3, y + h//3), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(image, (x + 2*w//3, y + h//3), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(image, (x + w//2, y + 2*h//3), (w//4, h//6), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    # Save original test image
    cv2.imwrite(str(output_dir / 'test_image_original.jpg'), image)
    print("✓ Test image created")
    
    # Test face detection
    print("🔍 Testing face detection...")
    detections = face_detector.detect_faces(image)
    print(f"✓ Detected {len(detections)} faces")
    
    # Print detection details
    for i, detection in enumerate(detections):
        print(f"  Face {i+1}: bbox={detection['bbox']}, confidence={detection.get('confidence', 'N/A')}")
    
    # Test visualization with simulated detections
    print("🎨 Testing visualization...")
    
    # Create simulated detections for visualization
    viz_detections = []
    for i, detection in enumerate(detections):
        viz_detection = {
            'track_id': i + 1,
            'student_id': f'Test_Student_{i + 1}',
            'bbox': detection['bbox'],
            'confidence': detection.get('confidence', 0.8),
            'recognition_confidence': 0.9,
            'emotion': 'neutral',
            'is_attentive': True
        }
        viz_detections.append(viz_detection)
    
    # If no real detections, create some test ones
    if not viz_detections:
        print("⚠️ No real detections, creating test detections for visualization")
        viz_detections = [
            {
                'track_id': 1,
                'student_id': 'Test_Student_1',
                'bbox': [100, 100, 80, 100],
                'confidence': 0.8,
                'recognition_confidence': 0.9,
                'emotion': 'neutral',
                'is_attentive': True
            },
            {
                'track_id': 2,
                'student_id': 'Test_Student_2',
                'bbox': [300, 150, 90, 110],
                'confidence': 0.85,
                'recognition_confidence': 0.88,
                'emotion': 'happy',
                'is_attentive': True
            }
        ]
    
    # Create annotated image
    annotated_image = visualizer.annotate_frame(image, viz_detections)
    
    # Save annotated image
    cv2.imwrite(str(output_dir / 'test_image_annotated.jpg'), annotated_image)
    print("✓ Annotated image created")
    
    # Save detection results
    import json
    detection_results = {
        'total_faces': len(detections),
        'real_detections': detections,
        'visualization_detections': viz_detections
    }
    
    with open(output_dir / 'detection_results.json', 'w') as f:
        json.dump(detection_results, f, indent=2, default=str)
    
    print(f"\n✅ Test completed! Check output in: {output_dir}")
    print("Files created:")
    print(f"  - {output_dir / 'test_image_original.jpg'}")
    print(f"  - {output_dir / 'test_image_annotated.jpg'}")
    print(f"  - {output_dir / 'detection_results.json'}")

if __name__ == "__main__":
    main()
