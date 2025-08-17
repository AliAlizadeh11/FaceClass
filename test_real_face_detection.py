#!/usr/bin/env python3
"""
Test Real Face Detection and Bounding Box Display
This script tests the actual face detection service to ensure bounding boxes are displayed correctly.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.face_detection import FaceDetectionService
from services.visualization import VisualizationService

def test_real_face_detection():
    """Test real face detection with bounding box display."""
    print("🧪 Testing Real Face Detection and Bounding Box Display")
    print("=" * 60)
    
    # Configuration
    config = {
        'face_detection': {
            'model': 'opencv',  # Use OpenCV for reliable detection
            'confidence_threshold': 0.3,
            'nms_threshold': 0.4,
            'min_face_size': 20,
            'max_faces': 50
        },
        'paths': {
            'outputs': 'test_real_detection_output'
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
    
    # Create a test image with faces (or use webcam)
    print("\n📷 Testing with webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam, creating test image instead")
        # Create a test image with simulated faces
        test_image = create_test_image_with_faces()
        test_real_detection_on_image(test_image, face_detector, visualizer, output_dir)
    else:
        print("✓ Webcam opened successfully")
        test_real_detection_on_webcam(cap, face_detector, visualizer, output_dir)
        cap.release()
    
    print(f"\n✅ Test completed! Check output in: {output_dir}")

def create_test_image_with_faces():
    """Create a test image with simulated faces for testing."""
    # Create a 640x480 test image
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
    
    return image

def test_real_detection_on_image(image, face_detector, visualizer, output_dir):
    """Test face detection on a static image."""
    print("🖼️ Testing face detection on test image...")
    
    # Detect faces
    detections = face_detector.detect_faces(image)
    print(f"✓ Detected {len(detections)} faces")
    
    # Prepare detections for visualization
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
    
    # Create annotated image
    annotated_image = visualizer.annotate_frame(image, viz_detections)
    
    # Save results
    cv2.imwrite(str(output_dir / 'test_image_original.jpg'), image)
    cv2.imwrite(str(output_dir / 'test_image_annotated.jpg'), annotated_image)
    
    # Save detection results
    detection_results = {
        'total_faces': len(detections),
        'detections': detections,
        'visualization_detections': viz_detections
    }
    
    import json
    with open(output_dir / 'detection_results.json', 'w') as f:
        json.dump(detection_results, f, indent=2, default=str)
    
    print(f"✓ Test image results saved to {output_dir}")

def test_real_detection_on_webcam(cap, face_detector, visualizer, output_dir):
    """Test face detection on webcam feed."""
    print("📹 Testing face detection on webcam feed...")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            break
        
        # Detect faces
        detections = face_detector.detect_faces(frame)
        
        # Prepare detections for visualization
        viz_detections = []
        for i, detection in enumerate(detections):
            viz_detection = {
                'track_id': i + 1,
                'student_id': f'Webcam_Student_{i + 1}',
                'bbox': detection['bbox'],
                'confidence': detection.get('confidence', 0.8),
                'recognition_confidence': 0.9,
                'emotion': 'neutral',
                'is_attentive': True
            }
            viz_detections.append(viz_detection)
        
        # Create annotated frame
        annotated_frame = visualizer.annotate_frame(frame, viz_detections)
        
        # Add frame info
        info_text = f"Frame: {frame_count} | Faces: {len(detections)} | Press 'q' to quit, 's' to save"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Real Face Detection Test', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            frame_filename = output_dir / f'webcam_frame_{timestamp}.jpg'
            cv2.imwrite(str(frame_filename), annotated_frame)
            print(f"💾 Frame saved: {frame_filename}")
        
        frame_count += 1
        
        # Limit to 100 frames for testing
        if frame_count >= 100:
            print("📊 Reached 100 frames, stopping test")
            break
    
    # Calculate and display results
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n📊 Webcam Test Results:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Time elapsed: {elapsed_time:.2f} seconds")
    print(f"   - Average FPS: {fps:.1f}")
    
    cv2.destroyAllWindows()

def main():
    """Main function."""
    try:
        test_real_face_detection()
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
