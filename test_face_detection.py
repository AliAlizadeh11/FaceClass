#!/usr/bin/env python3
"""
Test script for face detection functionality
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_face_detection():
    """Test face detection on a sample image or video frame."""
    
    # Create a simple test image with faces (or use a real image)
    print("Testing face detection...")
    
    # Method 1: Test with OpenCV Haar Cascade (always available)
    print("\n1. Testing OpenCV Haar Cascade...")
    try:
        # Create a simple test image (you can replace this with a real image)
        test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Add some colored rectangles to simulate faces
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(test_image, (300, 150), (400, 250), (255, 255, 255), -1)
        cv2.rectangle(test_image, (450, 100), (550, 200), (255, 255, 255), -1)
        
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("   ❌ Failed to load Haar cascade")
        else:
            print("   ✅ Haar cascade loaded successfully")
            
            # Detect faces with multiple parameters
            scale_factors = [1.05, 1.1, 1.15, 1.2]
            min_neighbors_options = [3, 4, 5]
            
            all_faces = []
            for scale_factor in scale_factors:
                for min_neighbors in min_neighbors_options:
                    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(20, 20))
                    for (x, y, w, h) in faces:
                        all_faces.append([x, y, w, h])
            
            print(f"   ✅ Detected {len(all_faces)} potential faces")
            
            # Draw detected faces
            annotated_image = test_image.copy()
            for i, (x, y, w, h) in enumerate(all_faces):
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(annotated_image, f"Face {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save test result
            cv2.imwrite('test_face_detection_result.jpg', annotated_image)
            print("   ✅ Test result saved as 'test_face_detection_result.jpg'")
            
    except Exception as e:
        print(f"   ❌ Haar cascade test failed: {e}")
    
    # Method 2: Test with OpenCV DNN (if available)
    print("\n2. Testing OpenCV DNN...")
    try:
        model_path = "models/face_detection/opencv_face_detector_uint8.pb"
        config_path = "models/face_detection/opencv_face_detector.pbtxt"
        
        if Path(model_path).exists() and Path(config_path).exists():
            net = cv2.dnn.readNet(model_path, config_path)
            print("   ✅ DNN model loaded successfully")
            
            # Test detection
            blob = cv2.dnn.blobFromImage(test_image, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()
            
            print(f"   ✅ DNN model ready, can process {detections.shape[2]} detections")
            
        else:
            print("   ⚠ DNN model files not found, skipping test")
            
    except Exception as e:
        print(f"   ❌ DNN test failed: {e}")
    
    # Method 3: Test with MediaPipe (if available)
    print("\n3. Testing MediaPipe...")
    try:
        import mediapipe as mp
        print("   ✅ MediaPipe imported successfully")
        
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)
            
            if results.detections:
                print(f"   ✅ MediaPipe detected {len(results.detections)} faces")
            else:
                print("   ⚠ MediaPipe detected no faces in test image")
                
    except ImportError:
        print("   ⚠ MediaPipe not available")
    except Exception as e:
        print(f"   ❌ MediaPipe test failed: {e}")
    
    print("\n" + "="*50)
    print("Face Detection Test Summary:")
    print("✅ OpenCV Haar Cascade: Always available")
    print("⚠ OpenCV DNN: Requires model files")
    print("⚠ MediaPipe: Requires installation")
    print("\nTo improve detection:")
    print("1. Use real images/videos instead of test patterns")
    print("2. Lower confidence thresholds")
    print("3. Use multiple detection methods")
    print("4. Adjust scale factors and parameters")

if __name__ == "__main__":
    test_face_detection() 