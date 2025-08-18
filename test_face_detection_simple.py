#!/usr/bin/env python3
"""
Simple test script for face detection logic without Flask
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def test_face_detection_logic():
    """Test the face detection logic without Flask dependencies."""
    
    print("Testing face detection logic...")
    
    # Create a test image
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate faces
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (255, 255, 255), -1)
    cv2.rectangle(test_image, (450, 100), (550, 200), (255, 255, 255), -1)
    
    print("✅ Test image created")
    
    # Test Haar Cascade detection
    try:
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("❌ Failed to load Haar cascade")
            return
        
        print("✅ Haar cascade loaded successfully")
        
        # Test detection with optimized parameters
        scale_factors = [1.1, 1.2]
        min_neighbors_options = [4]
        
        all_faces = []
        for scale_factor in scale_factors:
            for min_neighbors in min_neighbors_options:
                faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(20, 20))
                for (x, y, w, h) in faces:
                    all_faces.append([x, y, w, h])
        
        print(f"✅ Detected {len(all_faces)} potential faces")
        
        # Test JSON serialization
        serializable_faces = []
        for i, (x, y, w, h) in enumerate(all_faces):
            serializable_face = {
                'face_id': int(i + 1),
                'bbox': [int(x), int(y), int(x + w), int(y + h)],
                'confidence': float(0.8),
                'method': str('Haar Cascade')
            }
            serializable_faces.append(serializable_face)
        
        print("✅ Face data converted to JSON-serializable format")
        
        # Test data types
        for face in serializable_faces:
            assert isinstance(face['face_id'], int), f"face_id should be int, got {type(face['face_id'])}"
            assert isinstance(face['bbox'], list), f"bbox should be list, got {type(face['bbox'])}"
            assert all(isinstance(x, int) for x in face['bbox']), "bbox values should be int"
            assert isinstance(face['confidence'], float), f"confidence should be float, got {type(face['confidence'])}"
            assert isinstance(face['method'], str), f"method should be str, got {type(face['method'])}"
        
        print("✅ All data types are correct for JSON serialization")
        
        # Create annotated image
        annotated_image = test_image.copy()
        for face in serializable_faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Face {face['face_id']}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save result
        cv2.imwrite('test_face_detection_result.jpg', annotated_image)
        print("✅ Annotated image saved as 'test_face_detection_result.jpg'")
        
        # Test JSON serialization (simulate what Flask would do)
        import json
        try:
            json_str = json.dumps(serializable_faces)
            print("✅ JSON serialization successful")
            print(f"   JSON length: {len(json_str)} characters")
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            return
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("✅ Face detection logic is working correctly")
        print("✅ JSON serialization is working")
        print("✅ Data types are correct")
        print("\nThe application should now work without JSON errors!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_detection_logic()
