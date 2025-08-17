#!/usr/bin/env python3
"""
Test script for FaceClass face detection, tracking, and recognition pipeline.
Tests the complete pipeline on sample videos and images.
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from config import Config
from services.video_processor import VideoProcessor
from services.face_detection import FaceDetectionService
from services.face_tracking import FaceTrackingService
from services.face_recognition import FaceRecognitionService

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_face_pipeline.log')
        ]
    )

def test_face_detection_service(config):
    """Test face detection service with sample images."""
    print("\n" + "="*50)
    print("Testing Face Detection Service")
    print("="*50)
    
    try:
        # Initialize detection service
        detector = FaceDetectionService(config)
        print(f"✓ Detection service initialized with {detector.get_model_info()['model_type']} model")
        
        # Test with sample images
        sample_dir = Path("data/frames")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg"))[:3]  # Test first 3 images
            
            for img_path in sample_images:
                print(f"\nTesting detection on: {img_path.name}")
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  ✗ Failed to load image: {img_path}")
                    continue
                
                # Detect faces
                detections = detector.detect_faces(image)
                print(f"  ✓ Found {len(detections)} faces")
                
                for i, detection in enumerate(detections):
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    print(f"    Face {i+1}: bbox={bbox}, confidence={confidence:.3f}")
        
        else:
            print("  ⚠ No sample images found in data/frames/")
            
    except Exception as e:
        print(f"  ✗ Face detection test failed: {e}")
        return False
    
    return True

def test_face_tracking_service(config):
    """Test face tracking service."""
    print("\n" + "="*50)
    print("Testing Face Tracking Service")
    print("="*50)
    
    try:
        # Initialize tracking service
        tracker = FaceTrackingService(config)
        print("✓ Tracking service initialized")
        
        # Create mock detections for testing
        mock_detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.9, 'frame_id': 0},
            {'bbox': [110, 105, 50, 50], 'confidence': 0.8, 'frame_id': 1},
            {'bbox': [120, 110, 50, 50], 'confidence': 0.85, 'frame_id': 2},
        ]
        
        # Test tracking
        for i, detection in enumerate(mock_detections):
            tracked_detections = tracker.update([detection], i)
            print(f"  Frame {i}: {len(tracked_detections)} tracked objects")
            
            for tracked in tracked_detections:
                if 'track_id' in tracked:
                    print(f"    Track ID: {tracked['track_id']}, bbox: {tracked['bbox']}")
        
        # Get tracking summary
        summary = tracker.get_tracking_summary()
        print(f"\n  Tracking summary: {summary}")
        
    except Exception as e:
        print(f"  ✗ Face tracking test failed: {e}")
        return False
    
    return True

def test_face_recognition_service(config):
    """Test face recognition service."""
    print("\n" + "="*50)
    print("Testing Face Recognition Service")
    print("="*50)
    
    try:
        # Initialize recognition service
        recognizer = FaceRecognitionService(config)
        print(f"✓ Recognition service initialized with {recognizer.get_model_info()['model_type']} model")
        
        # Get database info
        db_info = recognizer.get_database_info()
        print(f"  Database info: {db_info}")
        
        # Test with sample images if available
        sample_dir = Path("data/frames")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg"))[:2]  # Test first 2 images
            
            for img_path in sample_images:
                print(f"\nTesting recognition on: {img_path.name}")
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  ✗ Failed to load image: {img_path}")
                    continue
                
                # Try to identify face
                student_id, confidence = recognizer.identify_face(image)
                if student_id:
                    print(f"  ✓ Identified as: {student_id} (confidence: {confidence:.3f})")
                else:
                    print(f"  ⚠ No match found (best confidence: {confidence:.3f})")
        
        else:
            print("  ⚠ No sample images found for recognition testing")
            
    except Exception as e:
        print(f"  ✗ Face recognition test failed: {e}")
        return False
    
    return True

def test_video_processor(config):
    """Test complete video processing pipeline."""
    print("\n" + "="*50)
    print("Testing Complete Video Processing Pipeline")
    print("="*50)
    
    try:
        # Initialize video processor
        processor = VideoProcessor(config)
        print("✓ Video processor initialized with all services")
        
        # Look for sample videos
        video_dir = Path("data/raw_videos")
        if video_dir.exists():
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
            
            if video_files:
                # Test with first video
                test_video = str(video_files[0])
                print(f"\nTesting pipeline with video: {test_video}")
                
                # Process video (with limited frames for testing)
                results = processor.process_video(
                    test_video,
                    output_dir="test_output",
                    save_annotated_video=True,
                    save_results=True
                )
                
                if 'error' not in results:
                    print("✓ Video processing completed successfully")
                    print(f"  Processing time: {results.get('processing_time', 0):.2f} seconds")
                    print(f"  Frames processed: {results['processing_stats']['frames_processed']}")
                    print(f"  Total detections: {results['processing_stats']['total_detections']}")
                    print(f"  Total tracks: {results['processing_stats']['total_tracks']}")
                    print(f"  Total recognitions: {results['processing_stats']['total_recognitions']}")
                    print(f"  Output directory: {results.get('output_directory', 'N/A')}")
                else:
                    print(f"  ✗ Video processing failed: {results['error']}")
            else:
                print("  ⚠ No video files found in data/raw_videos/")
        else:
            print("  ⚠ No video directory found")
            
    except Exception as e:
        print(f"  ✗ Video processing test failed: {e}")
        return False
    
    return True

def test_individual_services(config):
    """Test individual services separately."""
    print("\n" + "="*50)
    print("Testing Individual Services")
    print("="*50)
    
    # Test detection service
    detection_success = test_face_detection_service(config)
    
    # Test tracking service
    tracking_success = test_face_tracking_service(config)
    
    # Test recognition service
    recognition_success = test_face_recognition_service(config)
    
    return detection_success and tracking_success and recognition_success

def main():
    """Main test function."""
    print("FaceClass Face Pipeline Test Suite")
    print("="*50)
    
    # Setup logging
    setup_logging()
    
    try:
        # Load configuration
        config = Config()
        print("✓ Configuration loaded successfully")
        
        # Test individual services
        services_success = test_individual_services(config)
        
        # Test complete pipeline
        pipeline_success = test_video_processor(config)
        
        # Summary
        print("\n" + "="*50)
        print("Test Summary")
        print("="*50)
        
        if services_success:
            print("✓ Individual services: PASSED")
        else:
            print("✗ Individual services: FAILED")
        
        if pipeline_success:
            print("✓ Complete pipeline: PASSED")
        else:
            print("✗ Complete pipeline: FAILED")
        
        if services_success and pipeline_success:
            print("\n🎉 All tests passed! The face pipeline is working correctly.")
        else:
            print("\n⚠ Some tests failed. Check the logs for details.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        logging.error(f"Test suite error: {e}", exc_info=True)
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
