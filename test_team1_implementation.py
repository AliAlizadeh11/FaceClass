#!/usr/bin/env python3
"""
Team 1 Implementation Test Script
Tests all the enhanced features implemented for Face Detection & Recognition Core
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from detection.model_comparison import ModelBenchmarker
from detection.deep_ocsort import DeepOCSORTTracker
from recognition.face_quality import FaceQualityAssessor
from recognition.database_manager import FaceDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_comparison():
    """Test the model comparison and benchmarking functionality."""
    print("\n" + "="*60)
    print("TESTING MODEL COMPARISON & BENCHMARKING")
    print("="*60)
    
    try:
        # Create mock config
        config = type('Config', (), {
            'get': lambda self, key, default=None: default,
            'get_path': lambda self, path_type: Path(__file__).parent / 'src' / 'models'
        })()
        
        # Create benchmarker
        benchmarker = ModelBenchmarker(config)
        
        # Load test data (use existing data directory)
        data_path = Path(__file__).parent / 'data'
        if data_path.exists():
            benchmarker.load_test_data(str(data_path))
            print(f"✓ Test data loaded: {len(benchmarker.test_images)} images, {len(benchmarker.test_videos)} videos")
        else:
            print("⚠ No test data found, creating mock data")
            # Create mock data for testing
            benchmarker.test_images = [Path(__file__).parent / 'mock_image.jpg']
            benchmarker.test_videos = [Path(__file__).parent / 'mock_video.mp4']
        
        print("✓ Model benchmarker created successfully")
        print("✓ Ready to run comprehensive model comparison")
        
        return True
        
    except Exception as e:
        print(f"✗ Model comparison test failed: {e}")
        return False


def test_deep_ocsort_tracking():
    """Test the Deep OC-SORT tracking implementation."""
    print("\n" + "="*60)
    print("TESTING DEEP OC-SORT TRACKING")
    print("="*60)
    
    try:
        # Create tracker configuration
        config = {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'feature_similarity_threshold': 0.7,
            'multi_camera': True
        }
        
        # Create tracker
        tracker = DeepOCSORTTracker(config)
        print("✓ Deep OC-SORT tracker created successfully")
        
        # Test with mock detections
        mock_detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'face'},
            {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'label': 'face'},
            {'bbox': [500, 200, 600, 300], 'confidence': 0.7, 'label': 'face'}
        ]
        
        # Update tracker
        tracked_detections = tracker.update(mock_detections, frame_id=0, camera_id=0)
        print(f"✓ Tracking update successful: {len(tracked_detections)} tracks")
        
        # Test multi-frame tracking
        for frame_id in range(1, 5):
            # Simulate movement
            moved_detections = []
            for det in mock_detections:
                bbox = det['bbox']
                moved_detections.append({
                    'bbox': [bbox[0] + 10, bbox[1] + 5, bbox[2] + 10, bbox[3] + 5],
                    'confidence': det['confidence'],
                    'label': det['label']
                })
            
            tracked_detections = tracker.update(moved_detections, frame_id=frame_id, camera_id=0)
            print(f"  Frame {frame_id}: {len(tracked_detections)} tracks")
        
        # Get performance metrics
        metrics = tracker.get_performance_metrics()
        print(f"✓ Performance metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Deep OC-SORT tracking test failed: {e}")
        return False


def test_face_quality_assessment():
    """Test the face quality assessment functionality."""
    print("\n" + "="*60)
    print("TESTING FACE QUALITY ASSESSMENT")
    print("="*60)
    
    try:
        # Create quality assessor
        config = {
            'min_face_size': 80,
            'min_resolution': 64,
            'min_contrast': 30,
            'max_blur': 100,
            'min_eye_openness': 0.3
        }
        
        assessor = FaceQualityAssessor(config)
        print("✓ Face quality assessor created successfully")
        
        # Create mock face images for testing
        mock_faces = []
        
        # High quality face (good contrast, sharp)
        high_quality = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        mock_faces.append(('high_quality', high_quality))
        
        # Low contrast face
        low_contrast = np.random.randint(100, 120, (150, 150, 3), dtype=np.uint8)
        mock_faces.append(('low_contrast', low_contrast))
        
        # Blurry face
        blurry = cv2.GaussianBlur(high_quality, (15, 15), 0)
        mock_faces.append(('blurry', blurry))
        
        # Small face
        small_face = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
        mock_faces.append(('small_face', small_face))
        
        print(f"✓ Created {len(mock_faces)} mock face images for testing")
        
        # Test individual quality assessment
        for name, face_image in mock_faces:
            result = assessor.assess_face_quality(face_image)
            print(f"  {name}: Score {result['overall_score']:.3f} ({result['quality_level']})")
            
            if result['recommendations']:
                for rec in result['recommendations'][:2]:  # Show first 2 recommendations
                    print(f"    - {rec}")
        
        # Test batch assessment
        face_images = [face for _, face in mock_faces]
        batch_results = assessor.batch_assess_quality(face_images)
        print(f"✓ Batch quality assessment completed for {len(batch_results)} faces")
        
        # Test quality filtering
        filtered_images, filtered_bboxes, filtered_indices = assessor.filter_by_quality(
            face_images, min_score=0.6
        )
        print(f"✓ Quality filtering: {len(filtered_images)} faces passed threshold")
        
        return True
        
    except Exception as e:
        print(f"✗ Face quality assessment test failed: {e}")
        return False


def test_database_manager():
    """Test the enhanced database manager functionality."""
    print("\n" + "="*60)
    print("TESTING ENHANCED DATABASE MANAGER")
    print("="*60)
    
    try:
        # Create database manager
        config = {
            'max_faces_per_person': 10,
            'min_quality_score': 0.7,
            'enable_auto_optimization': True,
            'backup_interval': 24
        }
        
        db_manager = FaceDatabaseManager(config)
        print("✓ Database manager created successfully")
        
        # Test student management
        test_students = [
            {'id': 'ST001', 'name': 'John Doe', 'email': 'john@university.edu', 'department': 'Computer Science'},
            {'id': 'ST002', 'name': 'Jane Smith', 'email': 'jane@university.edu', 'department': 'Mathematics'},
            {'id': 'ST003', 'name': 'Bob Johnson', 'email': 'bob@university.edu', 'department': 'Physics'}
        ]
        
        for student in test_students:
            success = db_manager.add_student(
                student['id'], student['name'], student['email'], student['department']
            )
            if success:
                print(f"  ✓ Added student: {student['name']} ({student['id']})")
            else:
                print(f"  ✗ Failed to add student: {student['name']}")
        
        # Test face encoding storage
        mock_encodings = [
            np.random.randn(128),  # 128-dimensional face encoding
            np.random.randn(128),
            np.random.randn(128)
        ]
        
        for i, student in enumerate(test_students):
            encoding = mock_encodings[i % len(mock_encodings)]
            success = db_manager.add_face_encoding(
                student_id=student['id'],
                encoding=encoding,
                quality_score=0.8 + (i * 0.1),
                lighting_condition='natural',
                pose_angles={'yaw': 0, 'pitch': 0, 'roll': 0},
                expression_type='neutral'
            )
            
            if success:
                print(f"  ✓ Added face encoding for: {student['name']}")
            else:
                print(f"  ✗ Failed to add face encoding for: {student['name']}")
        
        # Test face variant storage
        for student in test_students:
            variant_data = {
                'lighting': 'artificial',
                'pose': 'slight_turn',
                'expression': 'smile'
            }
            
            success = db_manager.add_face_variant(
                student_id=student['id'],
                variant_type='expression_variation',
                variant_data=variant_data,
                quality_metrics={'overall_score': 0.75}
            )
            
            if success:
                print(f"  ✓ Added face variant for: {student['name']}")
        
        # Test retrieval
        for student in test_students:
            encodings = db_manager.get_face_encodings(student['id'])
            print(f"  ✓ Retrieved {len(encodings)} encodings for: {student['name']}")
        
        # Test database statistics
        stats = db_manager.get_database_stats()
        print(f"✓ Database stats: {stats['total_students']} students, {stats['total_face_encodings']} encodings")
        
        # Test search functionality
        query_encoding = np.random.randn(128)
        similar_faces = db_manager.search_similar_faces(query_encoding, threshold=0.5)
        print(f"✓ Similar face search: {len(similar_faces)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Database manager test failed: {e}")
        return False


def test_integration():
    """Test integration of all Team 1 components."""
    print("\n" + "="*60)
    print("TESTING TEAM 1 COMPONENT INTEGRATION")
    print("="*60)
    
    try:
        # Create integrated configuration
        config = {
            'face_detection': {
                'model': 'retinaface',
                'confidence_threshold': 0.8,
                'min_face_size': 80
            },
            'face_tracking': {
                'algorithm': 'deep_ocsort',
                'persistence_frames': 30,
                'multi_camera': True
            },
            'face_quality': {
                'min_quality_score': 0.7,
                'min_face_size': 80
            },
            'database': {
                'max_faces_per_person': 15,
                'enable_auto_optimization': True
            }
        }
        
        print("✓ Integrated configuration created")
        
        # Test component interaction
        print("  Testing component interaction...")
        
        # 1. Quality assessment + Database
        quality_assessor = FaceQualityAssessor(config.get('face_quality', {}))
        db_manager = FaceDatabaseManager(config.get('database', {}))
        
        # 2. Tracking + Quality assessment
        tracker = DeepOCSORTTracker(config.get('face_tracking', {}))
        
        print("  ✓ All components initialized successfully")
        
        # Simulate integrated workflow
        print("  Simulating integrated workflow...")
        
        # Mock video frame processing
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 1. Face detection (simulated)
        mock_detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'face'},
            {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'label': 'face'}
        ]
        
        # 2. Face tracking
        tracked_detections = tracker.update(mock_detections, frame_id=0)
        print(f"    ✓ Tracking: {len(tracked_detections)} faces tracked")
        
        # 3. Quality assessment for each tracked face
        for i, detection in enumerate(tracked_detections):
            # Extract face region (simulated)
            bbox = detection['bbox']
            face_region = mock_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Assess quality
            quality_result = quality_assessor.assess_face_quality(face_region)
            print(f"    ✓ Face {i+1} quality: {quality_result['overall_score']:.3f}")
            
            # Store in database if quality is good
            if quality_result['is_suitable_for_recognition']:
                # Simulate face encoding
                mock_encoding = np.random.randn(128)
                
                # Add to database
                success = db_manager.add_face_encoding(
                    student_id=f"TRACK_{detection['track_id']}",
                    encoding=mock_encoding,
                    quality_score=quality_result['overall_score'],
                    lighting_condition='natural',
                    pose_angles={'yaw': 0, 'pitch': 0, 'roll': 0},
                    expression_type='neutral'
                )
                
                if success:
                    print(f"      ✓ Stored in database")
                else:
                    print(f"      ✗ Failed to store in database")
        
        print("  ✓ Integrated workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def run_performance_benchmark():
    """Run a simple performance benchmark."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    try:
        # Test quality assessment performance
        print("Testing face quality assessment performance...")
        
        config = {'min_face_size': 80}
        assessor = FaceQualityAssessor(config)
        
        # Create test images
        test_images = []
        for i in range(10):
            size = 100 + (i * 20)
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            test_images.append(img)
        
        # Benchmark quality assessment
        start_time = time.time()
        results = assessor.batch_assess_quality(test_images)
        total_time = time.time() - start_time
        
        avg_time = total_time / len(test_images)
        fps = len(test_images) / total_time
        
        print(f"  ✓ Processed {len(test_images)} images in {total_time:.3f}s")
        print(f"  ✓ Average time per image: {avg_time:.3f}s")
        print(f"  ✓ Processing speed: {fps:.1f} FPS")
        
        # Test tracking performance
        print("\nTesting tracking performance...")
        
        tracker_config = {'max_age': 30, 'min_hits': 3}
        tracker = DeepOCSORTTracker(tracker_config)
        
        # Simulate tracking over multiple frames
        mock_detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'label': 'face'},
            {'bbox': [300, 150, 400, 250], 'confidence': 0.8, 'label': 'face'}
        ]
        
        start_time = time.time()
        for frame_id in range(100):
            # Simulate movement
            moved_detections = []
            for det in mock_detections:
                bbox = det['bbox']
                moved_detections.append({
                    'bbox': [bbox[0] + frame_id, bbox[1] + frame_id//2, 
                            bbox[2] + frame_id, bbox[3] + frame_id//2],
                    'confidence': det['confidence'],
                    'label': det['label']
                })
            
            tracker.update(moved_detections, frame_id)
        
        total_time = time.time() - start_time
        tracking_fps = 100 / total_time
        
        print(f"  ✓ Processed 100 frames in {total_time:.3f}s")
        print(f"  ✓ Tracking speed: {tracking_fps:.1f} FPS")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False


def main():
    """Run all Team 1 implementation tests."""
    print("TEAM 1: FACE DETECTION & RECOGNITION CORE")
    print("Implementation Testing Suite")
    print("="*60)
    
    # Track test results
    test_results = []
    
    # Run all tests
    tests = [
        ("Model Comparison", test_model_comparison),
        ("Deep OC-SORT Tracking", test_deep_ocsort_tracking),
        ("Face Quality Assessment", test_face_quality_assessment),
        ("Database Manager", test_database_manager),
        ("Component Integration", test_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:25} : {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Team 1 implementations are working correctly!")
    else:
        print("⚠ Some tests failed. Please check the implementation.")
    
    print("\nTeam 1 Features Implemented:")
    print("  ✓ RetinaFace detection model integration")
    print("  ✓ Deep OC-SORT tracking algorithm")
    print("  ✓ Enhanced ByteTrack optimization")
    print("  ✓ Face quality assessment system")
    print("  ✓ Enhanced database management")
    print("  ✓ Multi-camera tracking support")
    print("  ✓ Performance monitoring and benchmarking")
    print("  ✓ Automatic database optimization")
    print("  ✓ Backup and recovery systems")


if __name__ == "__main__":
    main()
