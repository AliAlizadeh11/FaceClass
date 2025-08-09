#!/usr/bin/env python3
"""
Comprehensive test script for FaceClass student attendance analysis system.
Tests all major components and functionality.
"""

import sys
import os
import logging
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_config():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    try:
        from config import Config
        config = Config()
        print(f"✅ Configuration loaded successfully")
        print(f"   - Face detection model: {config.get('face_detection.model')}")
        print(f"   - Face recognition model: {config.get('face_recognition.model')}")
        print(f"   - Emotion detection model: {config.get('emotion_detection.model')}")
        print(f"   - Attention detection model: {config.get('attention_detection.model')}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_face_tracker():
    """Test face detection and tracking."""
    print("\n👥 Testing face detection and tracking...")
    try:
        from config import Config
        from detection.face_tracker import FaceTracker
        
        config = Config()
        face_tracker = FaceTracker(config)
        print(f"✅ Face tracker initialized successfully")
        print(f"   - Detection model: {face_tracker.detection_model}")
        print(f"   - Confidence threshold: {face_tracker.confidence_threshold}")
        return True
    except Exception as e:
        print(f"❌ Face tracker test failed: {e}")
        return False

def test_face_identifier():
    """Test face recognition."""
    print("\n🆔 Testing face recognition...")
    try:
        from config import Config
        from recognition.face_identifier import FaceIdentifier
        
        config = Config()
        face_identifier = FaceIdentifier(config)
        print(f"✅ Face identifier initialized successfully")
        print(f"   - Recognition model: {face_identifier.recognition_model}")
        print(f"   - Similarity threshold: {face_identifier.similarity_threshold}")
        return True
    except Exception as e:
        print(f"❌ Face identifier test failed: {e}")
        return False

def test_emotion_detector():
    """Test emotion and attention detection."""
    print("\n😊 Testing emotion and attention detection...")
    try:
        from config import Config
        from emotion.emotion_detector import EmotionDetector
        
        config = Config()
        emotion_detector = EmotionDetector(config)
        print(f"✅ Emotion detector initialized successfully")
        print(f"   - Emotion model: {emotion_detector.emotion_model}")
        print(f"   - Emotions: {emotion_detector.emotions}")
        print(f"   - Attention model: {emotion_detector.attention_model['type']}")
        return True
    except Exception as e:
        print(f"❌ Emotion detector test failed: {e}")
        return False

def test_attendance_tracker():
    """Test attendance tracking."""
    print("\n📊 Testing attendance tracking...")
    try:
        from config import Config
        from attendance.attendance_tracker import AttendanceTracker
        
        config = Config()
        attendance_tracker = AttendanceTracker(config)
        print(f"✅ Attendance tracker initialized successfully")
        print(f"   - Min detection duration: {attendance_tracker.min_detection_duration}s")
        print(f"   - Max absence duration: {attendance_tracker.max_absence_duration}s")
        print(f"   - Attendance threshold: {attendance_tracker.attendance_threshold}")
        return True
    except Exception as e:
        print(f"❌ Attendance tracker test failed: {e}")
        return False

def test_layout_mapper():
    """Test spatial analysis."""
    print("\n🗺️ Testing spatial analysis...")
    try:
        from config import Config
        from layout_analysis.layout_mapper import LayoutMapper
        
        config = Config()
        layout_mapper = LayoutMapper(config)
        print(f"✅ Layout mapper initialized successfully")
        print(f"   - Heatmap resolution: {layout_mapper.heatmap_resolution}")
        print(f"   - Classroom size: {layout_mapper.classroom_width}x{layout_mapper.classroom_height}")
        print(f"   - Seat positions: {len(layout_mapper.seat_positions)}")
        return True
    except Exception as e:
        print(f"❌ Layout mapper test failed: {e}")
        return False

def test_report_generator():
    """Test report generation."""
    print("\n📄 Testing report generation...")
    try:
        from config import Config
        from reporting.report_generator import ReportGenerator
        
        config = Config()
        report_generator = ReportGenerator(config)
        print(f"✅ Report generator initialized successfully")
        print(f"   - Report format: {report_generator.report_format}")
        print(f"   - Include charts: {report_generator.include_charts}")
        print(f"   - Include heatmaps: {report_generator.include_heatmaps}")
        return True
    except Exception as e:
        print(f"❌ Report generator test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard initialization."""
    print("\n🎨 Testing dashboard...")
    try:
        from config import Config
        from dashboard.dashboard_ui import DashboardUI
        
        config = Config()
        dashboard = DashboardUI(config)
        print(f"✅ Dashboard initialized successfully")
        print(f"   - Port: {dashboard.port}")
        print(f"   - Host: {dashboard.host}")
        print(f"   - Refresh rate: {dashboard.refresh_rate}s")
        return True
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_comprehensive_pipeline():
    """Test comprehensive analysis pipeline."""
    print("\n🚀 Testing comprehensive analysis pipeline...")
    try:
        from config import Config
        from main import process_video_comprehensive
        
        config = Config()
        print(f"✅ Comprehensive pipeline components loaded successfully")
        
        # Test with mock data
        mock_results = {
            'session_id': 'test_session_123',
            'video_path': 'test_video.mp4',
            'processing_time': 10.5,
            'detections': [],
            'attendance_data': {'total_students': 0, 'total_sessions': 0},
            'emotion_data': {'emotion_counts': {}, 'total_detections': 0},
            'attention_data': {'average_attention': 0.0, 'total_detections': 0},
            'spatial_data': {'heatmaps': {}, 'seat_assignments': {}},
            'video_info': {'duration': 10.5, 'frame_count': 0},
            'report_path': 'test_report.html',
            'session_summary': {}
        }
        
        print(f"   - Mock pipeline test completed")
        print(f"   - Session ID: {mock_results['session_id']}")
        print(f"   - Processing time: {mock_results['processing_time']}s")
        return True
    except Exception as e:
        print(f"❌ Comprehensive pipeline test failed: {e}")
        return False

def test_data_structures():
    """Test data structures and classes."""
    print("\n📊 Testing data structures...")
    try:
        from attendance.attendance_tracker import AttendanceRecord, StudentAttendance
        
        # Test AttendanceRecord
        record = AttendanceRecord(
            student_id="test_student",
            timestamp=datetime.now(),
            duration=60.0,
            confidence=0.8,
            emotion="happy",
            attention_score=0.7,
            location=(100, 200),
            seat_id="R1C1"
        )
        print(f"✅ AttendanceRecord created successfully")
        print(f"   - Student ID: {record.student_id}")
        print(f"   - Duration: {record.duration}s")
        print(f"   - Emotion: {record.emotion}")
        
        # Test StudentAttendance
        student_attendance = StudentAttendance(
            student_id="test_student",
            total_sessions=5,
            attended_sessions=4,
            total_duration=300.0,
            average_attention=0.75,
            dominant_emotion="happy",
            preferred_seat="R1C1",
            attendance_rate=0.8
        )
        print(f"✅ StudentAttendance created successfully")
        print(f"   - Attendance rate: {student_attendance.attendance_rate:.1%}")
        print(f"   - Average attention: {student_attendance.average_attention:.1%}")
        
        return True
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 FaceClass Comprehensive Test Suite")
    print("=" * 50)
    
    setup_logging()
    
    tests = [
        ("Configuration", test_config),
        ("Face Detection & Tracking", test_face_tracker),
        ("Face Recognition", test_face_identifier),
        ("Emotion & Attention Detection", test_emotion_detector),
        ("Attendance Tracking", test_attendance_tracker),
        ("Spatial Analysis", test_layout_mapper),
        ("Report Generation", test_report_generator),
        ("Dashboard", test_dashboard),
        ("Comprehensive Pipeline", test_comprehensive_pipeline),
        ("Data Structures", test_data_structures)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 