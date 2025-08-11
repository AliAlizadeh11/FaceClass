#!/usr/bin/env python3
"""
Test script to verify FaceClass installation works with updated requirements.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🧪 Testing FaceClass Installation")
    print("=" * 40)
    
    # Test core packages
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import dash
        print(f"✅ Dash {dash.__version__}")
    except ImportError as e:
        print(f"❌ Dash import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print(f"✅ MediaPipe {mp.__version__}")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    try:
        import face_recognition
        print(f"✅ Face Recognition {face_recognition.__version__}")
    except ImportError as e:
        print(f"❌ Face Recognition import failed: {e}")
        return False
    
    return True

def test_face_detection():
    """Test face detection functionality."""
    print("\n🔍 Testing Face Detection...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from detection.face_tracker import FaceTracker
        from config import Config
        
        config = Config()
        face_tracker = FaceTracker(config)
        print("✅ Face detection initialized successfully")
        
        # Test with a simple image
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = face_tracker.detect_faces(test_image)
        print(f"✅ Face detection test completed (found {len(detections)} faces)")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_dashboard():
    """Test dashboard initialization."""
    print("\n🎨 Testing Dashboard...")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from dashboard.dashboard_ui import DashboardUI
        from config import Config
        
        config = Config()
        dashboard = DashboardUI(config)
        print("✅ Dashboard initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 FaceClass Installation Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return 1
    
    # Test face detection
    if not test_face_detection():
        print("\n⚠️ Face detection test failed, but core functionality may still work.")
    
    # Test dashboard
    if not test_dashboard():
        print("\n⚠️ Dashboard test failed, but core functionality may still work.")
    
    print("\n" + "=" * 40)
    print("🎉 Installation test completed!")
    print("\n📋 Next steps:")
    print("1. Run the dashboard: python src/main.py --mode dashboard")
    print("2. Open your browser to: http://localhost:8080")
    print("3. Upload a video to test the system")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 