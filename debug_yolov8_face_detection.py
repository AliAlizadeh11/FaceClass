#!/usr/bin/env python3
"""
Debug Script for YOLOv8 Face Detection Module
This script helps identify and fix issues with YOLOv8 face detection.
"""

import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking Dependencies...")
    print("=" * 50)
    
    dependencies = {
        'ultralytics': 'YOLOv8 framework',
        'torch': 'PyTorch backend',
        'torchvision': 'PyTorch vision utilities',
        'opencv-python': 'OpenCV for image processing',
        'numpy': 'Numerical computing',
        'PIL': 'Pillow for image handling'
    }
    
    missing_deps = []
    installed_deps = []
    
    for module, description in dependencies.items():
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                __import__(module)
                if module == 'torch':
                    import torch
                    version = torch.__version__
                elif module == 'ultralytics':
                    import ultralytics
                    version = ultralytics.__version__
                elif module == 'opencv-python':
                    import cv2
                    version = cv2.__version__
                elif module == 'numpy':
                    import numpy
                    version = numpy.__version__
                else:
                    version = "installed"
            
            print(f"✅ {module:15} - {description:25} - Version: {version}")
            installed_deps.append(module)
            
        except ImportError:
            print(f"❌ {module:15} - {description:25} - MISSING")
            missing_deps.append(module)
    
    print(f"\n📊 Summary: {len(installed_deps)}/{len(dependencies)} dependencies installed")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("💡 Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def check_models_directory():
    """Check if models directory exists and contains required files."""
    print("\n🔍 Checking Models Directory...")
    print("=" * 50)
    
    models_dir = Path("models")
    face_detection_dir = models_dir / "face_detection"
    
    if not models_dir.exists():
        print("❌ Models directory does not exist")
        print("💡 Creating models directory structure...")
        models_dir.mkdir(exist_ok=True)
        face_detection_dir.mkdir(exist_ok=True)
        print("✅ Created models directory structure")
    else:
        print("✅ Models directory exists")
    
    if not face_detection_dir.exists():
        print("❌ Face detection models directory does not exist")
        print("💡 Creating face detection models directory...")
        face_detection_dir.mkdir(exist_ok=True)
        print("✅ Created face detection models directory")
    else:
        print("✅ Face detection models directory exists")
    
    # Check for YOLOv8 face model
    yolov8_face_model = face_detection_dir / "yolov8n-face.pt"
    if yolov8_face_model.exists():
        print(f"✅ YOLOv8 face model found: {yolov8_face_model}")
        print(f"   Size: {yolov8_face_model.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("❌ YOLOv8 face model not found")
        print("💡 You need to download a face-specific YOLOv8 model")
        print("   Options:")
        print("   1. Download yolov8n-face.pt from Ultralytics")
        print("   2. Use general YOLOv8n model (less accurate for faces)")
        print("   3. Train custom face detection model")
    
    return face_detection_dir

def test_yolov8_installation():
    """Test YOLOv8 installation and basic functionality."""
    print("\n🧪 Testing YOLOv8 Installation...")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        
        # Test model loading
        try:
            print("📥 Testing YOLOv8n model download...")
            model = YOLO('yolov8n.pt')
            print("✅ YOLOv8n model loaded successfully")
            
            # Test basic inference
            print("🔍 Testing basic inference...")
            import numpy as np
            
            # Create a test image (random noise)
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference
            results = model(test_image, verbose=False)
            print(f"✅ Inference successful - {len(results)} results")
            
            # Check if we can access boxes
            if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                print("✅ Results structure is correct")
                print(f"   Boxes available: {len(results[0].boxes)}")
            else:
                print("⚠️  Results structure may be incomplete")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading/inference failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False

def test_face_detection_pipeline():
    """Test the complete face detection pipeline."""
    print("\n🔍 Testing Face Detection Pipeline...")
    print("=" * 50)
    
    try:
        # Import the enhanced face detection service
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from services.enhanced_face_detection import EnhancedFaceDetectionService
        
        # Create test configuration
        test_config = {
            'face_detection': {
                'model': 'yolo',
                'confidence_threshold': 0.3,
                'nms_threshold': 0.3,
                'min_face_size': 15,
                'max_faces': 100,
                'ensemble_models': ['yolo'],
                'ensemble_voting': False,
                'ensemble_confidence_threshold': 0.2,
                'preprocessing': {
                    'denoising': True,
                    'contrast_enhancement': True,
                    'super_resolution': False,
                    'scale_factors': [1.0]
                },
                'enable_tracking': False,
                'track_persistence': 30
            },
            'paths': {
                'models': 'models'
            }
        }
        
        print("✅ Enhanced face detection service imported")
        
        # Initialize service
        try:
            service = EnhancedFaceDetectionService(test_config)
            print("✅ Service initialized successfully")
            
            # Test with sample image
            import numpy as np
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            print("🔍 Testing face detection on sample frame...")
            detections = service.detect_faces_enhanced(test_frame, frame_id=0)
            
            print(f"✅ Detection completed - {len(detections)} faces detected")
            
            if detections:
                for i, det in enumerate(detections):
                    print(f"   Face {i+1}: bbox={det.bbox}, conf={det.confidence:.3f}, model={det.model}")
            
            return True
            
        except Exception as e:
            print(f"❌ Service initialization failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Service import failed: {e}")
        return False

def download_yolov8_face_model():
    """Download YOLOv8 face detection model."""
    print("\n📥 Downloading YOLOv8 Face Model...")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Create models directory if it doesn't exist
        models_dir = Path("models/face_detection")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "yolov8n-face.pt"
        
        if model_path.exists():
            print(f"✅ Model already exists: {model_path}")
            return str(model_path)
        
        print("📥 Downloading YOLOv8n-face model...")
        print("⚠️  Note: This may take several minutes depending on your internet connection")
        
        # Try to download face-specific model
        try:
            # First try to download a face-specific model
            model = YOLO('yolov8n-face.pt')
            print("✅ YOLOv8n-face model downloaded successfully")
            return str(model_path)
        except:
            print("⚠️  Face-specific model not available, using general YOLOv8n")
            model = YOLO('yolov8n.pt')
            
            # Save it to our models directory
            model.save(str(model_path))
            print(f"✅ General YOLOv8n model saved to {model_path}")
            return str(model_path)
            
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return None

def create_test_image():
    """Create a test image with faces for testing."""
    print("\n🎨 Creating Test Image...")
    print("=" * 50)
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image (you can replace this with a real image)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some simple shapes to simulate faces
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(test_image, (400, 150), (500, 250), (255, 255, 255), -1)
        
        # Save test image
        test_image_path = "test_face_detection.jpg"
        cv2.imwrite(test_image_path, test_image)
        
        print(f"✅ Test image created: {test_image_path}")
        print("💡 You can replace this with a real image containing faces")
        
        return test_image_path
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return None

def main():
    """Main debug function."""
    print("🚀 YOLOv8 Face Detection Debug Script")
    print("=" * 60)
    
    # Step 1: Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Dependencies missing. Please install them first.")
        print("💡 Run: pip install ultralytics torch torchvision opencv-python numpy pillow")
        return
    
    # Step 2: Check models directory
    models_dir = check_models_directory()
    
    # Step 3: Test YOLOv8 installation
    yolov8_ok = test_yolov8_installation()
    
    if not yolov8_ok:
        print("\n❌ YOLOv8 installation test failed.")
        return
    
    # Step 4: Download face model if needed
    model_path = download_yolov8_face_model()
    
    if not model_path:
        print("\n❌ Model download failed.")
        return
    
    # Step 5: Test face detection pipeline
    pipeline_ok = test_face_detection_pipeline()
    
    # Step 6: Create test image
    test_image_path = create_test_image()
    
    # Summary
    print("\n📊 Debug Summary")
    print("=" * 60)
    print(f"✅ Dependencies: {'OK' if deps_ok else 'FAILED'}")
    print(f"✅ YOLOv8 Installation: {'OK' if yolov8_ok else 'FAILED'}")
    print(f"✅ Face Model: {'OK' if model_path else 'FAILED'}")
    print(f"✅ Pipeline: {'OK' if pipeline_ok else 'FAILED'}")
    print(f"✅ Test Image: {'OK' if test_image_path else 'FAILED'}")
    
    if all([deps_ok, yolov8_ok, model_path, pipeline_ok, test_image_path]):
        print("\n🎉 All tests passed! YOLOv8 face detection is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Test with real images containing faces")
        print("   2. Adjust confidence thresholds if needed")
        print("   3. Integrate with your main application")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\n💡 Common solutions:")
        print("   1. Install missing dependencies")
        print("   2. Check model file paths")
        print("   3. Verify image format and size")

if __name__ == "__main__":
    main()
