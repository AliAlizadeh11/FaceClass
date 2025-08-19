#!/usr/bin/env python3
"""
Simple YOLOv8 Test Script
Tests basic YOLOv8 functionality without full FaceClass dependencies.
"""

import sys
from pathlib import Path

def test_ultralytics_import():
    """Test if ultralytics can be imported."""
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        print("💡 Install with: pip install ultralytics")
        return False

def test_yolo_model_loading():
    """Test YOLO model loading."""
    try:
        from ultralytics import YOLO
        
        print("📥 Testing YOLOv8n model download...")
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8n model loaded successfully")
        
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_inference(model):
    """Test basic inference."""
    try:
        import numpy as np
        
        print("🔍 Testing inference...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(test_image, verbose=False)
        
        print(f"✅ Inference successful - {len(results)} results")
        
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            print(f"✅ Results structure correct - {len(results[0].boxes)} boxes")
            
            # Show first few detections
            for i, box in enumerate(results[0].boxes[:3]):
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                print(f"   Detection {i+1}: Class {cls}, Confidence {conf:.3f}")
        else:
            print("⚠️  Results structure incomplete")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def test_face_specific_model():
    """Test if face-specific model can be loaded."""
    try:
        from ultralytics import YOLO
        
        face_model_path = Path("models/face_detection/yolov8n-face.pt")
        
        if face_model_path.exists():
            print(f"📥 Testing face-specific model: {face_model_path}")
            model = YOLO(str(face_model_path))
            print("✅ Face-specific model loaded successfully")
            return model
        else:
            print("⚠️  Face-specific model not found")
            print("💡 Download with: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt")
            return None
            
    except Exception as e:
        print(f"❌ Face model loading failed: {e}")
        return None

def main():
    """Main test function."""
    print("🧪 Simple YOLOv8 Test")
    print("=" * 40)
    
    # Test 1: Import
    if not test_ultralytics_import():
        return
    
    # Test 2: Model loading
    model = test_yolo_model_loading()
    if not model:
        return
    
    # Test 3: Inference
    if not test_inference(model):
        return
    
    # Test 4: Face-specific model (optional)
    face_model = test_face_specific_model()
    
    print("\n📊 Test Summary")
    print("=" * 40)
    print("✅ Ultralytics: OK")
    print("✅ Model Loading: OK")
    print("✅ Inference: OK")
    print(f"✅ Face Model: {'OK' if face_model else 'Not Available'}")
    
    print("\n🎉 Basic YOLOv8 functionality is working!")
    print("\n💡 Next steps:")
    print("   1. Run the full debug script: python debug_yolov8_face_detection.py")
    print("   2. Test with real images")
    print("   3. Integrate with FaceClass")

if __name__ == "__main__":
    main()
