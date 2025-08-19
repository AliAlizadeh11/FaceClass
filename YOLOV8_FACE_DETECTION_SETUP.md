# YOLOv8 Face Detection Setup Guide

## 🎯 Overview

This guide provides step-by-step instructions for setting up YOLOv8 face detection in the FaceClass project. YOLOv8 offers high-speed, accurate face detection suitable for classroom scenarios.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **RAM**: 4GB+ (8GB+ recommended for optimal performance)
- **GPU**: Optional but recommended (CUDA 11.0+ for PyTorch acceleration)
- **Storage**: 2GB+ free space for models

### Operating System Support
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+)
- ✅ **Windows** (10, 11)
- ✅ **macOS** (10.15+)

## 🔧 Installation Steps

### Step 1: Install Core Dependencies

```bash
# Install PyTorch (CPU version - lighter)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics (YOLOv8 framework)
pip install ultralytics

# Install OpenCV and other required packages
pip install opencv-python numpy pillow
```

### Step 2: Install GPU Support (Optional)

```bash
# For CUDA 11.8+ (check your CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Installation

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test Ultralytics
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## 🎭 Model Selection

### Option 1: Face-Specific YOLOv8 Model (Recommended)

```bash
# Download face-specific model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt

# Move to models directory
mkdir -p models/face_detection
mv yolov8n-face.pt models/face_detection/
```

### Option 2: General YOLOv8 Model

```bash
# The system will automatically download yolov8n.pt on first use
# This model detects people (class 0) which can be used for face detection
```

### Option 3: Custom Trained Model

```bash
# Place your custom .pt file in models/face_detection/
# The system will automatically detect and use it
```

## ⚙️ Configuration

### Update config.yaml

```yaml
face_detection:
  model: "yolo"  # Use YOLOv8
  confidence_threshold: 0.3  # Lower for better recall
  nms_threshold: 0.3  # Lower for better overlap handling
  min_face_size: 15  # Minimum face size in pixels
  max_faces: 100  # Maximum faces to detect
  
  # YOLOv8 specific settings
  yolo:
    model_path: "models/face_detection/yolov8n-face.pt"
    device: "auto"  # auto, cpu, 0, 1, 2... (GPU device)
    half: false  # Use FP16 (faster but less accurate)
    verbose: false  # Reduce output verbosity
```

### Environment Variables (Optional)

```bash
# Set CUDA device (if using GPU)
export CUDA_VISIBLE_DEVICES=0

# Set PyTorch threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 🧪 Testing & Validation

### Run Debug Script

```bash
# Use the provided debug script
python debug_yolov8_face_detection.py
```

### Manual Testing

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load model
model = YOLO('models/face_detection/yolov8n-face.pt')

# Test image
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Run inference
results = model(test_image, verbose=False)

# Check results
print(f"Detections: {len(results)}")
if results and hasattr(results[0], 'boxes'):
    print(f"Boxes: {len(results[0].boxes)}")
```

## 🚀 Integration with FaceClass

### Basic Usage

```python
from services.enhanced_face_detection import EnhancedFaceDetectionService

# Initialize service
config = {
    'face_detection': {
        'model': 'yolo',
        'ensemble_models': ['yolo'],
        'confidence_threshold': 0.3
    },
    'paths': {
        'models': 'models'
    }
}

service = EnhancedFaceDetectionService(config)

# Detect faces
import cv2
image = cv2.imread('test_image.jpg')
detections = service.detect_faces_enhanced(image)

print(f"Faces detected: {len(detections)}")
```

### Advanced Configuration

```python
# Ensemble detection with multiple models
config = {
    'face_detection': {
        'model': 'ensemble',
        'ensemble_models': ['yolo', 'mediapipe', 'mtcnn'],
        'ensemble_voting': True,
        'ensemble_confidence_threshold': 0.2
    }
}
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'ultralytics'"
```bash
# Solution: Install ultralytics
pip install ultralytics

# Verify installation
python -c "import ultralytics; print('OK')"
```

#### Issue 2: "CUDA out of memory"
```bash
# Solution: Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage

# Or reduce model size
# Use yolov8n.pt instead of yolov8l.pt or yolov8x.pt
```

#### Issue 3: "Model file not found"
```bash
# Solution: Check model path
ls -la models/face_detection/

# Download model if missing
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt
mv yolov8n-face.pt models/face_detection/
```

#### Issue 4: "No detections returned"
```python
# Solution: Check confidence threshold
config['face_detection']['confidence_threshold'] = 0.1  # Lower threshold

# Check input image format
# Ensure image is numpy array with shape (H, W, 3) in BGR format
```

#### Issue 5: "Slow performance"
```bash
# Solution: Enable GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Use smaller model
# yolov8n.pt is faster than yolov8l.pt or yolov8x.pt

# Reduce image size
# Resize input images to 640x640 or smaller
```

## 📊 Performance Optimization

### Model Selection Guide

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.pt | 6MB | Fast | Good | Real-time detection |
| yolov8n-face.pt | 6MB | Fast | Excellent | Face-specific detection |
| yolov8s.pt | 22MB | Medium | Better | Balanced performance |
| yolov8l.pt | 87MB | Slow | Best | High accuracy required |

### Inference Optimization

```python
# Enable half-precision (FP16) for faster inference
config['face_detection']['yolo']['half'] = True

# Use specific device
config['face_detection']['yolo']['device'] = 0  # GPU 0

# Batch processing for multiple images
images = [img1, img2, img3]
results = model(images, verbose=False)
```

## 📁 File Structure

```
FaceClass/
├── models/
│   └── face_detection/
│       ├── yolov8n-face.pt    # Face-specific model
│       └── yolov8n.pt         # General model (fallback)
├── src/
│   └── services/
│       └── enhanced_face_detection.py  # Main service
├── debug_yolov8_face_detection.py      # Debug script
└── YOLOV8_FACE_DETECTION_SETUP.md      # This guide
```

## 🧪 Testing Checklist

- [ ] Dependencies installed (ultralytics, torch, opencv-python)
- [ ] Model files downloaded to models/face_detection/
- [ ] Configuration updated in config.yaml
- [ ] Debug script runs successfully
- [ ] Basic inference works with test images
- [ ] Integration with FaceClass service works
- [ ] Performance meets requirements (30+ FPS)

## 📚 Additional Resources

### Documentation
- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Model Sources
- [Ultralytics Assets](https://github.com/ultralytics/assets/releases)
- [Hugging Face Models](https://huggingface.co/ultralytics)
- [Custom Training Guide](https://docs.ultralytics.com/guides/training/)

### Community Support
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [GitHub Issues](https://github.com/ultralytics/ultralytics/issues)

## 🎉 Success Indicators

Your YOLOv8 face detection is working correctly when:

1. ✅ **Debug script runs without errors**
2. ✅ **Model loads successfully**
3. ✅ **Inference returns detections**
4. ✅ **Integration with FaceClass works**
5. ✅ **Performance meets requirements (30+ FPS)**
6. ✅ **Face detection accuracy >95%**

## 🚨 Emergency Fallback

If YOLOv8 fails, the system automatically falls back to:
1. **MediaPipe** - Fast, reliable face detection
2. **MTCNN** - High-accuracy face detection
3. **OpenCV Cascade** - Lightweight, always available

This ensures your face detection system remains functional even if YOLOv8 encounters issues.

---

**Need Help?** Run the debug script first: `python debug_yolov8_face_detection.py`
