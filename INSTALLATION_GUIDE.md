# FaceClass Installation Guide

## 🚨 Dependency Conflict Resolution

The main issue is that your system has newer TensorFlow versions (2.16.0+) but some packages require older versions that are no longer available.

## 🔧 Step-by-Step Solution

### Step 1: Clean Environment (Recommended)
```bash
# Create a new virtual environment
python -m venv faceclass_env

# Activate it
# On Linux/Mac:
source faceclass_env/bin/activate
# On Windows:
faceclass_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Minimal Dependencies (Recommended First)
```bash
pip install -r requirements_minimal.txt
```

This will install only the essential packages that are guaranteed to work together.

### Step 3: Test Basic Functionality
```bash
python -c "
import numpy as np
import cv2
import torch
import ultralytics
print('✅ All basic packages imported successfully!')
"
```

### Step 4: Install Additional Packages (Optional)
If the minimal installation works, you can add more packages:

```bash
# Install additional useful packages
pip install Pillow scipy matplotlib pandas

# Install face detection alternatives
pip install mtcnn mediapipe

# Install face recognition alternatives
pip install insightface facenet-pytorch
```

## 📋 Alternative Installation Methods

### Method 1: Use the Interactive Installer
```bash
python install_dependencies.py
```
Choose option 2 (Modern installation) to avoid TensorFlow conflicts.

### Method 2: Manual Package Installation
```bash
# Core packages
pip install numpy opencv-python torch torchvision ultralytics

# Test each package individually
python -c "import numpy; print('NumPy OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import ultralytics; print('Ultralytics OK')"
```

### Method 3: Use Conda (Alternative Package Manager)
```bash
# Create conda environment
conda create -n faceclass python=3.9
conda activate faceclass

# Install packages
conda install numpy opencv pytorch torchvision -c pytorch
pip install ultralytics
```

## 🧪 Testing the Installation

### Test 1: Basic Imports
```bash
python -c "
try:
    import numpy as np
    import cv2
    import torch
    import ultralytics
    print('🎉 All packages imported successfully!')
    print(f'NumPy version: {np.__version__}')
    print(f'OpenCV version: {cv2.__version__}')
    print(f'PyTorch version: {torch.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

### Test 2: Test the Pipeline
```bash
python test_face_pipeline.py
```

## 🚫 What NOT to Install

**Avoid these packages due to conflicts:**
- `tensorflow` (version conflicts)
- `retinaface` (requires old TensorFlow)
- `deepface` (requires old TensorFlow)

## ✅ What WILL Work

**These packages work together:**
- `numpy` - Numerical computing
- `opencv-python` - Computer vision
- `torch` + `torchvision` - Deep learning
- `ultralytics` - YOLO models
- `mtcnn` - Face detection
- `insightface` - Face recognition
- `facenet-pytorch` - Face recognition

## 🔍 Troubleshooting

### Error: "No matching distribution found for tensorflow<2.6.0"
**Solution**: Don't install TensorFlow. Use PyTorch-based alternatives.

### Error: "Could not find a version that satisfies the requirement"
**Solution**: Use the minimal requirements file or install packages individually.

### Error: "ImportError: No module named 'cv2'"
**Solution**: Install OpenCV: `pip install opencv-python`

### Error: "CUDA not available" (PyTorch)
**Solution**: This is normal on CPU-only systems. PyTorch will use CPU.

## 📚 Next Steps After Installation

1. **Test the pipeline**:
   ```bash
   python test_face_pipeline.py
   ```

2. **Check the logs** for any warnings or errors

3. **Refer to documentation**:
   - `FACE_PIPELINE_README.md` - Complete pipeline documentation
   - `src/services/` - Service implementations

4. **Process a test video**:
   ```python
   from services.video_processor import VideoProcessor
   from config import Config
   
   config = Config()
   processor = VideoProcessor(config)
   results = processor.process_video("test_video.mp4")
   ```

## 🆘 Still Having Issues?

1. **Check Python version**: Ensure you're using Python 3.8+
2. **Use virtual environment**: Isolate dependencies
3. **Start minimal**: Install only essential packages first
4. **Check system requirements**: Ensure you have enough disk space and RAM
5. **Use the installer script**: `python install_dependencies.py`

## 📞 Support

If you continue to have issues:
1. Check the error messages carefully
2. Try the minimal installation first
3. Use the interactive installer
4. Check the logs in `faceclass.log`
