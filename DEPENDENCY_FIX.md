# Dependency Conflict Resolution Guide

## 🚨 **Issue: TensorFlow and RetinaFace Version Conflict**

### **Problem**
```
ERROR: Cannot install -r requirements.txt (line 20) and tensorflow>=2.6.0 because these package versions have conflicting dependencies.

The conflict is caused by:
    The user requested tensorflow>=2.6.0
    retinaface 1.1.1 depends on tensorflow==2.5.0
    The user requested tensorflow>=2.6.0
    retinaface 1.1.0 depends on tensorflow==2.5.0
```

## 🔧 **Solutions**

### **Solution 1: Use Fixed Requirements (Recommended)**

I've already updated the main `requirements.txt` file to fix the conflict:

```bash
# Install with the fixed requirements
pip install -r requirements.txt
```

**Changes made:**
- Changed `tensorflow>=2.6.0` to `tensorflow==2.5.0` for retinaface compatibility

### **Solution 2: Use Flexible Requirements (Alternative)**

If you still encounter issues, use the flexible requirements:

```bash
# Install with flexible requirements
pip install -r requirements-flexible.txt
```

**Features:**
- More flexible version constraints
- Retinaface commented out to avoid conflicts
- Compatible tensorflow version range

### **Solution 3: Use Minimal Requirements (Essential Only)**

For a minimal installation with only essential dependencies:

```bash
# Install with minimal requirements
pip install -r requirements-minimal.txt
```

**Features:**
- Only essential dependencies
- Minimal conflict potential
- Faster installation

## 🎯 **Step-by-Step Resolution**

### **Step 1: Clean Environment (Recommended)**
```bash
# Create a new virtual environment
python -m venv faceclass_env

# Activate the environment
# On Windows:
faceclass_env\Scripts\activate
# On macOS/Linux:
source faceclass_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### **Step 2: Install Dependencies**
```bash
# Option 1: Fixed requirements (recommended)
pip install -r requirements.txt

# Option 2: Flexible requirements (if conflicts persist)
pip install -r requirements-flexible.txt

# Option 3: Minimal requirements (essential only)
pip install -r requirements-minimal.txt
```

### **Step 3: Verify Installation**
```bash
# Test the installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import dash; print(f'Dash version: {dash.__version__}')"
```

## 🔍 **Alternative Solutions**

### **Option A: Manual Installation**
```bash
# Install core dependencies first
pip install numpy>=1.21.0 opencv-python>=4.5.0 Pillow>=8.3.0

# Install deep learning frameworks
pip install torch>=1.9.0 torchvision>=0.10.0
pip install tensorflow==2.5.0

# Install computer vision packages
pip install mediapipe>=0.8.0 face-recognition>=1.3.0

# Install web framework
pip install dash>=2.0.0 dash-bootstrap-components>=1.0.0 Flask>=2.0.0

# Install data processing
pip install pandas>=1.3.0 matplotlib>=3.4.0 plotly>=5.0.0
```

### **Option B: Conda Installation**
```bash
# Create conda environment
conda create -n faceclass python=3.8
conda activate faceclass

# Install packages with conda
conda install -c conda-forge tensorflow=2.5.0
conda install -c conda-forge pytorch torchvision
conda install -c conda-forge opencv pillow numpy pandas matplotlib

# Install remaining packages with pip
pip install dash dash-bootstrap-components Flask
pip install mediapipe face-recognition plotly
```

## 🎯 **Troubleshooting**

### **Common Issues and Solutions**

#### **Issue 1: Still getting tensorflow conflicts**
```bash
# Try installing tensorflow first
pip install tensorflow==2.5.0

# Then install other packages
pip install -r requirements.txt
```

#### **Issue 2: Retinaface installation fails**
```bash
# Skip retinaface for now (it's optional)
pip install -r requirements-flexible.txt

# Or install manually without retinaface
pip install tensorflow==2.5.0
pip install mediapipe face-recognition mtcnn
```

#### **Issue 3: CUDA compatibility issues**
```bash
# Install CPU-only tensorflow
pip install tensorflow-cpu==2.5.0

# Or use conda for better CUDA support
conda install -c conda-forge tensorflow-gpu=2.5.0
```

#### **Issue 4: Memory issues during installation**
```bash
# Install packages one by one
pip install numpy
pip install opencv-python
pip install tensorflow==2.5.0
pip install torch torchvision
# ... continue with other packages
```

## 📊 **Dependency Matrix**

| Package | Version | Compatibility | Notes |
|---------|---------|---------------|-------|
| tensorflow | 2.5.0 | Fixed | Required for retinaface |
| torch | >=1.9.0 | Flexible | Independent of tensorflow |
| opencv-python | >=4.5.0 | Flexible | No conflicts |
| mediapipe | >=0.8.0 | Flexible | No conflicts |
| face-recognition | >=1.3.0 | Flexible | No conflicts |
| dash | >=2.0.0 | Flexible | No conflicts |

## 🎉 **Success Verification**

After successful installation, you should be able to:

1. **Import all packages** without errors
2. **Run the dashboard** with `python src/main.py --mode dashboard`
3. **Access the website** at `http://localhost:8080`
4. **Upload and process videos** without issues

## 📞 **Support**

If you continue to experience issues:

1. **Check Python version** (recommended: 3.8-3.9)
2. **Verify virtual environment** is activated
3. **Try minimal requirements** first
4. **Check system compatibility** (OS, architecture)
5. **Review error logs** for specific issues

## 🔄 **Updates**

- **2024-01-XX**: Fixed tensorflow version conflict
- **2024-01-XX**: Added flexible and minimal requirements
- **2024-01-XX**: Created comprehensive troubleshooting guide 