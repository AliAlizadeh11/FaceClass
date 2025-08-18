#!/usr/bin/env python3
"""
Dependency installation script for FaceClass.
Handles package conflicts and provides installation options.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_core_dependencies():
    """Install core dependencies without conflicts."""
    print("\n📦 Installing core dependencies...")
    
    core_packages = [
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "mediapipe>=0.8.0",
        "mtcnn>=0.1.1",
        "ultralytics>=8.0.0"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")

def install_tensorflow_compatible():
    """Install TensorFlow-compatible version."""
    print("\n🤖 Installing TensorFlow-compatible dependencies...")
    
    tf_packages = [
        "tensorflow>=2.5.0,<2.6.0",
        "retinaface>=0.0.13",
        "deepface>=0.0.79"
    ]
    
    for package in tf_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")

def install_modern_dependencies():
    """Install modern dependencies without TensorFlow."""
    print("\n🚀 Installing modern dependencies...")
    
    modern_packages = [
        "insightface>=0.7.3",
        "facenet-pytorch>=0.5.3",
        "opencv-contrib-python>=4.5.0"
    ]
    
    for package in modern_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")

def install_optional_dependencies():
    """Install optional dependencies."""
    print("\n🔧 Installing optional dependencies...")
    
    optional_packages = [
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "Flask>=2.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0"
    ]
    
    for package in optional_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")

def test_imports():
    """Test if key packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("ultralytics", "Ultralytics"),
    ]
    
    all_imports_successful = True
    
    for module, name in test_packages:
        try:
            __import__(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            all_imports_successful = False
    
    return all_imports_successful

def main():
    """Main installation function."""
    print("🚀 FaceClass Dependency Installer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Ask user for installation preference
    print("\n📋 Installation Options:")
    print("1. Full installation (may have TensorFlow conflicts)")
    print("2. Modern installation (no TensorFlow conflicts)")
    print("3. Core only (minimal dependencies)")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Install based on choice
    if choice == '1':
        print("\n🔧 Installing full dependencies...")
        install_core_dependencies()
        install_tensorflow_compatible()
        install_modern_dependencies()
        install_optional_dependencies()
        
    elif choice == '2':
        print("\n🚀 Installing modern dependencies...")
        install_core_dependencies()
        install_modern_dependencies()
        install_optional_dependencies()
        
    else:  # choice == '3'
        print("\n📦 Installing core dependencies only...")
        install_core_dependencies()
    
    # Test imports
    if test_imports():
        print("\n🎉 Installation completed successfully!")
        print("\n📚 Next steps:")
        print("1. Run: python test_face_pipeline.py")
        print("2. Check the logs for any warnings")
        print("3. Refer to FACE_PIPELINE_README.md for usage")
    else:
        print("\n⚠️ Some packages failed to import.")
        print("Check the error messages above and try:")
        print("1. Upgrading pip: pip install --upgrade pip")
        print("2. Installing packages individually")
        print("3. Using a virtual environment")

if __name__ == "__main__":
    main()
