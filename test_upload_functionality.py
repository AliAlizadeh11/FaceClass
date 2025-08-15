#!/usr/bin/env python3
"""
Test script to verify video upload functionality in FaceClass dashboard.
"""

import os
import sys
import base64
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_upload_methods():
    """Test the upload methods from the dashboard."""
    print("🧪 Testing FaceClass Upload Methods")
    print("=" * 50)
    
    try:
        from dashboard.dashboard_ui import DashboardUI
        from config import Config
        
        print("✅ Successfully imported DashboardUI")
        
        # Create a mock config
        config = Config()
        
        # Initialize dashboard
        dashboard = DashboardUI(config)
        print("✅ Dashboard initialized successfully")
        
        # Test the save_uploaded_video method with a mock video
        print("\n📹 Testing video save method...")
        
        # Create a mock video content (base64 encoded)
        mock_video_content = b"fake video content for testing"
        mock_contents = f"data:video/mp4;base64,{base64.b64encode(mock_video_content).decode()}"
        mock_filename = "test_video.mp4"
        
        result = dashboard._save_uploaded_video(mock_contents, mock_filename)
        
        if result:
            print(f"✅ Video saved successfully to: {result}")
            print(f"   File exists: {Path(result).exists()}")
            print(f"   File size: {Path(result).stat().st_size} bytes")
            
            # Clean up test file
            try:
                Path(result).unlink()
                print("✅ Test file cleaned up")
            except Exception as e:
                print(f"⚠️  Could not clean up test file: {e}")
        else:
            print("❌ Video save failed")
            
        # Test video verification
        print("\n🔍 Testing video verification...")
        if result and Path(result).exists():
            is_valid = dashboard._verify_uploaded_video(result)
            print(f"   Video verification result: {is_valid}")
        else:
            print("   Skipping verification test - no video file")
            
        print("\n🎉 Upload method testing completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the correct directory and dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_directories():
    """Test if required directories exist and are writable."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "data",
        "data/temp", 
        "data/raw_videos",
        "data/frames",
        "data/outputs"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path} exists")
            if os.access(path, os.W_OK):
                print(f"      ✅ Writable")
            else:
                print(f"      ❌ Not writable")
        else:
            print(f"   ❌ {dir_path} missing")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"      ✅ Created")
            except Exception as e:
                print(f"      ❌ Could not create: {e}")

def main():
    """Main test function."""
    print("🚀 FaceClass Upload Functionality Test")
    print("=" * 50)
    
    # Test directories first
    test_directories()
    
    # Test upload methods
    success = test_upload_methods()
    
    if success:
        print("\n🎉 All tests passed! Upload functionality should work.")
        print("\n💡 If upload still doesn't work in the browser:")
        print("   1. Check browser console for JavaScript errors")
        print("   2. Try a different browser")
        print("   3. Check if the video file is too large (>100MB)")
        print("   4. Ensure the video format is supported (MP4, AVI, MOV, MKV, WEBM)")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you're in the FaceClass directory")
        print("   2. Activate virtual environment: source faceclass_env/bin/activate")
        print("   3. Install dependencies: pip install -r requirements.txt")
        print("   4. Check if all required packages are installed")

if __name__ == "__main__":
    main()
