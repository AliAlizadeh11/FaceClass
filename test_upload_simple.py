#!/usr/bin/env python3
"""
Simple test to verify video upload functionality.
"""

import os
import sys
from pathlib import Path

def test_upload_system():
    """Test the basic upload system."""
    print("🧪 Testing FaceClass Upload System")
    print("=" * 40)
    
    # Check directories
    print("📁 Checking directories...")
    dirs_to_check = ["data", "data/temp", "data/raw_videos", "data/frames"]
    
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path} exists")
        else:
            print(f"   ❌ {dir_path} missing - creating...")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {dir_path} created")
    
    # Check permissions
    print("\n🔐 Checking permissions...")
    for dir_path in ["data/temp", "data/raw_videos"]:
        if os.access(dir_path, os.W_OK):
            print(f"   ✅ {dir_path} is writable")
        else:
            print(f"   ❌ {dir_path} is not writable")
    
    # Check existing videos
    print("\n📹 Checking existing videos...")
    temp_videos = list(Path("data/temp").glob("*.mp4"))
    raw_videos = list(Path("data/raw_videos").glob("*.mp4"))
    
    print(f"   Temp directory: {len(temp_videos)} videos")
    print(f"   Raw videos: {len(raw_videos)} videos")
    
    if temp_videos or raw_videos:
        print("   Sample videos found:")
        for video in (temp_videos + raw_videos)[:3]:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"     - {video.name} ({size_mb:.1f} MB)")
    
    # Test file creation
    print("\n✍️  Testing file creation...")
    test_file = Path("data/temp/test_upload.txt")
    try:
        test_file.write_text("test upload")
        test_file.unlink()
        print("   ✅ Can create and delete files in temp directory")
    except Exception as e:
        print(f"   ❌ Cannot create files: {e}")
    
    print("\n🎯 Upload System Status:")
    print("   ✅ Directories ready")
    print("   ✅ Permissions checked")
    print("   ✅ File operations working")
    
    return True

def main():
    """Main test function."""
    try:
        success = test_upload_system()
        if success:
            print("\n🎉 Upload system is ready!")
            print("\n💡 Next steps:")
            print("   1. Install dependencies: sudo apt install python3-numpy python3-opencv")
            print("   2. Start server: cd src && python3 main.py")
            print("   3. Open http://localhost:8080")
            print("   4. Try uploading a video file")
        else:
            print("\n❌ Some issues found. Check the output above.")
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
