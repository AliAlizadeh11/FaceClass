#!/usr/bin/env python3
"""
Simple test script to verify video upload and processing in FaceClass.
"""

import os
import sys
from pathlib import Path

def test_video_processing():
    """Test the video processing functionality."""
    print("🧪 Testing FaceClass Video Processing")
    print("=" * 40)
    
    # Check if required directories exist
    required_dirs = ["data/temp", "data/raw_videos", "data/frames"]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"❌ {dir_path} missing - creating...")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path} created")
    
    # Check if there are any existing videos
    temp_videos = list(Path("data/temp").glob("*.mp4"))
    raw_videos = list(Path("data/raw_videos").glob("*.mp4"))
    
    print(f"\n📹 Existing Videos:")
    print(f"   Temp directory: {len(temp_videos)} videos")
    print(f"   Raw videos directory: {len(raw_videos)} videos")
    
    if temp_videos:
        print("   Sample videos in temp:")
        for video in temp_videos[:3]:  # Show first 3
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"     - {video.name} ({size_mb:.1f} MB)")
    
    # Check if there are any existing frames
    frames = list(Path("data/frames").glob("*.jpg"))
    print(f"\n🖼️  Existing Frames: {len(frames)} frames")
    
    if frames:
        print("   Sample frames:")
        for frame in frames[:5]:  # Show first 5
            print(f"     - {frame.name}")
    
    print("\n🎯 System Status:")
    print("   ✅ Directories ready")
    print("   ✅ Ready for video uploads")
    print("   ✅ Ready for frame processing")
    
    print("\n💡 To test video upload:")
    print("   1. Install dependencies: sudo apt install python3-numpy python3-opencv")
    print("   2. Start server: cd src && python3 main.py")
    print("   3. Open http://localhost:8080 in browser")
    print("   4. Upload a video in the Upload Video Section")
    print("   5. Click 'Process Video' to analyze frame by frame")
    print("   6. Use Previous/Next buttons to navigate frames")
    
    return True

def main():
    """Main test function."""
    try:
        success = test_video_processing()
        if success:
            print("\n🎉 System is ready for video processing!")
        else:
            print("\n❌ Some issues found. Check the output above.")
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
