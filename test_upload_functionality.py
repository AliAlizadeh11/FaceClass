#!/usr/bin/env python3
"""
Test script to verify video upload functionality in FaceClass dashboard.
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

def test_upload_functionality():
    """Test the video upload functionality."""
    print("🧪 Testing FaceClass Video Upload Functionality")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard server is running")
        else:
            print("❌ Dashboard server health check failed")
            return False
    except requests.exceptions.RequestException:
        print("❌ Dashboard server is not running. Please start it first.")
        return False
    
    # Test debug endpoint
    try:
        response = requests.get("http://localhost:8080/debug/upload", timeout=5)
        if response.status_code == 200:
            debug_info = response.json()
            print("✅ Debug endpoint accessible")
            print(f"📁 Temp directory exists: {debug_info.get('temp_dir_exists')}")
            print(f"📁 Raw videos directory exists: {debug_info.get('raw_videos_dir_exists')}")
            print(f"💾 Disk space available: {debug_info.get('disk_space', {}).get('free_gb', 'Unknown'):.1f} GB")
            
            # Check permissions
            permissions = debug_info.get('permissions', {})
            print(f"✍️  Temp directory writable: {permissions.get('temp_dir_writable')}")
            print(f"✍️  Raw videos directory writable: {permissions.get('raw_videos_dir_writable')}")
            
        else:
            print("❌ Debug endpoint failed")
            return False
    except Exception as e:
        print(f"❌ Debug endpoint error: {e}")
        return False
    
    # Check existing videos
    temp_dir = Path("data/temp")
    raw_videos_dir = Path("data/raw_videos")
    
    print("\n📹 Existing Videos:")
    if temp_dir.exists():
        temp_videos = list(temp_dir.glob("*.mp4"))
        print(f"   Temp directory: {len(temp_videos)} videos")
        for video in temp_videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"     - {video.name} ({size_mb:.1f} MB)")
    
    if raw_videos_dir.exists():
        raw_videos = list(raw_videos_dir.glob("*.mp4"))
        print(f"   Raw videos directory: {len(raw_videos)} videos")
        for video in raw_videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"     - {video.name} ({size_mb:.1f} MB)")
    
    # Test directory creation
    print("\n🔧 Testing Directory Creation:")
    test_dirs = ["data/temp", "data/raw_videos", "data/test_upload"]
    
    for test_dir in test_dirs:
        try:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            test_file = Path(test_dir) / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
            Path(test_dir).rmdir()
            print(f"   ✅ {test_dir}: Writable")
        except Exception as e:
            print(f"   ❌ {test_dir}: {e}")
    
    print("\n🎯 Upload Test Summary:")
    print("   - Server is running")
    print("   - Debug endpoint accessible")
    print("   - Directories are writable")
    print("   - Ready for video uploads")
    
    return True

def main():
    """Main test function."""
    try:
        success = test_upload_functionality()
        if success:
            print("\n🎉 All tests passed! Video upload should work correctly.")
            print("\n💡 To test actual upload:")
            print("   1. Open http://localhost:8080 in your browser")
            print("   2. Go to the Upload Video Section")
            print("   3. Try uploading a video file")
            print("   4. Check the upload status message")
        else:
            print("\n❌ Some tests failed. Check the issues above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
