#!/usr/bin/env python3
"""
Debug script for video processing issues
"""

import os
import sys
from pathlib import Path

def check_video_processing_files():
    """Check for video processing files and identify issues."""
    print("🔍 Debugging Video Processing Issues")
    print("=" * 50)
    
    # Check static directories
    print("\n📁 Checking Static Directories:")
    static_dirs = [
        'static',
        'static/processed_videos',
        'static/keyframes',
        'static/thumbnails'
    ]
    
    for dir_path in static_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}: {len(list(path.iterdir()))} files")
            if dir_path == 'static/processed_videos':
                # List video files
                video_files = list(path.glob('*.mp4')) + list(path.glob('*.avi')) + list(path.glob('*.mov'))
                if video_files:
                    print(f"    Video files found:")
                    for video_file in video_files:
                        print(f"      - {video_file.name} ({video_file.stat().st_size / (1024*1024):.1f} MB)")
                else:
                    print(f"    ❌ No video files found")
        else:
            print(f"  ❌ {dir_path}: Directory does not exist")
    
    # Check data directories
    print("\n📁 Checking Data Directories:")
    data_dirs = [
        'data/outputs',
        'data/frames',
        'data/raw_videos'
    ]
    
    for dir_path in data_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}: {len(list(path.iterdir()))} files")
            # Look for any output directories with timestamps
            if 'outputs' in dir_path:
                output_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith('20')]
                if output_dirs:
                    print(f"    Output directories found:")
                    for output_dir in output_dirs[:5]:  # Show first 5
                        print(f"      - {output_dir.name}")
                        # Check for annotated videos in output dir
                        annotated_videos = list(output_dir.glob('*_annotated.mp4'))
                        if annotated_videos:
                            print(f"        Annotated videos: {[v.name for v in annotated_videos]}")
        else:
            print(f"  ❌ {dir_path}: Directory does not exist")
    
    # Check for any MP4 files in the project
    print("\n🎥 Searching for MP4 Files:")
    mp4_files = []
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or 'env' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(Path(root) / file)
    
    if mp4_files:
        print(f"  Found {len(mp4_files)} MP4 files:")
        for mp4_file in mp4_files[:10]:  # Show first 10
            size_mb = mp4_file.stat().st_size / (1024*1024)
            print(f"    - {mp4_file}: {size_mb:.1f} MB")
    else:
        print("  ❌ No MP4 files found in project")
    
    # Check for annotated video files specifically
    print("\n🎯 Searching for Annotated Videos:")
    annotated_patterns = ['*_annotated.mp4', '*annotated*.mp4', '*_annotated.mp4']
    annotated_found = []
    
    for pattern in annotated_patterns:
        for root, dirs, files in os.walk('.'):
            if 'venv' in root or 'env' in root or '__pycache__' in root:
                continue
            for file in files:
                if pattern.replace('*', '') in file and file.endswith('.mp4'):
                    annotated_found.append(Path(root) / file)
    
    if annotated_found:
        print(f"  Found {len(annotated_found)} annotated videos:")
        for video in annotated_found:
            size_mb = video.stat().st_size / (1024*1024)
            print(f"    - {video}: {size_mb:.1f} MB")
    else:
        print("  ❌ No annotated videos found")
    
    # Check file permissions
    print("\n🔐 Checking File Permissions:")
    static_video_dir = Path('static/processed_videos')
    if static_video_dir.exists():
        try:
            # Try to create a test file
            test_file = static_video_dir / 'test_permissions.txt'
            test_file.write_text('test')
            test_file.unlink()
            print(f"  ✅ {static_video_dir}: Write permissions OK")
        except Exception as e:
            print(f"  ❌ {static_video_dir}: Permission error - {e}")
    else:
        print(f"  ⚠ {static_video_dir}: Directory does not exist")
    
    print("\n" + "=" * 50)
    print("🔧 Debug Information Complete")
    
    if not mp4_files:
        print("\n❌ ISSUE IDENTIFIED: No MP4 files found in project")
        print("   This suggests video processing is not working or files are not being saved")
    elif not annotated_found:
        print("\n⚠ ISSUE IDENTIFIED: No annotated videos found")
        print("   Video processing may be working but annotation is failing")
    else:
        print("\n✅ Video files found - check paths and permissions")

def check_video_processor():
    """Check if video processor is working."""
    print("\n🔧 Checking Video Processor:")
    
    try:
        # Try to import video processor
        sys.path.append('src')
        from services.video_processor import VideoProcessor
        print("  ✅ VideoProcessor imported successfully")
        
        # Check if we can create an instance
        try:
            processor = VideoProcessor({})
            print("  ✅ VideoProcessor instance created successfully")
        except Exception as e:
            print(f"  ❌ VideoProcessor instance creation failed: {e}")
            
    except ImportError as e:
        print(f"  ❌ VideoProcessor import failed: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

if __name__ == "__main__":
    check_video_processing_files()
    check_video_processor()
