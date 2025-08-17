#!/usr/bin/env python3
"""
Fix video paths by copying annotated videos to static directory
"""

import shutil
from pathlib import Path

def fix_video_paths():
    """Copy all annotated videos to static directory for web access."""
    print("🔧 Fixing Video Paths for Website Access")
    print("=" * 50)
    
    # Find all annotated videos in data/outputs
    output_dir = Path('data/outputs')
    static_dir = Path('static/processed_videos')
    static_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_videos = []
    
    if output_dir.exists():
        print(f"📁 Searching in: {output_dir}")
        for output_subdir in output_dir.iterdir():
            if output_subdir.is_dir() and output_subdir.name.startswith('20'):
                print(f"  🔍 Checking: {output_subdir.name}")
                # Look for annotated videos in this directory
                for video_file in output_subdir.glob('*_annotated.mp4'):
                    # Copy to static directory
                    static_path = static_dir / video_file.name
                    if not static_path.exists():
                        try:
                            shutil.copy2(video_file, static_path)
                            size_mb = video_file.stat().st_size / (1024*1024)
                            fixed_videos.append({
                                'source': str(video_file),
                                'destination': str(static_path),
                                'size_mb': size_mb
                            })
                            print(f"    ✅ Copied: {video_file.name} ({size_mb:.1f} MB)")
                        except Exception as e:
                            print(f"    ❌ Failed to copy {video_file.name}: {e}")
                    else:
                        print(f"    ⚠ Already exists: {video_file.name}")
    
    # Also check for any videos in the root data directory
    data_dir = Path('data')
    print(f"\n📁 Searching in: {data_dir}")
    for video_file in data_dir.rglob('*_annotated.mp4'):
        if 'outputs' not in str(video_file):  # Skip outputs directory
            static_path = static_dir / video_file.name
            if not static_path.exists():
                try:
                    shutil.copy2(video_file, static_path)
                    size_mb = video_file.stat().st_size / (1024*1024)
                    fixed_videos.append({
                        'source': str(video_file),
                        'destination': str(static_path),
                        'size_mb': size_mb
                    })
                    print(f"  ✅ Copied: {video_file.name} ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"  ❌ Failed to copy {video_file.name}: {e}")
            else:
                print(f"  ⚠ Already exists: {video_file.name}")
    
    # Check what's now available in static directory
    print(f"\n📁 Static Directory Contents: {static_dir}")
    if static_dir.exists():
        video_files = list(static_dir.glob('*.mp4'))
        if video_files:
            print(f"  Found {len(video_files)} MP4 files:")
            for video_file in video_files:
                size_mb = video_file.stat().st_size / (1024*1024)
                print(f"    - {video_file.name} ({size_mb:.1f} MB)")
        else:
            print("  ❌ No MP4 files found")
    else:
        print("  ❌ Static directory does not exist")
    
    print(f"\n🎉 Video Path Fix Complete!")
    print(f"  Total videos fixed: {len(fixed_videos)}")
    print(f"  Static directory: {static_dir.absolute()}")
    
    if fixed_videos:
        print(f"\n📋 Fixed Videos:")
        for video in fixed_videos:
            print(f"  - {Path(video['source']).name} -> {Path(video['destination']).name}")
    
    return fixed_videos

if __name__ == "__main__":
    fix_video_paths()
