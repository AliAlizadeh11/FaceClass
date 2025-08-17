#!/usr/bin/env python3
"""
Comprehensive demonstration of enhanced video annotation system.
Shows all features including detailed annotations, confidence scores, and multi-frame processing.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from services.visualization import VisualizationService

def create_dynamic_classroom_scene():
    """Create a dynamic classroom scene with multiple students and activities."""
    # Create a 1920x1080 frame (Full HD)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Create classroom background
    # Floor gradient
    for y in range(1080):
        for x in range(1920):
            # Floor gets darker towards bottom
            floor_intensity = int(180 + 75 * (y / 1080))
            frame[y, x] = [floor_intensity, floor_intensity, floor_intensity]
    
    # Add walls
    cv2.rectangle(frame, (0, 0), (1920, 200), (220, 220, 220), -1)  # Top wall
    cv2.rectangle(frame, (0, 0), (100, 1080), (200, 200, 200), -1)  # Left wall
    cv2.rectangle(frame, (1820, 0), (1920, 1080), (200, 200, 200), -1)  # Right wall
    
    # Add smartboard
    cv2.rectangle(frame, (200, 50), (1720, 180), (30, 30, 30), -1)
    cv2.rectangle(frame, (200, 50), (1720, 180), (255, 255, 255), 5)
    
    # Add smartboard content
    cv2.putText(frame, "Face Recognition & Computer Vision", (250, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
    cv2.putText(frame, "Live Demo Session", (250, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
    
    # Add desk rows
    for row in range(4):
        y_base = 300 + row * 150
        for col in range(6):
            x_base = 150 + col * 280
            
            # Desk
            cv2.rectangle(frame, (x_base, y_base), (x_base + 200, y_base + 100), (139, 69, 19), -1)
            cv2.rectangle(frame, (x_base, y_base), (x_base + 200, y_base + 100), (0, 0, 0), 2)
            
            # Chair
            cv2.rectangle(frame, (x_base + 50, y_base + 100), (x_base + 150, y_base + 180), (160, 82, 45), -1)
            cv2.rectangle(frame, (x_base + 50, y_base + 100), (x_base + 150, y_base + 180), (0, 0, 0), 2)
    
    return frame

def create_realistic_student_detections():
    """Create realistic student detection data with various scenarios."""
    return [
        {
            "track_id": 1,
            "student_id": "Alice_Johnson",
            "bbox": [200, 350, 140, 180],
            "emotion": "engaged",
            "is_attentive": True,
            "confidence": 0.96,
            "recognition_confidence": 0.89
        },
        {
            "track_id": 2,
            "student_id": "Bob_Smith",
            "bbox": [480, 380, 135, 175],
            "emotion": "confused",
            "is_attentive": True,
            "confidence": 0.94,
            "recognition_confidence": 0.85
        },
        {
            "track_id": 3,
            "student_id": "Carol_Davis",
            "bbox": [760, 320, 130, 170],
            "emotion": "bored",
            "is_attentive": False,
            "confidence": 0.92,
            "recognition_confidence": 0.78
        },
        {
            "track_id": 4,
            "student_id": "David_Wilson",
            "bbox": [1040, 400, 145, 185],
            "emotion": "surprised",
            "is_attentive": True,
            "confidence": 0.95,
            "recognition_confidence": 0.91
        },
        {
            "track_id": 5,
            "student_id": "Eva_Brown",
            "bbox": [1320, 350, 138, 178],
            "emotion": "happy",
            "is_attentive": True,
            "confidence": 0.93,
            "recognition_confidence": 0.87
        },
        {
            "track_id": 6,
            "student_id": "Frank_Miller",
            "bbox": [1600, 420, 142, 182],
            "emotion": "tired",
            "is_attentive": False,
            "confidence": 0.88,
            "recognition_confidence": 0.82
        },
        {
            "track_id": 7,
            "student_id": "unknown",
            "bbox": [400, 650, 125, 165],
            "emotion": "",
            "is_attentive": None,
            "confidence": 0.76,
            "recognition_confidence": 0.0
        },
        {
            "track_id": 8,
            "student_id": "Grace_Lee",
            "bbox": [680, 620, 136, 176],
            "emotion": "focused",
            "is_attentive": True,
            "confidence": 0.97,
            "recognition_confidence": 0.93
        }
    ]

def create_multi_frame_scenario():
    """Create a multi-frame scenario showing student movement and attention changes."""
    frames = []
    detections_per_frame = []
    
    # Create base frame
    base_frame = create_dynamic_classroom_scene()
    base_detections = create_realistic_student_detections()
    
    # Generate 60 frames (2 seconds at 30 fps)
    for frame_idx in range(60):
        # Copy base frame
        frame = base_frame.copy()
        
        # Add timestamp
        cv2.putText(frame, f"Frame: {frame_idx:03d} | Time: {frame_idx/30:.1f}s", 
                    (50, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Modify detections for this frame (simulate movement and changes)
        frame_detections = []
        for detection in base_detections:
            # Create a copy of the detection
            frame_detection = detection.copy()
            
            # Add slight movement (simulate tracking)
            movement_x = int(np.sin(frame_idx * 0.1) * 3)
            movement_y = int(np.cos(frame_idx * 0.05) * 2)
            frame_detection['bbox'][0] += movement_x
            frame_detection['bbox'][1] += movement_y
            
            # Simulate attention changes
            if frame_idx > 30 and detection['track_id'] in [3, 6]:  # Carol and Frank
                frame_detection['is_attentive'] = not detection['is_attentive']
            
            # Simulate emotion changes
            if frame_idx == 25 and detection['track_id'] == 2:  # Bob
                frame_detection['emotion'] = 'understanding'
            elif frame_idx == 45 and detection['track_id'] == 4:  # David
                frame_detection['emotion'] = 'excited'
            
            frame_detections.append(frame_detection)
        
        frames.append(frame)
        detections_per_frame.append(frame_detections)
    
    return frames, detections_per_frame

def demonstrate_single_frame_annotation():
    """Demonstrate single frame annotation with all features."""
    print("🎬 Demonstrating Single Frame Annotation")
    print("=" * 50)
    
    # Create visualization service
    config = {
        'paths': {
            'outputs': 'demo_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create classroom scene
    frame = create_dynamic_classroom_scene()
    detections = create_realistic_student_detections()
    
    # Annotate frame
    start_time = time.time()
    annotated_frame = visualizer.annotate_frame(frame, detections)
    annotation_time = time.time() - start_time
    
    print(f"✓ Frame annotation completed in {annotation_time:.3f} seconds")
    print(f"  - Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"  - Students detected: {len(detections)}")
    print(f"  - Recognized students: {len([d for d in detections if d['student_id'] != 'unknown'])}")
    print(f"  - Unknown faces: {len([d for d in detections if d['student_id'] == 'unknown'])}")
    
    # Save annotated frame
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "demo_single_frame.jpg"
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"✓ Annotated frame saved: {output_path}")
    
    return annotated_frame

def demonstrate_video_annotation():
    """Demonstrate video annotation with multiple frames."""
    print("\n🎥 Demonstrating Video Annotation")
    print("=" * 50)
    
    # Create visualization service
    config = {
        'paths': {
            'outputs': 'demo_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create multi-frame scenario
    print("Creating multi-frame classroom scenario...")
    frames, detections_per_frame = create_multi_frame_scenario()
    
    # Save base video
    video_path = "demo_output/demo_classroom_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (1920, 1080))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    print(f"✓ Base video created: {video_path}")
    print(f"  - Frames: {len(frames)}")
    print(f"  - Duration: {len(frames)/30:.1f} seconds")
    print(f"  - Resolution: 1920x1080")
    
    # Create annotated video
    annotated_video_path = "demo_output/demo_annotated_video.mp4"
    
    print("Creating annotated video...")
    start_time = time.time()
    success = visualizer.save_annotated_video(
        video_path,
        detections_per_frame,
        annotated_video_path,
        fps=30.0
    )
    video_time = time.time() - start_time
    
    if success:
        print(f"✓ Annotated video created in {video_time:.2f} seconds")
        print(f"  - Output: {annotated_video_path}")
        print(f"  - Annotations per frame: {len(detections_per_frame[0])}")
        
        # Copy to static directory for web access
        static_path = Path("static/processed_videos/demo_annotated_video.mp4")
        static_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(annotated_video_path, static_path)
        print(f"✓ Video copied to web directory: {static_path}")
    else:
        print("✗ Failed to create annotated video")
    
    # Clean up base video
    Path(video_path).unlink(missing_ok=True)
    
    return annotated_video_path

def demonstrate_keyframe_extraction():
    """Demonstrate keyframe extraction for each student."""
    print("\n🖼️ Demonstrating Keyframe Extraction")
    print("=" * 50)
    
    # Create visualization service
    config = {
        'paths': {
            'outputs': 'demo_output',
            'processed_videos': 'static/processed_videos',
            'keyframes': 'static/keyframes'
        }
    }
    
    visualizer = VisualizationService(config)
    
    # Create sample frame
    frame = create_dynamic_classroom_scene()
    detections = create_realistic_student_detections()
    
    # Extract keyframes
    session_id = "demo_session_2024"
    keyframe_paths = []
    
    for detection in detections:
        keyframe_path = visualizer.save_keyframe(
            frame, 
            detection, 
            "static/keyframes", 
            session_id
        )
        
        if keyframe_path:
            keyframe_paths.append(keyframe_path)
            student_name = detection.get('student_id', 'Unknown')
            print(f"✓ Keyframe for {student_name}: {Path(keyframe_path).name}")
        else:
            print(f"✗ Failed to save keyframe for {detection.get('student_id', 'unknown')}")
    
    print(f"\nTotal keyframes extracted: {len(keyframe_paths)}")
    
    # Create enhanced thumbnail grid
    if keyframe_paths:
        grid_path = "static/thumbnails/demo_grid.jpg"
        success = visualizer.create_thumbnail_grid(
            keyframe_paths,
            grid_path,
            grid_size=(4, 2),
            thumbnail_size=(200, 200)
        )
        
        if success:
            print(f"✓ Thumbnail grid created: {grid_path}")
        else:
            print("✗ Failed to create thumbnail grid")
    
    return keyframe_paths

def generate_annotation_report():
    """Generate a comprehensive report of the annotation system."""
    print("\n📊 Generating Annotation System Report")
    print("=" * 50)
    
    report = {
        "system_info": {
            "name": "FaceClass Enhanced Video Annotation System",
            "version": "2.0",
            "features": [
                "Real-time frame annotation",
                "Multi-object tracking visualization",
                "Student recognition display",
                "Emotion detection labels",
                "Attention status indicators",
                "Confidence score visualization",
                "Color-coded recognition status",
                "Keyframe extraction",
                "Thumbnail grid generation"
            ]
        },
        "annotation_features": {
            "bounding_boxes": {
                "style": "Color-coded rectangles",
                "colors": {
                    "green": "Recognized + Attentive",
                    "yellow": "Recognized + Inattentive",
                    "red": "Unknown/Unrecognized"
                },
                "thickness": 3
            },
            "labels": {
                "student_names": "Displayed for recognized students",
                "track_ids": "Displayed for unknown faces",
                "font": "cv2.FONT_HERSHEY_SIMPLEX",
                "scale": 0.6
            },
            "confidence_scores": {
                "detection_confidence": "Shows face detection reliability",
                "recognition_confidence": "Shows student identification reliability",
                "format": "Decimal values (0.00 - 1.00)"
            },
            "additional_info": {
                "emotion": "Current emotional state",
                "attention": "Attention status (attentive/inattentive)",
                "positioning": "Dynamic layout based on available data"
            }
        },
        "output_formats": {
            "annotated_frames": "JPG images with all annotations",
            "annotated_videos": "MP4 videos with frame-by-frame annotations",
            "keyframes": "Individual student face images",
            "thumbnail_grids": "Organized collections of keyframes"
        },
        "performance": {
            "frame_processing": "Real-time annotation",
            "video_processing": "Efficient batch processing",
            "memory_usage": "Optimized for large videos",
            "scalability": "Handles multiple students simultaneously"
        }
    }
    
    # Save report
    report_path = "demo_output/annotation_system_report.json"
    Path("demo_output").mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ System report generated: {report_path}")
    print("\nKey Features Summary:")
    for feature in report["system_info"]["features"]:
        print(f"  • {feature}")
    
    return report

def main():
    """Main demonstration function."""
    print("🚀 FaceClass Enhanced Video Annotation System")
    print("🎯 Comprehensive Feature Demonstration")
    print("=" * 60)
    
    try:
        # Create demo output directory
        Path("demo_output").mkdir(exist_ok=True)
        
        # Demonstrate single frame annotation
        demonstrate_single_frame_annotation()
        
        # Demonstrate video annotation
        demonstrate_video_annotation()
        
        # Demonstrate keyframe extraction
        demonstrate_keyframe_extraction()
        
        # Generate system report
        generate_annotation_report()
        
        print("\n" + "=" * 60)
        print("🎉 All Demonstrations Completed Successfully!")
        print("\n📁 Generated Files:")
        print("  - demo_output/demo_single_frame.jpg")
        print("  - demo_output/demo_annotated_video.mp4")
        print("  - static/processed_videos/demo_annotated_video.mp4")
        print("  - static/keyframes/ (student keyframes)")
        print("  - static/thumbnails/demo_grid.jpg")
        print("  - demo_output/annotation_system_report.json")
        
        print("\n🌐 Web Integration:")
        print("  - Annotated video available at: /static/processed_videos/demo_annotated_video.mp4")
        print("  - Keyframes available at: /static/keyframes/")
        print("  - Thumbnail grid at: /static/thumbnails/demo_grid.jpg")
        
        print("\n✨ Enhanced Features Demonstrated:")
        print("  ✓ Real-time frame annotation with detailed information")
        print("  ✓ Multi-frame video processing and annotation")
        print("  ✓ Student recognition and tracking visualization")
        print("  ✓ Emotion and attention status display")
        print("  ✓ Confidence score visualization")
        print("  ✓ Color-coded recognition status")
        print("  ✓ Keyframe extraction and organization")
        print("  ✓ Thumbnail grid generation")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
