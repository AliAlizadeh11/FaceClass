#!/usr/bin/env python3
"""
Prepare Website Frames
Copies and organizes sample annotated frames for the website display.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np

def create_sample_frames():
    """Create sample annotated frames for website display."""
    print("🖼️ Creating sample annotated frames for website...")
    
    # Create directories
    sample_frames_dir = Path("static/sample_frames")
    sample_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample frames with annotations
    frame_configs = [
        {"name": "frame_0010_annotated.jpg", "students": 3, "frame_id": 10},
        {"name": "frame_0030_annotated.jpg", "students": 4, "frame_id": 30},
        {"name": "frame_0060_annotated.jpg", "students": 5, "frame_id": 60},
        {"name": "frame_0090_annotated.jpg", "students": 5, "frame_id": 90},
        {"name": "frame_0120_annotated.jpg", "students": 5, "frame_id": 120},
        {"name": "frame_0150_annotated.jpg", "students": 5, "frame_id": 150}
    ]
    
    for config in frame_configs:
        frame = create_annotated_frame(config["students"], config["frame_id"])
        frame_path = sample_frames_dir / config["name"]
        cv2.imwrite(str(frame_path), frame)
        print(f"  ✓ Created: {frame_path}")
    
    print(f"✓ Sample frames created: {len(frame_configs)} frames")

def create_annotated_frame(num_students, frame_id):
    """Create a sample annotated frame with bounding boxes."""
    # Create base frame
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(600):
        for x in range(800):
            intensity = int(200 + 55 * (y / 600))
            frame[y, x] = [intensity, intensity, intensity]
    
    # Add frame info
    cv2.rectangle(frame, (20, 20), (780, 60), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 20), (780, 60), (255, 255, 255), 1)
    cv2.putText(frame, f"Frame {frame_id:04d} - FaceClass Analysis", (40, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add timestamp
    cv2.putText(frame, f"Time: {frame_id/30:.1f}s", (650, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Create student positions
    student_positions = [
        (150, 150), (400, 150), (600, 150),  # Top row
        (200, 350), (450, 350)               # Bottom row
    ]
    
    student_names = ["Alice", "Bob", "Carol", "David", "Eva"]
    colors = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (128, 0, 128)]
    
    for i in range(min(num_students, len(student_positions))):
        x, y = student_positions[i]
        student_name = student_names[i]
        color = colors[i]
        
        # Draw face (oval)
        cv2.ellipse(frame, (x, y), (30, 40), 0, 0, 360, (255, 218, 185), -1)
        cv2.ellipse(frame, (x, y), (30, 40), (0, 0, 0), 2)
        
        # Eyes
        cv2.circle(frame, (x - 12, y - 12), 4, (0, 0, 0), -1)
        cv2.circle(frame, (x + 12, y - 12), 4, (0, 0, 0), -1)
        
        # Nose
        cv2.circle(frame, (x, y), 3, (255, 150, 100), -1)
        
        # Mouth
        cv2.ellipse(frame, (x, y + 20), (10, 4), 0, 0, 180, (0, 0, 0), 2)
        
        # Hair
        cv2.ellipse(frame, (x, y - 20), (35, 25), 0, 0, 360, (139, 69, 19), -1)
        cv2.ellipse(frame, (x, y - 20), (35, 25), 0, 0, 360, (0, 0, 0), 1)
        
        # Bounding box
        bbox_x1, bbox_y1 = x - 40, y - 50
        bbox_x2, bbox_y2 = x + 40, y + 50
        
        # Box color based on student status
        if i < 3:  # First 3 students are recognized and attentive
            box_color = (0, 255, 0)  # Green
            status = "Attentive"
        elif i == 3:  # 4th student is recognized but inattentive
            box_color = (0, 255, 255)  # Yellow
            status = "Inattentive"
        else:  # 5th student is unknown
            box_color = (0, 0, 255)  # Red
            status = "Unknown"
        
        cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), box_color, 1)
        
        # Labels
        label_y = bbox_y1 - 10
        if label_y < 20:  # Adjust if label would go off-screen
            label_y = bbox_y2 + 20
        
        # Student name and track ID
        cv2.rectangle(frame, (bbox_x1, label_y - 20), (bbox_x1 + 200, label_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (bbox_x1, label_y - 20), (bbox_x1 + 200, label_y + 5), box_color, 1)
        cv2.putText(frame, f"Name: {student_name}", (bbox_x1 + 5, label_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Track ID
        track_label_y = label_y + 20
        cv2.rectangle(frame, (bbox_x1, track_label_y - 20), (bbox_x1 + 200, track_label_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (bbox_x1, track_label_y - 20), (bbox_x1 + 200, track_label_y + 5), box_color, 1)
        cv2.putText(frame, f"Track ID: {i+1}", (bbox_x1 + 5, track_label_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Confidence scores
        conf_label_y = track_label_y + 20
        cv2.rectangle(frame, (bbox_x1, conf_label_y - 20), (bbox_x1 + 200, conf_label_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (bbox_x1, conf_label_y - 20), (bbox_x1 + 200, conf_label_y + 5), box_color, 1)
        cv2.putText(frame, f"Detection: 95%", (bbox_x1 + 5, conf_label_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Recognition confidence
        rec_conf_label_y = conf_label_y + 20
        cv2.rectangle(frame, (bbox_x1, rec_conf_label_y - 20), (bbox_x1 + 200, rec_conf_label_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (bbox_x1, rec_conf_label_y - 20), (bbox_x1 + 200, rec_conf_label_y + 5), box_color, 1)
        
        if i < 4:  # Recognized students
            rec_conf = 95 - (i * 2)  # Varying confidence
            cv2.putText(frame, f"Recognition: {rec_conf}%", (bbox_x1 + 5, rec_conf_label_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:  # Unknown student
            cv2.putText(frame, f"Recognition: N/A", (bbox_x1 + 5, rec_conf_label_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status
        status_label_y = rec_conf_label_y + 20
        cv2.rectangle(frame, (bbox_x1, status_label_y - 20), (bbox_x1 + 200, status_label_y + 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (bbox_x1, status_label_y - 20), (bbox_x1 + 200, status_label_y + 5), box_color, 1)
        cv2.putText(frame, f"Status: {status}", (bbox_x1 + 5, status_label_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add legend
    legend_y = 500
    cv2.rectangle(frame, (20, legend_y), (780, legend_y + 80), (240, 240, 240), -1)
    cv2.rectangle(frame, (20, legend_y), (780, legend_y + 80), (100, 100, 100), 1)
    
    cv2.putText(frame, "Legend:", (40, legend_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Color indicators
    cv2.rectangle(frame, (40, legend_y + 30), (60, legend_y + 50), (0, 255, 0), -1)
    cv2.putText(frame, "Green: Recognized & Attentive", (70, legend_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.rectangle(frame, (40, legend_y + 55), (60, legend_y + 75), (0, 255, 255), -1)
    cv2.putText(frame, "Yellow: Recognized but Inattentive", (70, legend_y + 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.rectangle(frame, (350, legend_y + 30), (370, legend_y + 50), (0, 0, 255), -1)
    cv2.putText(frame, "Red: Unknown Face", (380, legend_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame

def copy_existing_frames():
    """Copy existing annotated frames to the website directory."""
    print("📁 Copying existing annotated frames...")
    
    source_dir = Path("frame_analysis_output")
    target_dir = Path("static/sample_frames")
    
    if not source_dir.exists():
        print("  ⚠️ Source directory not found, creating sample frames instead")
        return
    
    # Find annotated frames
    frame_files = list(source_dir.glob("frame_*_annotated.jpg"))
    
    if not frame_files:
        print("  ⚠️ No annotated frames found, creating sample frames instead")
        return
    
    # Copy frames (limit to 6 for website display)
    frames_to_copy = sorted(frame_files)[:6]
    
    for frame_file in frames_to_copy:
        target_file = target_dir / frame_file.name
        shutil.copy2(frame_file, target_file)
        print(f"  ✓ Copied: {frame_file.name}")
    
    print(f"✓ Frames copied: {len(frames_to_copy)} frames")

def main():
    """Main function to prepare website frames."""
    print("🚀 Preparing Website Frames")
    print("=" * 50)
    
    # Create static directories
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # Try to copy existing frames first
    copy_existing_frames()
    
    # If no existing frames, create sample ones
    sample_frames_dir = Path("static/sample_frames")
    if not sample_frames_dir.exists() or not list(sample_frames_dir.glob("*.jpg")):
        create_sample_frames()
    
    print("\n✅ Website frames preparation complete!")
    print("📁 Frames available at: static/sample_frames/")
    print("🌐 Ready for website display!")

if __name__ == "__main__":
    main()
