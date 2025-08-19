#!/usr/bin/env python3
"""
Test script for Enhanced Face Detection Service.
Demonstrates improved detection accuracy, preprocessing, and tracking.
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from services.enhanced_face_detection import EnhancedFaceDetectionService
from config import Config

def load_config():
    """Load configuration for testing."""
    try:
        config = Config()
        return config.config
    except Exception as e:
        print(f"Error loading config: {e}")
        # Fallback configuration
        return {
            'face_detection': {
                'model': 'ensemble',
                'confidence_threshold': 0.3,
                'nms_threshold': 0.3,
                'min_face_size': 15,
                'max_faces': 100,
                'ensemble_models': ['yolo', 'mediapipe', 'mtcnn', 'opencv'],
                'ensemble_voting': True,
                'ensemble_confidence_threshold': 0.2,
                'preprocessing': {
                    'denoising': True,
                    'contrast_enhancement': True,
                    'super_resolution': False,
                    'scale_factors': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                },
                'enable_tracking': True,
                'track_persistence': 30
            },
            'paths': {
                'models': 'models'
            }
        }

def visualize_detections(frame, detections, frame_id):
    """Visualize face detections on frame with enhanced information."""
    annotated_frame = frame.copy()
    
    # Color coding for different models
    model_colors = {
        'yolo': (0, 255, 0),      # Green
        'mediapipe': (255, 0, 0),  # Blue
        'mtcnn': (0, 0, 255),     # Red
        'opencv': (255, 255, 0),   # Cyan
        'ensemble': (255, 0, 255)  # Magenta
    }
    
    for i, det in enumerate(detections):
        x, y, w, h = det.bbox
        
        # Determine color based on model
        base_color = model_colors.get(det.model.split('_')[0], (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), base_color, 2)
        
        # Draw label with detection info
        label_parts = [
            f"ID: {det.track_id if det.track_id else 'N/A'}",
            f"Conf: {det.confidence:.2f}",
            f"Size: {det.face_size if det.face_size else 'N/A'}",
            f"Model: {det.model}"
        ]
        
        if det.quality_score:
            label_parts.append(f"Quality: {det.quality_score:.2f}")
        
        label = " | ".join(label_parts)
        
        # Calculate label position
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_y = max(y - 10, label_size[1] + 10)
        
        # Draw label background
        cv2.rectangle(annotated_frame, 
                     (x, label_y - label_size[1] - 10),
                     (x + label_size[0], label_y + 5),
                     base_color, -1)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw track ID prominently
        if det.track_id is not None:
            cv2.putText(annotated_frame, f"#{det.track_id}", (x + w - 30, y + h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add frame info
    info_text = f"Frame: {frame_id} | Faces: {len(detections)}"
    cv2.putText(annotated_frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return annotated_frame

def test_image_detection(service, image_path):
    """Test face detection on a single image."""
    print(f"\n🔍 Testing image detection: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"📏 Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    start_time = time.time()
    detections = service.detect_faces_enhanced(image, frame_id=0)
    processing_time = time.time() - start_time
    
    print(f"⚡ Processing time: {processing_time:.3f}s")
    print(f"👥 Faces detected: {len(detections)}")
    
    # Display detection details
    for i, det in enumerate(detections):
        print(f"  Face {i+1}:")
        print(f"    Bbox: {det.bbox}")
        print(f"    Confidence: {det.confidence:.3f}")
        print(f"    Model: {det.model}")
        print(f"    Size: {det.face_size}")
        print(f"    Quality: {det.quality_score:.3f}")
        if det.track_id is not None:
            print(f"    Track ID: {det.track_id}")
        print()
    
    # Visualize results
    annotated_image = visualize_detections(image, detections, 0)
    
    # Save annotated image
    output_path = Path("test_output") / f"enhanced_detection_{Path(image_path).stem}.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), annotated_image)
    print(f"💾 Annotated image saved: {output_path}")
    
    return detections

def test_video_detection(service, video_path, max_frames=100):
    """Test face detection on video with tracking."""
    print(f"\n🎥 Testing video detection: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"🎯 Processing max {max_frames} frames...")
    
    # Setup video writer for output
    output_path = Path("test_output") / f"enhanced_detection_{Path(video_path).stem}.mp4"
    output_path.parent.mkdir(exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    total_faces = 0
    start_time = time.time()
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections = service.detect_faces_enhanced(frame, frame_id=frame_count)
            total_faces += len(detections)
            
            # Visualize detections
            annotated_frame = visualize_detections(frame, detections, frame_count)
            
            # Write to output video
            out.write(annotated_frame)
            
            # Progress update
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_count}/{max_frames} | Faces: {len(detections)} | Avg FPS: {avg_fps:.1f}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
    
    finally:
        cap.release()
        out.release()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_faces_per_frame = total_faces / frame_count if frame_count > 0 else 0
    
    print(f"\n📊 Video processing completed:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total faces detected: {total_faces}")
    print(f"  Average faces per frame: {avg_faces_per_frame:.1f}")
    print(f"  Processing time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"💾 Output video saved: {output_path}")

def test_performance_metrics(service):
    """Test and display performance metrics."""
    print(f"\n📈 Performance Metrics:")
    
    metrics = service.get_performance_summary()
    
    print(f"  Total frames processed: {metrics['total_frames_processed']}")
    print(f"  Total faces detected: {metrics['total_faces_detected']}")
    print(f"  Average detection FPS: {metrics['average_detection_fps']:.1f}")
    print(f"  Detection models loaded: {metrics['detection_models_loaded']}")
    print(f"  Tracking enabled: {metrics['tracking_enabled']}")
    print(f"  Active tracks: {metrics['active_tracks']}")
    
    print(f"\n  Configuration:")
    config = metrics['configuration']
    print(f"    Confidence threshold: {config['confidence_threshold']}")
    print(f"    Min face size: {config['min_face_size']}")
    print(f"    Ensemble models: {config['ensemble_models']}")
    
    preprocessing = config['preprocessing_enabled']
    print(f"    Preprocessing:")
    print(f"      Denoising: {preprocessing['denoising']}")
    print(f"      Contrast enhancement: {preprocessing['contrast_enhancement']}")
    print(f"      Super resolution: {preprocessing['super_resolution']}")

def main():
    """Main test function."""
    print("🚀 Enhanced Face Detection Service Test")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print("✅ Configuration loaded")
    
    # Initialize service
    try:
        service = EnhancedFaceDetectionService(config)
        print("✅ Enhanced face detection service initialized")
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        return
    
    # Test performance metrics
    test_performance_metrics(service)
    
    # Test with sample images if available
    test_images = [
        "test_face_detection.jpg",
        "data/frames/sample.jpg",
        "static/sample_frames/frame_001.jpg"
    ]
    
    for image_path in test_images:
        if Path(image_path).exists():
            test_image_detection(service, image_path)
            break
    else:
        print("\n⚠️ No test images found, skipping image detection test")
    
    # Test with sample video if available
    test_videos = [
        "test_video.mp4",
        "data/raw_videos/sample.mp4",
        "static/sample_video.mp4"
    ]
    
    for video_path in test_videos:
        if Path(video_path).exists():
            test_video_detection(service, video_path, max_frames=50)
            break
    else:
        print("\n⚠️ No test videos found, skipping video detection test")
    
    # Final performance summary
    print("\n" + "=" * 50)
    test_performance_metrics(service)
    
    print("\n🎉 Enhanced face detection test completed!")
    print("💡 Check the 'test_output' directory for results")

if __name__ == "__main__":
    main()
