#!/usr/bin/env python3
"""
FaceClass - Comprehensive Student Attendance Analysis System
Main entry point for the classroom face analysis system.

This module orchestrates the entire pipeline:
1. Video processing and face detection
2. Face recognition and identification
3. Emotion and attention analysis
4. Attendance tracking and recording
5. Spatial analysis and heatmap generation
6. Comprehensive reporting and visualization
7. Dashboard interface
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime
from typing import Dict
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from dashboard.dashboard_ui import DashboardUI
from detection.face_tracker import FaceTracker
from recognition.face_identifier import FaceIdentifier
from emotion.emotion_detector import EmotionDetector
from attendance.attendance_tracker import AttendanceTracker
from layout_analysis.layout_mapper import LayoutMapper
from reporting.report_generator import ReportGenerator

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('faceclass.log')
        ]
    )

def extract_video_frames(video_path: str, output_dir: str = "data/frames"):
    """Extract frames from video for dashboard use."""
    from utils.video_utils import extract_key_frames, get_video_info
    
    logger = logging.getLogger(__name__)
    
    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        logger.error("Failed to get video information")
        return []
    
    logger.info(f"Video duration: {video_info['duration']:.2f} seconds")
    logger.info(f"Video resolution: {video_info['width']}x{video_info['height']}")
    
    # Extract key frames (10 frames for dashboard)
    frame_paths = extract_key_frames(video_path, output_dir, num_frames=10)
    
    if frame_paths:
        logger.info(f"Successfully extracted {len(frame_paths)} frames to {output_dir}")
    else:
        logger.warning("No frames were extracted")
    
    return frame_paths

def process_video_comprehensive(video_path: str, config: Config) -> Dict:
    """Process video with comprehensive analysis pipeline."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting comprehensive video analysis: {video_path}")
    
    start_time = time.time()
    
    # Initialize components
    face_tracker = FaceTracker(config)
    face_identifier = FaceIdentifier(config)
    emotion_detector = EmotionDetector(config)
    attendance_tracker = AttendanceTracker(config)
    layout_mapper = LayoutMapper(config)
    report_generator = ReportGenerator(config)
    
    # Start attendance session
    session_id = attendance_tracker.start_session()
    logger.info(f"Started attendance session: {session_id}")
    
    # Process video
    logger.info("Processing video for face detection and tracking...")
    detections = face_tracker.process_video(video_path)
    
    if not detections:
        logger.warning("No faces detected in video")
        return {}
    
    logger.info(f"Detected {len(detections)} face instances")
    
    # Process detections through the pipeline
    processed_detections = []
    
    for detection in detections:
        # Face recognition
        if 'frame' in detection and 'bbox' in detection:
            face_region = face_identifier._extract_face_region(detection['frame'], detection['bbox'])
            if face_region is not None:
                student_id, confidence = face_identifier.identify_face(face_region)
                detection['student_id'] = student_id
                detection['recognition_confidence'] = confidence
        
        # Emotion and attention analysis
        if 'frame' in detection and 'bbox' in detection:
            face_region = emotion_detector._extract_face_region(detection['frame'], detection['bbox'])
            if face_region is not None:
                # Detect emotions
                emotions = emotion_detector.detect_emotions(face_region)
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                detection['emotion'] = {
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion,
                    'confidence': emotions[dominant_emotion]
                }
                
                # Detect attention
                attention_data = emotion_detector.detect_attention(face_region)
                detection['attention'] = attention_data
        
        processed_detections.append(detection)
    
    # Process attendance
    logger.info("Processing attendance data...")
    timestamp = datetime.now()
    attendance_detections = attendance_tracker.process_detections(processed_detections, timestamp)
    
    # Generate spatial analysis
    logger.info("Generating spatial analysis...")
    spatial_data = layout_mapper.analyze_spatial_distribution(processed_detections)
    heatmaps = layout_mapper.generate_multiple_heatmaps(processed_detections)
    spatial_data['heatmaps'] = heatmaps
    
    # End attendance session
    session_summary = attendance_tracker.end_session()
    logger.info(f"Ended attendance session: {session_id}")
    
    # Generate statistics
    attendance_stats = attendance_tracker.get_attendance_statistics()
    emotion_stats = emotion_detector.get_emotion_statistics(processed_detections)
    attention_stats = emotion_detector.get_attention_statistics(processed_detections)
    
    # Prepare video info
    video_info = {
        'path': video_path,
        'duration': time.time() - start_time,
        'frame_count': len(processed_detections),
        'resolution': 'Unknown',
        'processing_time': time.time() - start_time,
        'detection_summary': {
            'total_detections': len(processed_detections),
            'unique_students': len(set(d.get('student_id', 'unknown') for d in processed_detections)),
            'average_confidence': np.mean([d.get('confidence', 0.0) for d in processed_detections])
        }
    }
    
    # Generate comprehensive report
    logger.info("Generating comprehensive report...")
    report_path = report_generator.generate_comprehensive_report(
        attendance_stats,
        emotion_stats,
        attention_stats,
        spatial_data,
        video_info
    )
    
    # Prepare results
    results = {
        'session_id': session_id,
        'video_path': video_path,
        'processing_time': time.time() - start_time,
        'detections': processed_detections,
        'attendance_data': attendance_stats,
        'emotion_data': emotion_stats,
        'attention_data': attention_stats,
        'spatial_data': spatial_data,
        'video_info': video_info,
        'report_path': report_path,
        'session_summary': session_summary
    }
    
    logger.info(f"Comprehensive analysis completed in {time.time() - start_time:.2f} seconds")
    return results

def main():
    """Main function to run the FaceClass analysis pipeline."""
    parser = argparse.ArgumentParser(description='FaceClass - Comprehensive Student Attendance Analysis')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='data/outputs', help='Output directory')
    parser.add_argument('--mode', choices=['detection', 'recognition', 'emotion', 'attendance', 'full', 'dashboard', 'extract-frames', 'report'], 
                       default='full', help='Analysis mode')
    parser.add_argument('--extract-frames', action='store_true', help='Extract frames from video')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting FaceClass comprehensive analysis pipeline")
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Extract frames if requested
        if args.extract_frames or args.mode == 'extract-frames':
            if not args.video:
                logger.error("Video path required for frame extraction")
                return
            
            logger.info("Extracting frames from video...")
            frame_paths = extract_video_frames(args.video)
            if frame_paths:
                logger.info(f"Frames extracted successfully. Use these in dashboard.")
            return
        
        # Launch dashboard only
        if args.mode == 'dashboard':
            logger.info("Launching dashboard only...")
            dashboard = DashboardUI(config)
            dashboard.run(debug=False)
            return
        
        # Process video with comprehensive analysis
        if args.video:
            if args.mode == 'full' or args.mode == 'attendance':
                logger.info("Running comprehensive video analysis...")
                results = process_video_comprehensive(args.video, config)
                
                if results:
                    logger.info("Analysis completed successfully!")
                    logger.info(f"Report generated: {results['report_path']}")
                    logger.info(f"Session ID: {results['session_id']}")
                    logger.info(f"Processing time: {results['processing_time']:.2f} seconds")
                    
                    # Print summary
                    print("\n=== ANALYSIS SUMMARY ===")
                    print(f"Total detections: {len(results['detections'])}")
                    print(f"Unique students: {results['video_info']['detection_summary']['unique_students']}")
                    print(f"Average confidence: {results['video_info']['detection_summary']['average_confidence']:.2f}")
                    print(f"Report location: {results['report_path']}")
                else:
                    logger.error("Analysis failed or no results generated")
            else:
                logger.info(f"Running {args.mode} analysis...")
                # Implement specific mode analysis here
                pass
        else:
            logger.error("Video path required for analysis")
            return
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return

if __name__ == "__main__":
    main() 