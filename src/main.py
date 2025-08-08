#!/usr/bin/env python3
"""
FaceClass - Computer Vision Final Project
Main entry point for the classroom face analysis system.

This module orchestrates the entire pipeline:
1. Video processing and face detection
2. Face recognition and identification
3. Emotion and attention analysis
4. Layout mapping and heatmap generation
5. Dashboard visualization
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from dashboard.dashboard_ui import DashboardUI

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

def main():
    """Main function to run the FaceClass analysis pipeline."""
    parser = argparse.ArgumentParser(description='FaceClass - Classroom Face Analysis')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='data/outputs', help='Output directory')
    parser.add_argument('--mode', choices=['detection', 'recognition', 'emotion', 'full', 'dashboard', 'extract-frames'], 
                       default='full', help='Analysis mode')
    parser.add_argument('--extract-frames', action='store_true', help='Extract frames from video')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting FaceClass analysis pipeline")
    
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
        
        # Import other modules only if needed
        if args.mode in ['detection', 'recognition', 'emotion', 'full']:
            from detection.face_tracker import FaceTracker
            from recognition.face_identifier import FaceIdentifier
            from emotion.emotion_detector import EmotionDetector
            from layout_analysis.layout_mapper import LayoutMapper
            
            # Initialize components
            face_tracker = FaceTracker(config)
            face_identifier = FaceIdentifier(config)
            emotion_detector = EmotionDetector(config)
            layout_mapper = LayoutMapper(config)
            dashboard = DashboardUI(config)
        
        # Process video if provided
        if args.video:
            logger.info(f"Processing video: {args.video}")
            
            # Extract frames for dashboard if not already done
            frames_dir = Path("data/frames")
            if not frames_dir.exists() or not list(frames_dir.glob("*.jpg")):
                logger.info("Extracting frames for dashboard...")
                extract_video_frames(args.video)
            
            # Run analysis pipeline
            if args.mode in ['detection', 'full']:
                logger.info("Running face detection and tracking...")
                face_tracker.process_video(args.video)
            
            if args.mode in ['recognition', 'full']:
                logger.info("Running face recognition...")
                detections = face_tracker.get_detections()
                if detections:
                    face_identifier.process_detections(detections)
                else:
                    logger.warning("No detections found for face recognition")
            
            if args.mode in ['emotion', 'full']:
                logger.info("Running emotion analysis...")
                detections = face_tracker.get_detections()
                if detections:
                    emotion_detector.process_detections(detections)
                else:
                    logger.warning("No detections found for emotion analysis")
            
            if args.mode in ['full']:
                logger.info("Running layout analysis...")
                detections = face_tracker.get_detections()
                if detections:
                    layout_mapper.generate_heatmap(detections, args.output_dir)
                else:
                    logger.warning("No detections found for layout analysis")
        
        # Launch dashboard
        if args.mode in ['full', 'dashboard']:
            logger.info("Launching dashboard...")
            dashboard.run(debug=False)
    
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 