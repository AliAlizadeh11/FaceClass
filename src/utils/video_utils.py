"""
Video utilities for FaceClass project.
Handles video processing, frame extraction, and analysis.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import os

logger = logging.getLogger(__name__)


def extract_frames(video_path: str, output_dir: str, frame_interval: int = 30) -> List[str]:
    """
    Extract frames from video at regular intervals.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30 = 1 frame per second at 30fps)
    
    Returns:
        List of paths to extracted frame images
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f} seconds")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at regular intervals
        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{extracted_count:04d}.jpg"
            frame_path = output_path / frame_filename
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame)
            if success:
                frame_paths.append(str(frame_path))
                extracted_count += 1
                logger.info(f"Extracted frame {extracted_count}: {frame_filename}")
            else:
                logger.error(f"Failed to save frame {frame_count}")
        
        frame_count += 1
        
        # Progress update
        if frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from video")
    return frame_paths


def get_video_info(video_path: str) -> dict:
    """
    Get video information including duration, frame count, and resolution.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    info = {
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0
    }
    
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info


def extract_key_frames(video_path: str, output_dir: str, num_frames: int = 10) -> List[str]:
    """
    Extract key frames from video using scene detection.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of key frames to extract
    
    Returns:
        List of paths to extracted key frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frames at regular intervals
        if frame_count % frame_interval == 0 and extracted_count < num_frames:
            frame_filename = f"keyframe_{extracted_count:02d}.jpg"
            frame_path = output_path / frame_filename
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame)
            if success:
                frame_paths.append(str(frame_path))
                extracted_count += 1
                logger.info(f"Extracted key frame {extracted_count}: {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frame_paths)} key frames from video")
    return frame_paths


def create_frame_sequence(video_path: str, output_dir: str, start_time: float = 0, 
                         duration: float = 60, frame_interval: int = 5) -> List[str]:
    """
    Extract a sequence of frames from a specific time period.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        start_time: Start time in seconds
        duration: Duration to extract in seconds
        frame_interval: Extract every Nth frame
    
    Returns:
        List of paths to extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    # Set start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_count = start_frame
    extracted_count = 0
    
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at regular intervals
        if (frame_count - start_frame) % frame_interval == 0:
            frame_filename = f"sequence_{extracted_count:03d}.jpg"
            frame_path = output_path / frame_filename
            
            # Save frame
            success = cv2.imwrite(str(frame_path), frame)
            if success:
                frame_paths.append(str(frame_path))
                extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from sequence")
    return frame_paths 