"""
Visualization service for FaceClass project.
Provides frame annotation and video processing utilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for annotating frames and creating visual outputs."""
    
    def __init__(self, config: Dict):
        """Initialize visualization service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Color scheme for annotations
        self.colors = {
            'recognized_attentive': (0, 255, 0),      # Green
            'recognized_inattentive': (0, 255, 255),  # Yellow
            'unknown': (0, 0, 255),                   # Red
            'detection': (255, 255, 0),               # Cyan
            'text': (255, 255, 255),                  # White
            'text_bg': (0, 0, 0)                     # Black
        }
        
        # Font settings (thin, modern style)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.text_thickness = 1
        
        # Create output directories
        self._create_output_directories()
        
        logger.info("Visualization service initialized")
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        dirs = [
            'static/processed_videos',
            'static/keyframes',
            'static/thumbnails'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def annotate_frame(
        self, 
        frame: np.ndarray, 
        detections: List[Dict]
    ) -> np.ndarray:
        """Annotate a frame with detection results.
        
        Args:
            frame: Input frame as numpy array
            detections: List of detection dictionaries with format:
                {
                    "track_id": int,
                    "student_id": str,   # or "unknown"
                    "bbox": [x, y, w, h],
                    "emotion": str,      # optional
                    "is_attentive": bool # optional
                }
                
        Returns:
            Annotated frame as numpy array
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            if 'bbox' not in detection:
                continue
                
            x, y, w, h = detection['bbox']
            track_id = detection.get('track_id', 'N/A')
            student_id = detection.get('student_id', 'unknown')
            emotion = detection.get('emotion', '')
            is_attentive = detection.get('is_attentive', True)
            confidence = detection.get('confidence', 0.0)
            recognition_confidence = detection.get('recognition_confidence', 0.0)
            
            # Determine color based on recognition and attention status
            if student_id != 'unknown':
                if is_attentive:
                    color = self.colors['recognized_attentive']  # Green
                else:
                    color = self.colors['recognized_inattentive']  # Yellow
            else:
                color = self.colors['unknown']  # Red
            
            # Draw bounding box (thin stroke)
            cv2.rectangle(
                annotated_frame, 
                (x, y), 
                (x + w, y + h), 
                color, 
                1
            )
            
            # Prepare label text
            if student_id != 'unknown':
                label = f"Name: {student_id}"
            else:
                label = f"ID: {track_id}"
            
            # Calculate text position and size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Draw text background
            text_bg_x1 = x
            text_bg_y1 = max(0, y - text_height - 10)
            text_bg_x2 = x + text_width + 10
            text_bg_y2 = y
            
            cv2.rectangle(
                annotated_frame,
                (text_bg_x1, text_bg_y1),
                (text_bg_x2, text_bg_y2),
                self.colors['text_bg'],
                -1
            )
            
            # Draw main label
            cv2.putText(
                annotated_frame,
                label,
                (x + 5, y - 5),
                self.font,
                self.font_scale,
                self.colors['text'],
                self.font_thickness,
                cv2.LINE_AA
            )
            
            # Draw detection confidence (always display as percentage)
            if confidence > 0:
                conf_y = y + h + 20
                conf_percentage = int(confidence * 100)
                conf_text = f"Detection: {conf_percentage}%"
                
                # Confidence text background
                (conf_width, conf_height), _ = cv2.getTextSize(
                    conf_text, self.font, 0.4, 1
                )
                
                cv2.rectangle(
                    annotated_frame,
                    (x, conf_y - conf_height - 3),
                    (x + conf_width + 6, conf_y + 3),
                    self.colors['text_bg'],
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    conf_text,
                    (x + 3, conf_y),
                    self.font,
                    0.4,
                    self.colors['text'],
                    1,
                    cv2.LINE_AA
                )
            
            # Draw recognition confidence if available (always display as percentage)
            if recognition_confidence > 0:
                rec_conf_y = y + h + (40 if confidence > 0 else 20)
                rec_conf_percentage = int(recognition_confidence * 100)
                rec_conf_text = f"Recognition: {rec_conf_percentage}%"
                
                # Recognition confidence text background
                (rec_conf_width, rec_conf_height), _ = cv2.getTextSize(
                    rec_conf_text, self.font, 0.4, 1
                )
                
                cv2.rectangle(
                    annotated_frame,
                    (x, rec_conf_y - rec_conf_height - 3),
                    (x + rec_conf_width + 6, rec_conf_y + 3),
                    self.colors['text_bg'],
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    rec_conf_text,
                    (x + 3, rec_conf_y),
                    self.font,
                    0.4,
                    self.colors['text'],
                    1,
                    cv2.LINE_AA
                )
            
            # Draw emotion if available
            if emotion:
                emotion_y = y + h + (60 if recognition_confidence > 0 else 40)
                emotion_text = f"Emotion: {emotion}"
                
                # Emotion text background
                (emotion_width, emotion_height), _ = cv2.getTextSize(
                    emotion_text, self.font, 0.5, 1
                )
                
                cv2.rectangle(
                    annotated_frame,
                    (x, emotion_y - emotion_height - 5),
                    (x + emotion_width + 10, emotion_y + 5),
                    self.colors['text_bg'],
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    emotion_text,
                    (x + 5, emotion_y),
                    self.font,
                    0.5,
                    self.colors['text'],
                    1,
                    cv2.LINE_AA
                )
            
            # Draw attention status if available
            if 'is_attentive' in detection:
                attention_y = y + h + (80 if emotion else 60)
                attention_text = "Attentive" if is_attentive else "Not Attentive"
                attention_color = self.colors['recognized_attentive'] if is_attentive else self.colors['unknown']
                
                # Attention text background
                (attention_width, attention_height), _ = cv2.getTextSize(
                    attention_text, self.font, 0.5, 1
                )
                
                cv2.rectangle(
                    annotated_frame,
                    (x, attention_y - attention_height - 5),
                    (x + attention_width + 10, attention_y + 5),
                    self.colors['text_bg'],
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    attention_text,
                    (x + 5, attention_y),
                    self.font,
                    0.5,
                    attention_color,
                    1,
                    cv2.LINE_AA
                )
        
        return annotated_frame
    
    def save_annotated_video(
        self,
        video_path: str,
        detections_per_frame: List[List[Dict]],
        output_path: str,
        fps: int = 30
    ) -> bool:
        """Save annotated video with all detections.
        
        Args:
            video_path: Path to input video
            detections_per_frame: List of detections for each frame
            output_path: Path to save annotated video
            fps: Frames per second for output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Annotate frame if we have detections
                if frame_idx < len(detections_per_frame):
                    annotated_frame = self.annotate_frame(
                        frame, 
                        detections_per_frame[frame_idx]
                    )
                else:
                    annotated_frame = frame
                
                out.write(annotated_frame)
                frame_idx += 1
                
                # Log progress
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx} frames for annotated video")
            
            cap.release()
            out.release()
            
            logger.info(f"Annotated video saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotated video: {e}")
            return False
    
    def save_keyframe(
        self,
        frame: np.ndarray,
        detection: Dict,
        output_dir: str,
        session_id: str
    ) -> Optional[str]:
        """Save a keyframe for a specific student detection.
        
        Args:
            frame: Input frame
            detection: Detection dictionary
            output_dir: Output directory for keyframes
            session_id: Session identifier
            
        Returns:
            Path to saved keyframe or None if failed
        """
        try:
            if 'bbox' not in detection or 'student_id' not in detection:
                return None
            
            x, y, w, h = detection['bbox']
            student_id = detection['student_id']
            track_id = detection.get('track_id', 'unknown')
            
            # Extract face region with some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Create filename
            if student_id != 'unknown':
                filename = f"{student_id}_{track_id}_{session_id}.jpg"
            else:
                filename = f"unknown_{track_id}_{session_id}.jpg"
            
            output_path = Path(output_dir) / filename
            
            # Save keyframe
            success = cv2.imwrite(str(output_path), face_region)
            
            if success:
                logger.info(f"Keyframe saved: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Failed to save keyframe: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving keyframe: {e}")
            return None
    
    def create_thumbnail_grid(
        self,
        keyframe_paths: List[str],
        output_path: str,
        grid_size: Tuple[int, int] = (4, 3),
        thumbnail_size: Tuple[int, int] = (200, 200)
    ) -> bool:
        """Create a thumbnail grid from keyframe images.
        
        Args:
            keyframe_paths: List of paths to keyframe images
            output_path: Path to save thumbnail grid
            grid_size: Grid dimensions (rows, cols)
            thumbnail_size: Size of each thumbnail (width, height)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            rows, cols = grid_size
            thumb_w, thumb_h = thumbnail_size
            
            # Calculate grid dimensions
            grid_width = cols * thumb_w
            grid_height = rows * thumb_h
            
            # Create blank grid
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
            
            # Place thumbnails in grid
            for i, keyframe_path in enumerate(keyframe_paths[:rows*cols]):
                if not Path(keyframe_path).exists():
                    continue
                
                # Load and resize thumbnail
                img = cv2.imread(keyframe_path)
                if img is None:
                    continue
                
                img_resized = cv2.resize(img, thumbnail_size)
                
                # Calculate position in grid
                row = i // cols
                col = i % cols
                
                y1 = row * thumb_h
                y2 = y1 + thumb_h
                x1 = col * thumb_w
                x2 = x1 + thumb_w
                
                # Place thumbnail in grid
                grid[y1:y2, x1:x2] = img_resized
                
                # Add border (thin)
                cv2.rectangle(grid, (x1, y1), (x2, y2), (0, 0, 0), 1)
            
            # Save grid
            success = cv2.imwrite(output_path, grid)
            
            if success:
                logger.info(f"Thumbnail grid saved: {output_path}")
                return True
            else:
                logger.error(f"Failed to save thumbnail grid: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating thumbnail grid: {e}")
            return False
    
    def get_annotation_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get the color scheme used for annotations.
        
        Returns:
            Dictionary mapping annotation types to BGR colors
        """
        return self.colors.copy()
    
    def set_custom_colors(self, colors: Dict[str, Tuple[int, int, int]]):
        """Set custom colors for annotations.
        
        Args:
            colors: Dictionary mapping annotation types to BGR colors
        """
        self.colors.update(colors)
        logger.info("Custom annotation colors updated")
