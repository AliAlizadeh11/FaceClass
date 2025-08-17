"""
Video processing service for FaceClass project.
Orchestrates the complete face detection, tracking, and recognition pipeline.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import json
from datetime import datetime

from .face_detection import FaceDetectionService
from .face_tracking import FaceTrackingService
from .face_recognition import FaceRecognitionService
from .visualization import VisualizationService

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main video processing service that coordinates detection, tracking, and recognition."""
    
    def __init__(self, config: Dict):
        """Initialize video processor with all required services.
        
        Args:
            config: Configuration dictionary containing all service parameters
        """
        self.config = config
        
        # Initialize services
        self.face_detector = FaceDetectionService(config)
        self.face_tracker = FaceTrackingService(config)
        
        # Initialize face recognizer if available
        try:
            self.face_recognizer = FaceRecognitionService(config)
            self.recognition_available = True
            logger.info("Face recognition service initialized")
        except Exception as e:
            logger.warning(f"Face recognition service not available: {e}")
            self.face_recognizer = None
            self.recognition_available = False
        
        self.visualizer = VisualizationService(config)
        
        # Processing state
        self.current_session = None
        self.processing_stats = {}
        
        logger.info("Video processor initialized with face detection and tracking services")
    
    def process_video(
        self, 
        video_path: str, 
        output_dir: Optional[str] = None,
        save_annotated_video: bool = True,
        save_results: bool = True
    ) -> Dict:
        """Process video through the complete pipeline.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs (optional)
            save_annotated_video: Whether to save annotated video
            save_results: Whether to save processing results
            
        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info(f"Starting video processing: {video_path}")
        start_time = time.time()
        
        # Validate video file
        if not self._validate_video(video_path):
            return {'error': 'Invalid video file'}
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(self.config.get('paths.outputs', 'data/outputs')) / datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing
        self._initialize_processing(video_path)
        
        # Process video frame by frame
        results = self._process_video_frames(video_path, output_dir, save_annotated_video)
        
        # Save annotated video if requested
        if save_annotated_video and results.get('detections'):
            annotated_video_path = self._save_annotated_video(video_path, results, output_dir)
            if annotated_video_path:
                results['annotated_video_path'] = annotated_video_path
        
        # Generate summary and save results
        if save_results:
            self._save_processing_results(results, output_dir)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        results['output_directory'] = str(output_dir)
        
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        return results
    
    def _validate_video(self, video_path: str) -> bool:
        """Validate video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video is valid
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if fps <= 0 or frame_count <= 0:
            logger.error(f"Invalid video properties: fps={fps}, frames={frame_count}")
            return False
        
        logger.info(f"Video validated: {frame_count} frames, {fps:.2f} fps, {width}x{height}")
        return True
    
    def _initialize_processing(self, video_path: str):
        """Initialize processing session.
        
        Args:
            video_path: Path to video file
        """
        self.current_session = {
            'video_path': video_path,
            'start_time': datetime.now(),
            'frame_count': 0,
            'detections': [],
            'tracks': {},
            'recognitions': []
        }
        
        # Reset tracking service
        self.face_tracker.reset()
        
        logger.info("Processing session initialized")
    
    def _process_video_frames(
        self, 
        video_path: str, 
        output_dir: Path, 
        save_annotated_video: bool
    ) -> Dict:
        """Process video frames through the pipeline.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory
            save_annotated_video: Whether to save annotated video
            
        Returns:
            Dictionary containing processing results
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer for annotated output
        video_writer = None
        if save_annotated_video:
            output_video_path = output_dir / 'annotated_video.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        all_tracks = []
        all_recognitions = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process current frame
                frame_results = self._process_single_frame(frame, frame_count)
                
                # Store results
                if frame_results['detections']:
                    all_detections.extend(frame_results['detections'])
                if frame_results['tracks']:
                    all_tracks.extend(frame_results['tracks'])
                if frame_results['recognitions']:
                    all_recognitions.extend(frame_results['recognitions'])
                
                # Draw annotations and save frame
                if save_annotated_video:
                    annotated_frame = self._draw_annotations(frame, frame_results)
                    video_writer.write(annotated_frame)
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        except Exception as e:
            logger.error(f"Error processing video frames: {e}")
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
        
        # Prepare results
        results = {
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'duration': total_frames / fps if fps > 0 else 0
            },
            'processing_stats': {
                'frames_processed': frame_count,
                'total_detections': len(all_detections),
                'total_tracks': len(all_tracks),
                'total_recognitions': len(all_recognitions)
            },
            'detections': all_detections,
            'tracks': all_tracks,
            'recognitions': all_recognitions,
            'tracking_summary': self.face_tracker.get_tracking_summary(),
            'recognition_summary': self.face_recognizer.get_database_info() if self.recognition_available else "Face recognition service not available."
        }
        
        return results
    
    def _save_annotated_video(self, video_path: str, results: Dict, output_dir: Path) -> Optional[str]:
        """Save annotated video with all detections.
        
        Args:
            video_path: Path to input video
            results: Processing results
            output_dir: Output directory
            
        Returns:
            Path to annotated video or None if failed
        """
        try:
            # Prepare detections per frame with enhanced information
            detections_per_frame = []
            total_frames = results['video_info']['total_frames']
            
            for frame_id in range(total_frames):
                frame_detections = []
                
                # Get detections for this frame
                for detection in results.get('detections', []):
                    if detection.get('frame_id') == frame_id:
                        # Enhance detection with additional info
                        enhanced_detection = {
                            'track_id': detection.get('track_id', 'N/A'),
                            'student_id': 'unknown',  # Will be updated if recognition exists
                            'bbox': detection['bbox'],
                            'confidence': detection.get('confidence', 0.0),
                            'emotion': detection.get('emotion', ''),
                            'is_attentive': detection.get('is_attentive', True)
                        }
                        frame_detections.append(enhanced_detection)
                
                # Get recognition results for this frame and merge with detections
                for recognition in results.get('recognitions', []):
                    if recognition.get('frame_id') == frame_id:
                        # Find matching detection to update
                        detection_updated = False
                        for detection in frame_detections:
                            if detection.get('track_id') == recognition.get('track_id'):
                                detection.update({
                                    'student_id': recognition.get('student_id', 'unknown'),
                                    'recognition_confidence': recognition.get('confidence', 0.0)
                                })
                                detection_updated = True
                                break
                        
                        # If no matching detection found, create new one
                        if not detection_updated:
                            frame_detections.append({
                                'track_id': recognition.get('track_id', 'N/A'),
                                'student_id': recognition.get('student_id', 'unknown'),
                                'bbox': recognition['bbox'],
                                'confidence': recognition.get('confidence', 0.0),
                                'emotion': '',
                                'is_attentive': True
                            })
                
                detections_per_frame.append(frame_detections)
                
                # Log progress for long videos
                if frame_id % 100 == 0:
                    logger.info(f"Prepared annotations for frame {frame_id}/{total_frames}")
            
            # Generate output path
            video_name = Path(video_path).stem
            annotated_path = output_dir / f"{video_name}_annotated.mp4"
            
            # Save annotated video
            success = self.visualizer.save_annotated_video(
                video_path,
                detections_per_frame,
                str(annotated_path),
                results['video_info']['fps']
            )
            
            if success:
                logger.info(f"Annotated video saved: {annotated_path}")
                
                # Also save to static directory for web access
                static_path = Path('static/processed_videos') / f"{video_name}_annotated.mp4"
                static_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file to static directory
                import shutil
                shutil.copy2(annotated_path, static_path)
                logger.info(f"Annotated video copied to static directory: {static_path}")
                
                return str(static_path)
            else:
                logger.error("Failed to save annotated video")
                return None
                
        except Exception as e:
            logger.error(f"Error saving annotated video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_keyframes(self, results: Dict, output_dir: Path, session_id: str) -> List[str]:
        """Save keyframes for each detected student.
        
        Args:
            results: Processing results
            output_dir: Output directory
            session_id: Session identifier
            
        Returns:
            List of keyframe paths
        """
        try:
            keyframe_paths = []
            
            # Group detections by student
            student_detections = {}
            for detection in results.get('detections', []):
                student_id = detection.get('student_id', 'unknown')
                if student_id not in student_detections:
                    student_detections[student_id] = []
                student_detections[student_id].append(detection)
            
            # Save first clear detection for each student
            for student_id, detections in student_detections.items():
                if not detections:
                    continue
                
                # Get the first detection with highest confidence
                best_detection = max(detections, key=lambda x: x.get('confidence', 0))
                
                # We need to extract the frame for this detection
                # For now, we'll use the first available frame
                # In a full implementation, you'd extract the specific frame
                keyframe_path = self.visualizer.save_keyframe(
                    np.zeros((480, 640, 3), dtype=np.uint8),  # Placeholder frame
                    best_detection,
                    str(output_dir / 'keyframes'),
                    session_id
                )
                
                if keyframe_path:
                    keyframe_paths.append(keyframe_path)
            
            return keyframe_paths
            
        except Exception as e:
            logger.error(f"Error saving keyframes: {e}")
            return []
    
    def _process_single_frame(self, frame: np.ndarray, frame_id: int) -> Dict:
        """Process a single frame through the pipeline.
        
        Args:
            frame: Input frame as numpy array
            frame_id: Current frame ID
            
        Returns:
            Dictionary containing frame processing results
        """
        frame_results = {
            'frame_id': frame_id,
            'detections': [],
            'tracks': [],
            'recognitions': []
        }
        
        try:
            # Step 1: Face Detection
            detections = self.face_detector.detect_faces(frame)
            
            # Add frame information to detections
            for detection in detections:
                detection['frame_id'] = frame_id
                detection['timestamp'] = frame_id / 30.0  # Assuming 30 fps
            
            frame_results['detections'] = detections
            
            # Step 2: Face Tracking
            if detections:
                tracked_detections = self.face_tracker.update(detections, frame_id)
                frame_results['tracks'] = tracked_detections
                
                # Step 3: Face Recognition (if available)
                if self.recognition_available and detections:
                    for tracked_detection in tracked_detections:
                        if 'bbox' in tracked_detection:
                            # Extract face region
                            face_region = self._extract_face_region(frame, tracked_detection['bbox'])
                            if face_region is not None:
                                # Perform recognition
                                student_id, confidence = self.face_recognizer.identify_face(face_region)
                                
                                recognition_result = {
                                    'frame_id': frame_id,
                                    'track_id': tracked_detection.get('track_id'),
                                    'student_id': student_id,
                                    'confidence': confidence,
                                    'bbox': tracked_detection['bbox']
                                }
                                
                                frame_results['recognitions'].append(recognition_result)
                                
                                # Update detection with recognition info
                                tracked_detection['student_id'] = student_id
                                tracked_detection['recognition_confidence'] = confidence
                else:
                    # If face recognition not available, assign generic IDs
                    for tracked_detection in tracked_detections:
                        tracked_detection['student_id'] = f'Person_{tracked_detection.get("track_id", "Unknown")}'
                        tracked_detection['recognition_confidence'] = 0.0
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
        
        return frame_results
    
    def _extract_face_region(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract face region from frame using bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            
        Returns:
            Extracted face region or None if extraction fails
        """
        try:
            x, y, w, h = bbox
            
            # Ensure coordinates are within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Ensure minimum size
            if face_region.shape[0] < 20 or face_region.shape[1] < 20:
                return None
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face region: {e}")
            return None
    
    def _draw_annotations(self, frame: np.ndarray, frame_results: Dict) -> np.ndarray:
        """Draw annotations on frame using visualization service.
        
        Args:
            frame: Input frame
            frame_results: Frame processing results
            
        Returns:
            Annotated frame
        """
        # Prepare detections in the format expected by visualization service
        detections = []
        
        # Add detection results
        for detection in frame_results.get('detections', []):
            if 'bbox' in detection:
                detections.append({
                    'track_id': detection.get('track_id', 'N/A'),
                    'student_id': 'unknown',
                    'bbox': detection['bbox'],
                    'confidence': detection.get('confidence', 0.0)
                })
        
        # Add recognition results
        for recognition in frame_results.get('recognitions', []):
            if 'bbox' in recognition:
                # Find matching detection to update
                for detection in detections:
                    if detection.get('track_id') == recognition.get('track_id'):
                        detection.update({
                            'student_id': recognition.get('student_id', 'unknown'),
                            'confidence': recognition.get('confidence', 0.0)
                        })
                        break
                else:
                    # Create new detection if no match found
                    detections.append({
                        'track_id': recognition.get('track_id', 'N/A'),
                        'student_id': recognition.get('student_id', 'unknown'),
                        'bbox': recognition['bbox'],
                        'confidence': recognition.get('confidence', 0.0)
                    })
        
        # Use visualization service to annotate frame
        return self.visualizer.annotate_frame(frame, detections)
    
    def _save_processing_results(self, results: Dict, output_dir: Path):
        """Save processing results to files.
        
        Args:
            results: Processing results dictionary
            output_dir: Output directory
        """
        try:
            # Save JSON results
            results_file = output_dir / 'processing_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary statistics
            summary_file = output_dir / 'summary.txt'
            with open(summary_file, 'w') as f:
                f.write("FaceClass Video Processing Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Video: {results['video_info']['path']}\n")
                f.write(f"Duration: {results['video_info']['duration']:.2f} seconds\n")
                f.write(f"Frames: {results['video_info']['total_frames']}\n")
                f.write(f"Resolution: {results['video_info']['resolution']}\n\n")
                
                f.write(f"Processing Time: {results['processing_time']:.2f} seconds\n")
                f.write(f"Frames Processed: {results['processing_stats']['frames_processed']}\n")
                f.write(f"Total Detections: {results['processing_stats']['total_detections']}\n")
                f.write(f"Total Tracks: {results['processing_stats']['total_tracks']}\n")
                f.write(f"Total Recognitions: {results['processing_stats']['total_recognitions']}\n\n")
                
                f.write("Tracking Summary:\n")
                tracking_summary = results['tracking_summary']
                for key, value in tracking_summary.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nRecognition Summary:\n")
                recognition_summary = results['recognition_summary']
                for key, value in recognition_summary.items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"Processing results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving processing results: {e}")
    
    def get_processing_summary(self) -> Dict:
        """Get summary of current processing session.
        
        Returns:
            Dictionary containing processing summary
        """
        if not self.current_session:
            return {}
        
        return {
            'session_info': self.current_session,
            'detection_stats': self.face_detector.get_model_info(),
            'tracking_stats': self.face_tracker.get_tracking_summary(),
            'recognition_stats': self.face_recognizer.get_model_info() if self.recognition_available else "Face recognition service not available."
        }
    
    def add_student_face(self, student_id: str, face_image: np.ndarray) -> bool:
        """Add a student face to the recognition database.
        
        Args:
            student_id: Student identifier
            face_image: Face image as numpy array
            
        Returns:
            True if face was added successfully
        """
        return self.face_recognizer.add_face(student_id, face_image) if self.recognition_available else False
    
    def remove_student_face(self, student_id: str) -> bool:
        """Remove a student face from the recognition database.
        
        Args:
            student_id: Student identifier
            
        Returns:
            True if face was removed successfully
        """
        return self.face_recognizer.remove_face(student_id) if self.recognition_available else False
    
    def get_database_info(self) -> Dict:
        """Get information about the face recognition database.
        
        Returns:
            Dictionary containing database information
        """
        return self.face_recognizer.get_database_info() if self.recognition_available else "Face recognition service not available."
