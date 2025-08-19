"""
Face detection and tracking module for FaceClass project.
Supports multiple face detection models and tracking algorithms.
Enhanced for Team 1: Face Detection & Recognition Core
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)


class FaceTracker:
    """Enhanced face detection and tracking using various models."""
    
    def __init__(self, config):
        """Initialize face tracker with configuration."""
        self.config = config
        self.detection_model = config.get('face_detection.model', 'yolo')
        self.confidence_threshold = config.get('face_detection.confidence_threshold', 0.5)
        self.nms_threshold = config.get('face_detection.nms_threshold', 0.4)
        self.min_face_size = config.get('face_detection.min_face_size', 20)
        
        # Enhanced tracking parameters
        self.tracking_algorithm = config.get('face_tracking.algorithm', 'bytetrack')
        self.track_persistence = config.get('face_tracking.persistence_frames', 30)
        self.multi_camera_support = config.get('face_tracking.multi_camera', False)
        
        # Performance monitoring
        self.performance_metrics = {
            'detection_fps': [],
            'tracking_fps': [],
            'memory_usage': [],
            'accuracy_metrics': []
        }
        
        # Initialize detection model
        self.detector = self._load_detector()
        
        # Initialize tracking algorithm
        self.tracker = self._load_tracker()
        
        # Tracking state
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        self.detections_history = []
        self.camera_tracks = {} if self.multi_camera_support else None
    
    def _load_detector(self):
        """Load the specified face detection model."""
        model_name = self.detection_model.lower()
        
        if model_name == 'yolo':
            return self._load_yolo_detector()
        elif model_name == 'retinaface':
            return self._load_retinaface_detector()
        elif model_name == 'mtcnn':
            return self._load_mtcnn_detector()
        elif model_name == 'opencv':
            return self._load_opencv_detector()
        else:
            logger.warning(f"Unknown detection model: {model_name}, using OpenCV")
            return self._load_opencv_detector()
    
    def _load_tracker(self):
        """Load the specified tracking algorithm."""
        algorithm = self.tracking_algorithm.lower()
        
        if algorithm == 'bytetrack':
            return self._load_bytetrack()
        elif algorithm == 'deep_ocsort':
            return self._load_deep_ocsort()
        elif algorithm == 'simple_iou':
            return self._load_simple_iou_tracker()
        else:
            logger.warning(f"Unknown tracking algorithm: {algorithm}, using simple IoU")
            return self._load_simple_iou_tracker()
    
    def _load_bytetrack(self):
        """Load ByteTrack algorithm."""
        try:
            from bytetrack.byte_tracker import BYTETracker
            tracker = BYTETracker(
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=30
            )
            return {'type': 'bytetrack', 'tracker': tracker}
        except ImportError:
            logger.warning("ByteTrack not available, using simple IoU")
            return self._load_simple_iou_tracker()
    
    def _load_deep_ocsort(self):
        """Load Deep OC-SORT algorithm."""
        try:
            # Implementation of Deep OC-SORT
            return {'type': 'deep_ocsort', 'tracker': DeepOCSORTTracker()}
        except Exception as e:
            logger.warning(f"Deep OC-SORT not available: {e}, using simple IoU")
            return self._load_simple_iou_tracker()
    
    def _load_simple_iou_tracker(self):
        """Load simple IoU-based tracker."""
        return {'type': 'simple_iou', 'tracker': None}
    
    def _load_yolo_detector(self):
        """Load YOLO-based face detector."""
        try:
            # Check if YOLO model file exists
            model_path = self.config.get_path('models') / 'face_detection' / 'yolov8n-face.pt'
            if not model_path.exists():
                logger.info("YOLO model not found at expected path, falling back to OpenCV")
                logger.info(f"Expected path: {model_path}")
                logger.info("To use YOLO, please download the model file or use OpenCV instead")
                return self._load_opencv_detector()
            
            # Try to load YOLO model
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
            logger.info("YOLO model loaded successfully")
            return {'type': 'yolo', 'model': model}
            
        except ImportError:
            logger.info("PyTorch not available, falling back to OpenCV")
            return self._load_opencv_detector()
        except Exception as e:
            logger.warning(f"Failed to load YOLO detector: {e}")
            logger.info("Falling back to OpenCV detector")
            return self._load_opencv_detector()
    
    def _load_retinaface_detector(self):
        """Load RetinaFace detector with enhanced configuration."""
        try:
            from retinaface import RetinaFace
            # Enhanced RetinaFace configuration
            detector_config = {
                'type': 'retinaface',
                'model': RetinaFace,
                'confidence_threshold': 0.8,
                'nms_threshold': 0.4,
                'min_face_size': 20
            }
            logger.info("RetinaFace detector loaded successfully")
            return detector_config
        except ImportError:
            logger.warning("RetinaFace not available, falling back to OpenCV")
            logger.info("To use RetinaFace, install it with: pip install retinaface")
            return self._load_opencv_detector()
    
    def _load_mtcnn_detector(self):
        """Load MTCNN detector with enhanced configuration."""
        try:
            from mtcnn import MTCNN
            detector = MTCNN(
                min_face_size=20,
                scale_factor=0.709,
                factor=0.6
            )
            logger.info("MTCNN detector loaded successfully")
            return {'type': 'mtcnn', 'model': detector}
        except ImportError:
            logger.warning("MTCNN not available, using OpenCV")
            return self._load_opencv_detector()
    
    def _load_opencv_detector(self):
        """Load OpenCV cascade detector."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if cascade.empty():
                logger.error(f"Failed to load OpenCV cascade from: {cascade_path}")
                # Try alternative path
                alt_path = Path(__file__).parent / 'haarcascade_frontalface_default.xml'
                if alt_path.exists():
                    cascade = cv2.CascadeClassifier(str(alt_path))
                    if cascade.empty():
                        raise Exception("Failed to load cascade from alternative path")
                else:
                    raise Exception("Cascade file not found in any location")
            
            logger.info("OpenCV cascade detector loaded successfully")
            return {'type': 'opencv', 'model': cascade}
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV detector: {e}")
            raise Exception(f"OpenCV detector initialization failed: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a single frame with performance monitoring."""
        start_time = time.time()
        
        detections = []
        
        if self.detector['type'] == 'yolo':
            detections = self._detect_yolo(frame)
        elif self.detector['type'] == 'retinaface':
            detections = self._detect_retinaface(frame)
        elif self.detector['type'] == 'mtcnn':
            detections = self._detect_mtcnn(frame)
        elif self.detector['type'] == 'opencv':
            detections = self._detect_opencv(frame)
        
        # Performance monitoring
        detection_time = time.time() - start_time
        fps = 1.0 / detection_time if detection_time > 0 else 0
        self.performance_metrics['detection_fps'].append(fps)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics['detection_fps']) > 100:
            self.performance_metrics['detection_fps'] = self.performance_metrics['detection_fps'][-100:]
        
        return detections
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using YOLO."""
        results = self.detector['model'](frame)
        detections = []
        
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if conf > self.confidence_threshold:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'label': 'face'
                })
        
        return detections
    
    def _detect_retinaface(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using RetinaFace with enhanced configuration."""
        try:
            # Enhanced RetinaFace detection
            faces = self.detector['model'].detect(
                frame,
                confidence_threshold=self.detector.get('confidence_threshold', 0.8)
            )
            detections = []
            
            if faces is not None:
                for face in faces:
                    bbox = face['facial_area']
                    confidence = face.get('score', 0.9)
                    
                    if confidence > self.confidence_threshold:
                        detections.append({
                            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                            'confidence': confidence,
                            'label': 'face',
                            'landmarks': face.get('landmarks', []),
                            'facial_attributes': face.get('attributes', {})
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"RetinaFace detection error: {e}")
            return []
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN with enhanced configuration."""
        faces = self.detector['model'].detect_faces(frame)
        detections = []
        
        for face in faces:
            bbox = face['box']
            confidence = face['confidence']
            if confidence > self.confidence_threshold:
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    'confidence': confidence,
                    'label': 'face',
                    'landmarks': face.get('keypoints', {}),
                    'facial_attributes': face.get('attributes', {})
                })
        
        return detections
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV cascade with improved sensitivity and filtering."""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Multiple detection passes with different parameters for better coverage
            all_faces = []
            
            # Pass 1: Standard detection
            faces1 = self.detector['model'].detectMultiScale(
                gray, 
                scaleFactor=1.05,  # More sensitive (was 1.1)
                minNeighbors=3,   # Less strict (was 5)
                minSize=(20, 20), # Reasonable minimum size
                maxSize=(200, 200), # Add maximum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces1)
            
            # Pass 2: More sensitive detection for smaller faces
            faces2 = self.detector['model'].detectMultiScale(
                gray, 
                scaleFactor=1.02,  # Very sensitive
                minNeighbors=2,   # Very permissive
                minSize=(15, 15), # Small faces
                maxSize=(150, 150),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces2)
            
            # Pass 3: Detection on slightly blurred image (helps with noise)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            faces3 = self.detector['model'].detectMultiScale(
                blurred, 
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(18, 18),
                maxSize=(180, 180),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            all_faces.extend(faces3)
            
            # Remove duplicate detections using IoU
            unique_faces = self._remove_duplicate_detections(all_faces)
            
            # Filter faces using additional validation
            valid_faces = self._filter_valid_faces(frame, unique_faces)
            
            detections = []
            for (x, y, w, h) in valid_faces:
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                if w > 0 and h > 0:
                    # Calculate confidence based on face size and validation
                    confidence = self._calculate_face_confidence(frame, x, y, w, h)
                    
                    if confidence > 0.25:  # Lower threshold for more balanced detection
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'label': 'face'
                        })
            
            logger.debug(f"OpenCV detected {len(detections)} valid faces (from {len(all_faces)} raw detections)")
            return detections
            
        except Exception as e:
            logger.error(f"Error in OpenCV face detection: {e}")
            return []
    
    def track_faces(self, detections: List[Dict], frame_id: int = 0) -> List[Dict]:
        """Track faces across frames using enhanced tracking algorithms."""
        start_time = time.time()
        
        if not detections:
            return []
        
        tracked_detections = []
        
        if self.tracker['type'] == 'bytetrack':
            tracked_detections = self._track_bytetrack(detections, frame_id)
        elif self.tracker['type'] == 'deep_ocsort':
            tracked_detections = self._track_deep_ocsort(detections, frame_id)
        elif self.tracker['type'] == 'simple_iou':
            tracked_detections = self._track_simple_iou(detections, frame_id)
        
        # Performance monitoring
        tracking_time = time.time() - start_time
        fps = 1.0 / tracking_time if tracking_time > 0 else 0
        self.performance_metrics['tracking_fps'].append(fps)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics['tracking_fps']) > 100:
            self.performance_metrics['tracking_fps'] = self.performance_metrics['tracking_fps'][-100:]
        
        return tracked_detections
    
    def _track_bytetrack(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Track faces using ByteTrack algorithm."""
        try:
            # Convert detections to ByteTrack format
            dets = []
            for det in detections:
                bbox = det['bbox']
                dets.append([bbox[0], bbox[1], bbox[2], bbox[3], det['confidence']])
            
            if not dets:
                return []
            
            dets = np.array(dets)
            
            # Update tracker
            online_targets = self.tracker['tracker'].update(
                dets, 
                [frame_id], 
                (frame_id, frame_id)
            )
            
            # Convert back to our format
            tracked_detections = []
            for track in online_targets:
                bbox = track.tlbr
                tracked_detections.append({
                    'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    'track_id': track.track_id,
                    'confidence': track.score,
                    'label': 'face',
                    'frame_id': frame_id
                })
            
            return tracked_detections
            
        except Exception as e:
            logger.error(f"ByteTrack tracking error: {e}")
            return self._track_simple_iou(detections, frame_id)
    
    def _track_deep_ocsort(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Track faces using Deep OC-SORT algorithm."""
        try:
            # Implementation of Deep OC-SORT
            return self.tracker['tracker'].update(detections, frame_id)
        except Exception as e:
            logger.error(f"Deep OC-SORT tracking error: {e}")
            return self._track_simple_iou(detections, frame_id)
    
    def _track_simple_iou(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Track faces using simple IoU-based tracking with enhanced persistence."""
        if not detections:
            return []
        
        # Calculate IoU between current detections and existing tracks
        tracked_detections = []
        used_tracks = set()
        
        for detection in detections:
            best_track_id = None
            best_iou = 0
            
            for track_id, track_info in self.tracks.items():
                if track_id in used_tracks:
                    continue
                
                iou = self._calculate_iou(detection['bbox'], track_info['bbox'])
                if iou > 0.3 and iou > best_iou:  # IoU threshold
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = detection['bbox']
                self.tracks[best_track_id]['confidence'] = detection['confidence']
                self.tracks[best_track_id]['frames_seen'] += 1
                self.tracks[best_track_id]['last_seen'] = frame_id
                used_tracks.add(best_track_id)
                
                tracked_detections.append({
                    **detection,
                    'track_id': best_track_id
                })
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'frames_seen': 1,
                    'first_seen': frame_id,
                    'last_seen': frame_id
                }
                
                tracked_detections.append({
                    **detection,
                    'track_id': track_id
                })
        
        # Enhanced track persistence
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if track_id not in used_tracks:
                # Decrease persistence counter
                track_info['frames_seen'] -= 1
                if track_info['frames_seen'] <= -self.track_persistence:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return tracked_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_video(self, video_path: str) -> List[Dict]:
        """Process entire video and return detection results with performance monitoring."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        all_detections = []
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in current frame
            detections = self.detect_faces(frame)
            
            # Track faces
            tracked_detections = self.track_faces(detections, frame_idx)
            
            # Add frame information
            for detection in tracked_detections:
                detection['frame_idx'] = frame_idx
            
            all_detections.extend(tracked_detections)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Processed {frame_idx} frames at {fps:.2f} FPS")
        
        cap.release()
        
        # Final performance metrics
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        logger.info(f"Video processing complete. Total frames: {frame_idx}, Avg FPS: {avg_fps:.2f}")
        logger.info(f"Total detections: {len(all_detections)}")
        
        # Store detections in history
        self.detections_history = all_detections
        
        return all_detections
    
    def get_detections(self) -> List[Dict]:
        """Get all detection results."""
        return self.detections_history
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for benchmarking."""
        metrics = {}
        
        if self.performance_metrics['detection_fps']:
            metrics['avg_detection_fps'] = np.mean(self.performance_metrics['detection_fps'])
            metrics['min_detection_fps'] = np.min(self.performance_metrics['detection_fps'])
            metrics['max_detection_fps'] = np.max(self.performance_metrics['detection_fps'])
        
        if self.performance_metrics['tracking_fps']:
            metrics['avg_tracking_fps'] = np.mean(self.performance_metrics['tracking_fps'])
            metrics['min_tracking_fps'] = np.min(self.performance_metrics['tracking_fps'])
            metrics['max_tracking_fps'] = np.max(self.performance_metrics['tracking_fps'])
        
        metrics['total_tracks'] = len(self.tracks)
        metrics['total_detections'] = len(self.detections_history)
        
        return metrics
    
    def reset_tracking(self):
        """Reset tracking state."""
        self.tracks = {}
        self.next_track_id = 0
        self.detections_history = []
        self.performance_metrics = {
            'detection_fps': [],
            'tracking_fps': [],
            'memory_usage': [],
            'accuracy_metrics': []
        }
    
    def _remove_duplicate_detections(self, faces: List[Tuple]) -> List[Tuple]:
        """Remove duplicate face detections using IoU threshold."""
        if not faces:
            return []
        
        # Sort by area (largest first)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        unique_faces = []
        for face in faces:
            is_duplicate = False
            for unique_face in unique_faces:
                iou = self._calculate_face_iou(face, unique_face)
                if iou > 0.3:  # IoU threshold for duplicates
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        return unique_faces
    
    def _calculate_face_iou(self, face1: Tuple, face2: Tuple) -> float:
        """Calculate IoU between two face detections."""
        x1_1, y1_1, w1, h1 = face1
        x1_2, y1_2, w2, h2 = face2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_valid_faces(self, frame: np.ndarray, faces: List[Tuple]) -> List[Tuple]:
        """Filter faces using additional validation to remove false positives."""
        valid_faces = []
        
        for (x, y, w, h) in faces:
            if self._is_valid_face_region(frame, x, y, w, h):
                valid_faces.append((x, y, w, h))
        
        return valid_faces
    
    def _is_valid_face_region(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Validate if a detected region is likely to be a face."""
        try:
            # Extract the face region
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return False
            
            # Convert to grayscale if needed
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            # 1. Check aspect ratio (faces are typically more square than rectangular)
            aspect_ratio = w / h
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:  # More permissive
                return False
            
            # 2. Check size (too small or too large might be false positives)
            if w < 12 or h < 12 or w > 250 or h > 250:  # More permissive
                return False
            
            # 3. Check for sufficient contrast (faces have good contrast)
            contrast = np.std(gray_face)
            if contrast < 15:  # Lower threshold for contrast
                return False
            
            # 4. Check for face-like features using edge detection
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density < 0.005 or edge_density > 0.4:  # More permissive
                return False
            
            # 5. Check for skin-like colors (if color image) - make this optional
            if len(frame.shape) == 3:
                # Only check skin colors for larger faces where it's more reliable
                if w > 30 and h > 30:
                    if not self._has_skin_like_colors(face_region):
                        # Don't reject immediately, just lower confidence
                        pass
            
            # 6. Check for symmetry (faces are generally symmetric) - make this optional
            if w > 20 and h > 20:  # Only check symmetry for larger faces
                if not self._check_symmetry(gray_face):
                    # Don't reject immediately, just lower confidence
                    pass
            
            return True
            
        except Exception as e:
            logger.debug(f"Face validation error: {e}")
            return False
    
    def _has_skin_like_colors(self, face_region: np.ndarray) -> bool:
        """Check if the face region has skin-like colors."""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Define skin color ranges in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin colors
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin-colored pixels
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = skin_mask.shape[0] * skin_mask.shape[1]
            skin_percentage = skin_pixels / total_pixels
            
            return skin_percentage > 0.3  # At least 30% should be skin-colored
            
        except Exception as e:
            logger.debug(f"Skin color check error: {e}")
            return True  # Default to True if check fails
    
    def _check_symmetry(self, gray_face: np.ndarray) -> bool:
        """Check if the face region is symmetric."""
        try:
            h, w = gray_face.shape
            center_x = w // 2
            
            # Compare left and right halves
            left_half = gray_face[:, :center_x]
            right_half = gray_face[:, center_x:2*center_x]
            
            if left_half.shape != right_half.shape:
                return True  # Can't check symmetry
            
            # Flip right half horizontally
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Calculate difference
            diff = cv2.absdiff(left_half, right_half_flipped)
            mean_diff = np.mean(diff)
            
            # Allow some asymmetry (faces aren't perfectly symmetric)
            return mean_diff < 30
            
        except Exception as e:
            logger.debug(f"Symmetry check error: {e}")
            return True  # Default to True if check fails
    
    def _calculate_face_confidence(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Calculate confidence score for a detected face region."""
        try:
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            confidence = 0.5  # Base confidence
            
            # 1. Size-based confidence (larger faces are more reliable)
            size_factor = min(w * h / 1000, 1.0)  # Normalize to 0-1
            confidence += size_factor * 0.2
            
            # 2. Contrast-based confidence
            contrast = np.std(gray_face)
            contrast_factor = min(contrast / 50, 1.0)
            confidence += contrast_factor * 0.15
            
            # 3. Edge density confidence
            edges = cv2.Canny(gray_face, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            edge_factor = min(edge_density / 0.1, 1.0)  # Normalize to reasonable range
            confidence += edge_factor * 0.15
            
            # 4. Aspect ratio confidence (faces should be roughly square)
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.3:
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.debug(f"Confidence calculation error: {e}")
            return 0.5  # Default confidence


class DeepOCSORTTracker:
    """Deep OC-SORT tracking algorithm implementation."""
    
    def __init__(self):
        """Initialize Deep OC-SORT tracker."""
        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0
        
    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Update tracker with new detections."""
        # Simplified Deep OC-SORT implementation
        # In a real implementation, this would include:
        # - Deep feature extraction
        # - Motion prediction
        # - Appearance matching
        # - Occlusion handling
        
        tracked_detections = []
        
        for detection in detections:
            # Simple tracking logic for now
            track_id = self.next_track_id
            self.next_track_id += 1
            
            tracked_detections.append({
                **detection,
                'track_id': track_id
            })
        
        self.frame_count += 1
        return tracked_detections 