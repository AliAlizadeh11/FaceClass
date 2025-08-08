"""
Face detection and tracking module for FaceClass project.
Supports multiple face detection models and tracking algorithms.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceTracker:
    """Face detection and tracking using various models."""
    
    def __init__(self, config):
        """Initialize face tracker with configuration."""
        self.config = config
        self.detection_model = config.get('face_detection.model', 'yolo')
        self.confidence_threshold = config.get('face_detection.confidence_threshold', 0.5)
        self.nms_threshold = config.get('face_detection.nms_threshold', 0.4)
        self.min_face_size = config.get('face_detection.min_face_size', 20)
        
        # Initialize detection model
        self.detector = self._load_detector()
        
        # Tracking state
        self.tracks = {}  # track_id -> track_info
        self.next_track_id = 0
        self.detections_history = []
    
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
        """Load RetinaFace detector."""
        try:
            from retinaface import RetinaFace
            return {'type': 'retinaface', 'model': RetinaFace}
        except ImportError:
            logger.warning("RetinaFace not available, using OpenCV")
            return self._load_opencv_detector()
    
    def _load_mtcnn_detector(self):
        """Load MTCNN detector."""
        try:
            from mtcnn import MTCNN
            detector = MTCNN()
            return {'type': 'mtcnn', 'model': detector}
        except ImportError:
            logger.warning("MTCNN not available, using OpenCV")
            return self._load_opencv_detector()
    
    def _load_opencv_detector(self):
        """Load OpenCV Haar cascade detector."""
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
            
            logger.info("OpenCV Haar cascade detector loaded successfully")
            return {'type': 'opencv', 'model': cascade}
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV detector: {e}")
            raise Exception(f"OpenCV detector initialization failed: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a single frame."""
        detections = []
        
        if self.detector['type'] == 'yolo':
            detections = self._detect_yolo(frame)
        elif self.detector['type'] == 'retinaface':
            detections = self._detect_retinaface(frame)
        elif self.detector['type'] == 'mtcnn':
            detections = self._detect_mtcnn(frame)
        elif self.detector['type'] == 'opencv':
            detections = self._detect_opencv(frame)
        
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
        """Detect faces using RetinaFace."""
        faces = self.detector['model'].detect(frame)
        detections = []
        
        if faces is not None:
            for face in faces:
                bbox = face['facial_area']
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                    'confidence': 0.9,  # RetinaFace doesn't provide confidence
                    'label': 'face'
                })
        
        return detections
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN."""
        faces = self.detector['model'].detect_faces(frame)
        detections = []
        
        for face in faces:
            bbox = face['box']
            confidence = face['confidence']
            if confidence > self.confidence_threshold:
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    'confidence': confidence,
                    'label': 'face'
                })
        
        return detections
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar cascade with improved sensitivity and filtering."""
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
    
    def track_faces(self, detections: List[Dict]) -> List[Dict]:
        """Track faces across frames using simple IoU-based tracking."""
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
                    'frames_seen': 1
                }
                
                tracked_detections.append({
                    **detection,
                    'track_id': track_id
                })
        
        # Remove tracks that haven't been seen recently
        tracks_to_remove = []
        for track_id, track_info in self.tracks.items():
            if track_id not in used_tracks:
                track_info['frames_seen'] -= 1
                if track_info['frames_seen'] <= 0:
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
        """Process entire video and return detection results."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        all_detections = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in current frame
            detections = self.detect_faces(frame)
            
            # Track faces
            tracked_detections = self.track_faces(detections)
            
            # Add frame information
            for detection in tracked_detections:
                detection['frame_idx'] = frame_idx
            
            all_detections.extend(tracked_detections)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames")
        
        cap.release()
        logger.info(f"Video processing complete. Total detections: {len(all_detections)}")
        
        # Store detections in history
        self.detections_history = all_detections
        
        return all_detections
    
    def get_detections(self) -> List[Dict]:
        """Get all detection results."""
        return self.detections_history
    
    def reset_tracking(self):
        """Reset tracking state."""
        self.tracks = {}
        self.next_track_id = 0
        self.detections_history = [] 
    
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
            
            # 7. Check for reasonable position (faces shouldn't be at the very edges)
            margin = 5  # More permissive
            if x < margin or y < margin or x + w > frame.shape[1] - margin or y + h > frame.shape[0] - margin:
                # Allow edge faces but be more strict
                if x < 2 or y < 2 or x + w > frame.shape[1] - 2 or y + h > frame.shape[0] - 2:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error in face validation: {e}")
            return False
    
    def _has_skin_like_colors(self, face_region: np.ndarray) -> bool:
        """Check if the region has skin-like colors."""
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Define skin color ranges in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin colors
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin-like pixels
            skin_pixels = np.sum(skin_mask > 0)
            total_pixels = face_region.shape[0] * face_region.shape[1]
            skin_percentage = skin_pixels / total_pixels
            
            # Return True if more than 10% of pixels are skin-like
            return skin_percentage > 0.1
            
        except Exception:
            return True  # If we can't check colors, assume it's valid
    
    def _check_symmetry(self, gray_face: np.ndarray) -> bool:
        """Check if the face region has reasonable symmetry."""
        try:
            # Check vertical symmetry
            height, width = gray_face.shape
            if width < 10:
                return True  # Too small to check symmetry
            
            # Compare left and right halves
            mid = width // 2
            left_half = gray_face[:, :mid]
            right_half = gray_face[:, -mid:]
            
            # Flip right half to compare with left
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Ensure same size for comparison
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            if min_width < 5:
                return True
            
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate correlation between left and right halves
            correlation = np.corrcoef(left_half.flatten(), right_half_flipped.flatten())[0, 1]
            
            # Return True if correlation is reasonable (not too low)
            return correlation > -0.5  # Allow some asymmetry
            
        except Exception:
            return True  # If we can't check symmetry, assume it's valid
    
    def _calculate_face_confidence(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Calculate confidence score for a detected face."""
        try:
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale if needed
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            confidence = 0.5  # Base confidence
            
            # 1. Size-based confidence (larger faces are more likely to be real)
            size_factor = min(1.0, (w * h) / 4000)  # Normalize by typical face size
            confidence += size_factor * 0.2
            
            # 2. Contrast-based confidence
            contrast = np.std(gray_face)
            contrast_factor = min(1.0, contrast / 50)
            confidence += contrast_factor * 0.15
            
            # 3. Aspect ratio confidence (faces are typically square-ish)
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.3:
                confidence += 0.1  # Good aspect ratio
            elif 0.5 <= aspect_ratio <= 1.5:
                confidence += 0.05  # Acceptable aspect ratio
            
            # 4. Position confidence (faces in center are more likely)
            center_x = x + w/2
            center_y = y + h/2
            frame_center_x = frame.shape[1] / 2
            frame_center_y = frame.shape[0] / 2
            
            distance_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
            max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
            position_factor = 1.0 - (distance_from_center / max_distance)
            confidence += position_factor * 0.1
            
            return min(0.95, confidence)
            
        except Exception:
            return 0.5  # Default confidence if calculation fails 