"""
Enhanced Face Detection Service for FaceClass project.
Implements advanced face detection with multiple models, preprocessing, and tracking.
Optimized for classroom scenarios with high accuracy for distant and small faces.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import json
from dataclasses import dataclass
import mediapipe as mp

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structured detection result with enhanced metadata."""
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    model: str
    frame_id: int
    track_id: Optional[int] = None
    face_size: Optional[int] = None
    quality_score: Optional[float] = None
    preprocessing_applied: List[str] = None

class EnhancedFaceDetectionService:
    """Enhanced face detection service with multiple models and preprocessing."""
    
    def __init__(self, config: Dict):
        """Initialize enhanced face detection service.
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        
        # Detection parameters - optimized for classroom scenarios
        self.confidence_threshold = config.get('face_detection.confidence_threshold', 0.3)  # Lowered for better recall
        self.nms_threshold = config.get('face_detection.nms_threshold', 0.3)  # Lowered for better overlap handling
        self.min_face_size = config.get('face_detection.min_face_size', 15)  # Lowered for distant faces
        self.max_faces = config.get('face_detection.max_faces', 100)  # Increased for classroom
        
        # Preprocessing parameters
        self.enable_denoising = config.get('face_detection.preprocessing.denoising', True)
        self.enable_contrast_enhancement = config.get('face_detection.preprocessing.contrast_enhancement', True)
        self.enable_super_resolution = config.get('face_detection.preprocessing.super_resolution', False)
        self.scale_factors = config.get('face_detection.preprocessing.scale_factors', [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
        
        # Multi-model ensemble parameters
        self.ensemble_models = config.get('face_detection.ensemble_models', ['yolo', 'mediapipe', 'mtcnn'])
        self.ensemble_voting = config.get('face_detection.ensemble_voting', True)
        self.ensemble_confidence_threshold = config.get('face_detection.ensemble_confidence_threshold', 0.2)
        
        # Tracking integration
        self.enable_tracking = config.get('face_detection.enable_tracking', True)
        self.track_persistence = config.get('face_detection.track_persistence', 30)
        
        # Performance monitoring
        self.performance_metrics = {
            'detection_fps': [],
            'total_faces_detected': 0,
            'frames_processed': 0,
            'detection_confidence_avg': [],
            'face_size_distribution': []
        }
        
        # Initialize detection models
        self.detectors = self._load_detectors()
        
        # Initialize tracking
        if self.enable_tracking:
            self.tracker = self._load_tracker()
            self.tracks = {}
            self.next_track_id = 0
        
        # Initialize MediaPipe
        self.mp_face_detection = None
        self._init_mediapipe()
        
        logger.info(f"Enhanced face detection service initialized with {len(self.detectors)} models")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face detection."""
        try:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.2  # Lowered for better recall
            )
            logger.info("MediaPipe face detection initialized successfully")
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            self.mp_face_detection = None
    
    def _load_detectors(self) -> Dict[str, Dict]:
        """Load multiple face detection models for ensemble detection."""
        detectors = {}
        
        # Load YOLO detector
        if 'yolo' in self.ensemble_models:
            detectors['yolo'] = self._load_yolo_detector()
        
        # Load MTCNN detector
        if 'mtcnn' in self.ensemble_models:
            detectors['mtcnn'] = self._load_mtcnn_detector()
        
        # Load OpenCV detector
        if 'opencv' in self.ensemble_models:
            detectors['opencv'] = self._load_opencv_detector()
        
        # MediaPipe is initialized separately
        
        if not detectors:
            logger.warning("No detectors loaded, falling back to OpenCV")
            detectors['opencv'] = self._load_opencv_detector()
        
        return detectors
    
    def _load_yolo_detector(self) -> Dict:
        """Load YOLOv8-based face detector."""
        try:
            from ultralytics import YOLO
            
            # Try to load face-specific model first
            model_path = Path(self.config.get('paths.models', 'models')) / 'face_detection' / 'yolov8n-face.pt'
            
            if model_path.exists():
                model = YOLO(str(model_path))
                logger.info(f"YOLOv8 face model loaded from {model_path}")
            else:
                # Use general YOLOv8n model
                model = YOLO('yolov8n.pt')
                logger.info("Using pre-trained YOLOv8n model for face detection")
            
            return {'type': 'yolo', 'model': model, 'loaded': True}
            
        except ImportError:
            logger.warning("Ultralytics not available")
            return {'type': 'yolo', 'model': None, 'loaded': False}
        except Exception as e:
            logger.warning(f"Failed to load YOLO detector: {e}")
            return {'type': 'yolo', 'model': None, 'loaded': False}
    
    def _load_mtcnn_detector(self) -> Dict:
        """Load MTCNN detector with optimized parameters."""
        try:
            from mtcnn import MTCNN
            detector = MTCNN(
                min_face_size=self.min_face_size,
                scale_factor=0.7,  # More sensitive
                factor=0.7,  # More sensitive
                thresholds=[0.6, 0.7, 0.7]  # Lowered thresholds
            )
            logger.info("MTCNN detector loaded successfully")
            return {'type': 'mtcnn', 'model': detector, 'loaded': True}
        except ImportError:
            logger.warning("MTCNN not available")
            return {'type': 'mtcnn', 'model': None, 'loaded': False}
        except Exception as e:
            logger.warning(f"Failed to load MTCNN detector: {e}")
            return {'type': 'mtcnn', 'model': None, 'loaded': False}
    
    def _load_opencv_detector(self) -> Dict:
        """Load OpenCV cascade detector with multiple cascade files."""
        try:
            cascades = {}
            
            # Load multiple cascade files for better detection
            cascade_files = [
                'haarcascade_frontalface_default.xml',
                'haarcascade_frontalface_alt.xml',
                'haarcascade_frontalface_alt2.xml'
            ]
            
            for cascade_file in cascade_files:
                cascade_path = cv2.data.haarcascades + cascade_file
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    cascades[cascade_file] = cascade
            
            if not cascades:
                logger.error("No OpenCV cascades loaded")
                return {'type': 'opencv', 'model': None, 'loaded': False}
            
            logger.info(f"OpenCV cascades loaded: {list(cascades.keys())}")
            return {'type': 'opencv', 'model': cascades, 'loaded': True}
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV detector: {e}")
            return {'type': 'opencv', 'model': None, 'loaded': False}
    
    def _load_tracker(self):
        """Load face tracking algorithm."""
        try:
            # Simple IoU-based tracker for now
            return {'type': 'simple_iou', 'tracker': None}
        except Exception as e:
            logger.warning(f"Tracker initialization failed: {e}")
            return {'type': 'simple_iou', 'tracker': None}
    
    def detect_faces_enhanced(self, frame: np.ndarray, frame_id: int = 0) -> List[DetectionResult]:
        """Enhanced face detection with preprocessing and ensemble detection.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_id: Frame identifier for tracking
            
        Returns:
            List of enhanced detection results
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided for face detection")
            return []
        
        start_time = time.time()
        
        try:
            # Apply preprocessing
            preprocessed_frames = self._apply_preprocessing(frame)
            
            # Run ensemble detection
            all_detections = []
            
            for scale_factor, processed_frame in preprocessed_frames:
                # Detect with each model
                for model_name, detector in self.detectors.items():
                    if detector['loaded']:
                        detections = self._detect_with_model(
                            detector, processed_frame, model_name, scale_factor
                        )
                        all_detections.extend(detections)
                
                # Detect with MediaPipe
                if self.mp_face_detection:
                    mediapipe_detections = self._detect_with_mediapipe(
                        processed_frame, scale_factor
                    )
                    all_detections.extend(mediapipe_detections)
            
            # Apply ensemble voting and NMS
            final_detections = self._apply_ensemble_voting(all_detections)
            
            # Update tracking
            if self.enable_tracking:
                final_detections = self._update_tracking(final_detections, frame_id)
            
            # Update performance metrics
            self._update_performance_metrics(len(final_detections), time.time() - start_time)
            
            return final_detections
            
        except Exception as e:
            logger.error(f"Error during enhanced face detection: {e}")
            return []
    
    def _apply_preprocessing(self, frame: np.ndarray) -> List[Tuple[float, np.ndarray]]:
        """Apply multiple preprocessing techniques to improve detection."""
        preprocessed_frames = []
        
        # Original frame
        preprocessed_frames.append((1.0, frame.copy()))
        
        # Multi-scale versions
        for scale in self.scale_factors:
            if scale != 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                preprocessed_frames.append((scale, scaled_frame))
        
        # Enhanced versions
        if self.enable_denoising:
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
            preprocessed_frames.append((1.0, denoised))
        
        if self.enable_contrast_enhancement:
            # CLAHE for better contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            preprocessed_frames.append((1.0, enhanced))
        
        return preprocessed_frames
    
    def _detect_with_model(self, detector: Dict, frame: np.ndarray, 
                          model_name: str, scale_factor: float) -> List[DetectionResult]:
        """Detect faces using a specific model."""
        try:
            if detector['type'] == 'yolo':
                return self._detect_yolo(detector, frame, model_name, scale_factor)
            elif detector['type'] == 'mtcnn':
                return self._detect_mtcnn(detector, frame, model_name, scale_factor)
            elif detector['type'] == 'opencv':
                return self._detect_opencv(detector, frame, model_name, scale_factor)
            else:
                return []
        except Exception as e:
            logger.error(f"Error with {model_name} detection: {e}")
            return []
    
    def _detect_yolo(self, detector: Dict, frame: np.ndarray, 
                     model_name: str, scale_factor: float) -> List[DetectionResult]:
        """Detect faces using YOLOv8."""
        try:
            results = detector['model'](frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if confidence >= self.ensemble_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Scale back to original coordinates
                            x1, y1, x2, y2 = x1 / scale_factor, y1 / scale_factor, x2 / scale_factor, y2 / scale_factor
                            
                            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                            face_size = min(bbox[2], bbox[3])
                            
                            detections.append(DetectionResult(
                                bbox=bbox,
                                confidence=confidence,
                                model=model_name,
                                frame_id=0,
                                face_size=face_size,
                                quality_score=self._calculate_face_quality(frame, bbox),
                                preprocessing_applied=[f"scale_{scale_factor}"]
                            ))
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def _detect_mtcnn(self, detector: Dict, frame: np.ndarray, 
                      model_name: str, scale_factor: float) -> List[DetectionResult]:
        """Detect faces using MTCNN."""
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector['model'].detect_faces(rgb_frame)
            
            detections = []
            for face in faces:
                confidence = face['confidence']
                
                if confidence >= self.ensemble_confidence_threshold:
                    bbox = face['box']
                    x, y, w, h = bbox
                    
                    # Scale back to original coordinates
                    x, y, w, h = x / scale_factor, y / scale_factor, w / scale_factor, h / scale_factor
                    
                    bbox = [int(x), int(y), int(w), int(h)]
                    face_size = min(w, h)
                    
                    detections.append(DetectionResult(
                        bbox=bbox,
                        confidence=confidence,
                        model=model_name,
                        frame_id=0,
                        face_size=face_size,
                        quality_score=self._calculate_face_quality(frame, bbox),
                        preprocessing_applied=[f"scale_{scale_factor}"]
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []
    
    def _detect_opencv(self, detector: Dict, frame: np.ndarray, 
                       model_name: str, scale_factor: float) -> List[DetectionResult]:
        """Detect faces using OpenCV with multiple cascade files."""
        try:
            detections = []
            
            for cascade_name, cascade in detector['model'].items():
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                # Multiple detection passes with different parameters
                detection_params = [
                    {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (15, 15)},
                    {'scaleFactor': 1.02, 'minNeighbors': 2, 'minSize': (12, 12)},
                    {'scaleFactor': 1.08, 'minNeighbors': 4, 'minSize': (18, 18)}
                ]
                
                for params in detection_params:
                    faces = cascade.detectMultiScale(
                        gray,
                        scaleFactor=params['scaleFactor'],
                        minNeighbors=params['minNeighbors'],
                        minSize=params['minSize'],
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    for (x, y, w, h) in faces:
                        # Scale back to original coordinates
                        x, y, w, h = x / scale_factor, y / scale_factor, w / scale_factor, h / scale_factor
                        
                        bbox = [int(x), int(y), int(w), int(h)]
                        face_size = min(w, h)
                        
                        # OpenCV doesn't provide confidence, estimate based on detection method
                        confidence = 0.6 + (0.2 * (1.0 / params['scaleFactor']))
                        
                        detections.append(DetectionResult(
                            bbox=bbox,
                            confidence=confidence,
                            model=f"{model_name}_{cascade_name}",
                            frame_id=0,
                            face_size=face_size,
                            quality_score=self._calculate_face_quality(frame, bbox),
                            preprocessing_applied=[f"scale_{scale_factor}", f"params_{params['scaleFactor']}"]
                        ))
            
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return []
    
    def _detect_with_mediapipe(self, frame: np.ndarray, scale_factor: float) -> List[DetectionResult]:
        """Detect faces using MediaPipe."""
        try:
            if self.mp_face_detection is None:
                return []
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_frame)
            
            detections = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    confidence = detection.score[0]
                    
                    if confidence >= self.ensemble_confidence_threshold:
                        h, w = frame.shape[:2]
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Scale back to original coordinates
                        x, y, width, height = x / scale_factor, y / scale_factor, width / scale_factor, height / scale_factor
                        
                        bbox = [int(x), int(y), int(width), int(height)]
                        face_size = min(width, height)
                        
                        detections.append(DetectionResult(
                            bbox=bbox,
                            confidence=confidence,
                            model='mediapipe',
                            frame_id=0,
                            face_size=face_size,
                            quality_score=self._calculate_face_quality(frame, bbox),
                            preprocessing_applied=[f"scale_{scale_factor}"]
                        ))
            
            return detections
            
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return []
    
    def _calculate_face_quality(self, frame: np.ndarray, bbox: List[int]) -> float:
        """Calculate face quality score based on various factors."""
        try:
            x, y, w, h = bbox
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region
            
            # Calculate quality metrics
            quality_score = 0.0
            
            # Size-based quality (larger faces are generally better)
            size_score = min(1.0, (w * h) / (64 * 64))  # Normalize to 64x64
            quality_score += size_score * 0.3
            
            # Sharpness-based quality
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)  # Normalize to typical values
            quality_score += sharpness_score * 0.4
            
            # Contrast-based quality
            contrast_score = gray.std() / 128.0  # Normalize to 0-1
            quality_score += contrast_score * 0.3
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.debug(f"Face quality calculation error: {e}")
            return 0.5  # Default quality score
    
    def _apply_ensemble_voting(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply ensemble voting to combine detections from multiple models."""
        if not self.ensemble_voting or len(detections) <= 1:
            return detections
        
        # Group detections by spatial proximity
        detection_groups = self._group_detections_by_proximity(detections)
        
        # Vote on each group
        final_detections = []
        for group in detection_groups:
            if len(group) >= 2:  # At least 2 models agree
                # Select the best detection from the group
                best_detection = self._select_best_detection(group)
                final_detections.append(best_detection)
            else:
                # Single detection, keep if confident enough
                if group[0].confidence >= self.confidence_threshold:
                    final_detections.append(group[0])
        
        # Apply NMS to remove overlapping detections
        final_detections = self._apply_enhanced_nms(final_detections)
        
        return final_detections
    
    def _group_detections_by_proximity(self, detections: List[DetectionResult], 
                                     iou_threshold: float = 0.3) -> List[List[DetectionResult]]:
        """Group detections that are spatially close to each other."""
        if len(detections) <= 1:
            return [detections]
        
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                if self._calculate_iou(det1.bbox, det2.bbox) > iou_threshold:
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _select_best_detection(self, group: List[DetectionResult]) -> DetectionResult:
        """Select the best detection from a group based on multiple criteria."""
        if len(group) == 1:
            return group[0]
        
        # Score each detection
        scored_detections = []
        for det in group:
            score = 0.0
            
            # Confidence score
            score += det.confidence * 0.4
            
            # Quality score
            if det.quality_score:
                score += det.quality_score * 0.3
            
            # Size score (prefer medium-sized faces)
            if det.face_size:
                size_score = 1.0 - abs(det.face_size - 64) / 64.0  # Peak at 64px
                score += max(0, size_score) * 0.2
            
            # Model reliability score
            model_scores = {'yolo': 0.9, 'mediapipe': 0.8, 'mtcnn': 0.7, 'opencv': 0.6}
            model_score = model_scores.get(det.model.split('_')[0], 0.5)
            score += model_score * 0.1
            
            scored_detections.append((det, score))
        
        # Return detection with highest score
        return max(scored_detections, key=lambda x: x[1])[0]
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_enhanced_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply enhanced Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        kept = []
        for det in sorted_detections:
            should_keep = True
            
            for kept_det in kept:
                if self._calculate_iou(det.bbox, kept_det.bbox) > self.nms_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(det)
        
        return kept
    
    def _update_tracking(self, detections: List[DetectionResult], frame_id: int) -> List[DetectionResult]:
        """Update face tracking across frames."""
        if not self.enable_tracking:
            return detections
        
        # Simple IoU-based tracking
        current_tracks = {}
        
        for det in detections:
            best_track_id = None
            best_iou = 0.0
            
            # Find best matching track
            for track_id, track_info in self.tracks.items():
                if track_info['last_frame'] < frame_id - self.track_persistence:
                    continue  # Track expired
                
                iou = self._calculate_iou(det.bbox, track_info['bbox'])
                if iou > best_iou and iou > 0.3:  # IoU threshold
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                det.track_id = best_track_id
                self.tracks[best_track_id].update({
                    'bbox': det.bbox,
                    'last_frame': frame_id,
                    'confidence': det.confidence
                })
                current_tracks[best_track_id] = self.tracks[best_track_id]
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                det.track_id = track_id
                
                self.tracks[track_id] = {
                    'bbox': det.bbox,
                    'first_frame': frame_id,
                    'last_frame': frame_id,
                    'confidence': det.confidence,
                    'detection_count': 1
                }
                current_tracks[track_id] = self.tracks[track_id]
        
        # Update tracks dictionary
        self.tracks = current_tracks
        
        return detections
    
    def _update_performance_metrics(self, num_faces: int, processing_time: float):
        """Update performance monitoring metrics."""
        self.performance_metrics['total_faces_detected'] += num_faces
        self.performance_metrics['frames_processed'] += 1
        
        if processing_time > 0:
            fps = 1.0 / processing_time
            self.performance_metrics['detection_fps'].append(fps)
            
            # Keep only last 100 FPS measurements
            if len(self.performance_metrics['detection_fps']) > 100:
                self.performance_metrics['detection_fps'] = self.performance_metrics['detection_fps'][-100:]
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        fps_values = self.performance_metrics['detection_fps']
        avg_fps = np.mean(fps_values) if fps_values else 0.0
        
        return {
            'total_frames_processed': self.performance_metrics['frames_processed'],
            'total_faces_detected': self.performance_metrics['total_faces_detected'],
            'average_detection_fps': avg_fps,
            'detection_models_loaded': len([d for d in self.detectors.values() if d['loaded']]),
            'tracking_enabled': self.enable_tracking,
            'active_tracks': len(self.tracks),
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'min_face_size': self.min_face_size,
                'ensemble_models': self.ensemble_models,
                'preprocessing_enabled': {
                    'denoising': self.enable_denoising,
                    'contrast_enhancement': self.enable_contrast_enhancement,
                    'super_resolution': self.enable_super_resolution
                }
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'detection_fps': [],
            'total_faces_detected': 0,
            'frames_processed': 0,
            'detection_confidence_avg': [],
            'face_size_distribution': []
        }
        if self.enable_tracking:
            self.tracks = {}
            self.next_track_id = 0
