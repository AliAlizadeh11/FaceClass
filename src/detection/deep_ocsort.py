"""
Deep OC-SORT Tracking Algorithm Implementation for Team 1
Enhanced tracking with deep features, motion prediction, and occlusion handling
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class DeepOCSORTTracker:
    """
    Deep OC-SORT (Observation-Centric SORT) tracker implementation.
    
    Features:
    - Deep feature extraction for appearance matching
    - Motion prediction using Kalman filtering
    - Occlusion handling and track persistence
    - Multi-camera support
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Deep OC-SORT tracker."""
        self.config = config or {}
        
        # Tracking parameters
        self.max_age = self.config.get('max_age', 30)
        self.min_hits = self.config.get('min_hits', 3)
        self.iou_threshold = self.config.get('iou_threshold', 0.3)
        self.feature_similarity_threshold = self.config.get('feature_similarity_threshold', 0.7)
        
        # Track management
        self.tracks = {}  # track_id -> Track object
        self.next_track_id = 0
        self.frame_count = 0
        
        # Feature extraction
        self.feature_extractor = self._load_feature_extractor()
        
        # Motion prediction
        self.kalman_filters = {}
        
        # Multi-camera support
        self.multi_camera = self.config.get('multi_camera', False)
        self.camera_tracks = {} if self.multi_camera else None
        
        # Performance monitoring
        self.performance_metrics = {
            'tracking_fps': [],
            'total_tracks': 0,
            'id_switches': 0,
            'fragmentation': 0
        }
    
    def _load_feature_extractor(self):
        """Load deep feature extractor for appearance matching."""
        try:
            # Try to load a pre-trained feature extractor
            # This could be ResNet, VGG, or any other CNN-based feature extractor
            import torch
            import torchvision.models as models
            
            # Use ResNet-18 as feature extractor (remove classification layer)
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
            model.eval()
            
            logger.info("ResNet-18 feature extractor loaded successfully")
            return {'type': 'resnet18', 'model': model}
            
        except ImportError:
            logger.warning("PyTorch not available, using simple feature extraction")
            return {'type': 'simple', 'model': None}
        except Exception as e:
            logger.warning(f"Failed to load deep feature extractor: {e}")
            return {'type': 'simple', 'model': None}
    
    def update(self, detections: List[Dict], frame_id: int, camera_id: int = 0) -> List[Dict]:
        """Update tracker with new detections."""
        start_time = time.time()
        self.frame_count += 1
        
        # Extract features from detections
        detection_features = self._extract_features(detections)
        
        # Predict new locations of existing tracks
        self._predict_tracks()
        
        # Associate detections with existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, detection_features
        )
        
        # Update matched tracks
        for detection_idx, track_id in matched_pairs:
            self._update_track(track_id, detections[detection_idx], detection_features[detection_idx])
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._create_track(detections[detection_idx], detection_features[detection_idx], camera_id)
        
        # Handle unmatched tracks (occlusion, leaving scene)
        for track_id in unmatched_tracks:
            self._handle_unmatched_track(track_id)
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        # Performance monitoring
        tracking_time = time.time() - start_time
        fps = 1.0 / tracking_time if tracking_time > 0 else 0
        self.performance_metrics['tracking_fps'].append(fps)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics['tracking_fps']) > 100:
            self.performance_metrics['tracking_fps'] = self.performance_metrics['tracking_fps'][-100:]
        
        # Return active tracks
        return self._get_active_tracks()
    
    def _extract_features(self, detections: List[Dict]) -> List[np.ndarray]:
        """Extract deep features from detections for appearance matching."""
        features = []
        
        for detection in detections:
            if self.feature_extractor['type'] == 'resnet18':
                feature = self._extract_deep_features(detection)
            else:
                feature = self._extract_simple_features(detection)
            features.append(feature)
        
        return features
    
    def _extract_deep_features(self, detection: Dict) -> np.ndarray:
        """Extract deep features using ResNet-18."""
        try:
            import torch
            from PIL import Image
            
            # Get bounding box
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract face region (assuming we have the full frame)
            # In practice, you'd need to pass the frame to this method
            # For now, return a placeholder feature
            feature = np.random.randn(512)  # Placeholder for 512-dimensional feature
            return feature
            
        except Exception as e:
            logger.warning(f"Deep feature extraction failed: {e}")
            return self._extract_simple_features(detection)
    
    def _extract_simple_features(self, detection: Dict) -> np.ndarray:
        """Extract simple features (color histogram, HOG, etc.)."""
        # Simple feature extraction based on bounding box properties
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Basic features: position, size, aspect ratio
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        aspect_ratio = width / height if height > 0 else 1.0
        area = width * height
        
        # Normalize features
        features = np.array([
            center_x / 1920,  # Normalize by typical frame width
            center_y / 1080,  # Normalize by typical frame height
            width / 1920,
            height / 1080,
            aspect_ratio,
            area / (1920 * 1080)
        ])
        
        return features
    
    def _predict_tracks(self):
        """Predict new locations of existing tracks using Kalman filtering."""
        for track_id, track in self.tracks.items():
            if track.is_active():
                # Predict next state
                predicted_state = track.predict()
                
                # Update track with prediction
                track.update_prediction(predicted_state)
    
    def _associate_detections_to_tracks(self, detections: List[Dict], 
                                      detection_features: List[np.ndarray]) -> Tuple[List, List, List]:
        """Associate detections with existing tracks using multiple cues."""
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(detections, detection_features)
        
        # Hungarian algorithm for optimal assignment
        matched_pairs, unmatched_detections, unmatched_tracks = self._hungarian_assignment(
            similarity_matrix
        )
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _calculate_similarity_matrix(self, detections: List[Dict], 
                                   detection_features: List[np.ndarray]) -> np.ndarray:
        """Calculate similarity matrix between detections and tracks."""
        num_detections = len(detections)
        num_tracks = len(self.tracks)
        
        similarity_matrix = np.zeros((num_detections, num_tracks))
        
        for i, detection in enumerate(detections):
            for j, (track_id, track) in enumerate(self.tracks.items()):
                if track.is_active():
                    # IoU similarity
                    iou_similarity = self._calculate_iou(detection['bbox'], track.get_bbox())
                    
                    # Feature similarity
                    feature_similarity = self._calculate_feature_similarity(
                        detection_features[i], track.get_features()
                    )
                    
                    # Motion similarity (predicted vs actual position)
                    motion_similarity = self._calculate_motion_similarity(detection, track)
                    
                    # Combined similarity score
                    similarity = (
                        0.4 * iou_similarity +
                        0.4 * feature_similarity +
                        0.2 * motion_similarity
                    )
                    
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
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
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors."""
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        # Normalize features
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return max(0.0, similarity)  # Ensure non-negative
    
    def _calculate_motion_similarity(self, detection: Dict, track) -> float:
        """Calculate motion similarity between detection and predicted track position."""
        try:
            predicted_bbox = track.get_predicted_bbox()
            if predicted_bbox is None:
                return 0.0
            
            # Calculate distance between predicted and actual position
            actual_center = np.array([
                (detection['bbox'][0] + detection['bbox'][2]) / 2,
                (detection['bbox'][1] + detection['bbox'][3]) / 2
            ])
            
            predicted_center = np.array([
                (predicted_bbox[0] + predicted_bbox[2]) / 2,
                (predicted_bbox[1] + predicted_bbox[3]) / 2
            ])
            
            distance = np.linalg.norm(actual_center - predicted_center)
            
            # Convert to similarity (closer = higher similarity)
            max_distance = 100  # Maximum expected distance
            similarity = max(0.0, 1.0 - distance / max_distance)
            
            return similarity
            
        except Exception as e:
            logger.debug(f"Motion similarity calculation failed: {e}")
            return 0.0
    
    def _hungarian_assignment(self, similarity_matrix: np.ndarray) -> Tuple[List, List, List]:
        """Use Hungarian algorithm for optimal assignment."""
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Convert similarity to cost (higher similarity = lower cost)
            cost_matrix = 1.0 - similarity_matrix
            
            # Apply Hungarian algorithm
            detection_indices, track_indices = linear_sum_assignment(cost_matrix)
            
            # Filter assignments based on similarity threshold
            matched_pairs = []
            unmatched_detections = list(range(similarity_matrix.shape[0]))
            unmatched_tracks = list(range(similarity_matrix.shape[1]))
            
            for det_idx, track_idx in zip(detection_indices, track_indices):
                if similarity_matrix[det_idx, track_idx] >= self.iou_threshold:
                    matched_pairs.append((det_idx, track_idx))
                    unmatched_detections.remove(det_idx)
                    unmatched_tracks.remove(track_idx)
            
            return matched_pairs, unmatched_detections, unmatched_tracks
            
        except ImportError:
            logger.warning("SciPy not available, using greedy assignment")
            return self._greedy_assignment(similarity_matrix)
    
    def _greedy_assignment(self, similarity_matrix: np.ndarray) -> Tuple[List, List, List]:
        """Greedy assignment algorithm as fallback."""
        matched_pairs = []
        unmatched_detections = list(range(similarity_matrix.shape[0]))
        unmatched_tracks = list(range(similarity_matrix.shape[1]))
        
        # Sort by similarity (highest first)
        similarities = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                similarities.append((similarity_matrix[i, j], i, j))
        
        similarities.sort(reverse=True)
        
        for similarity, det_idx, track_idx in similarities:
            if (det_idx in unmatched_detections and 
                track_idx in unmatched_tracks and 
                similarity >= self.iou_threshold):
                
                matched_pairs.append((det_idx, track_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _update_track(self, track_id: int, detection: Dict, features: np.ndarray):
        """Update existing track with new detection."""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.update(detection, features)
    
    def _create_track(self, detection: Dict, features: np.ndarray, camera_id: int):
        """Create new track for unmatched detection."""
        track = Track(
            track_id=self.next_track_id,
            detection=detection,
            features=features,
            camera_id=camera_id,
            config=self.config
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        
        # Initialize Kalman filter
        self._initialize_kalman_filter(track)
    
    def _handle_unmatched_track(self, track_id: int):
        """Handle track that wasn't matched to any detection."""
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.mark_missed()
    
    def _remove_dead_tracks(self):
        """Remove tracks that have exceeded max_age."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if not track.is_active() and track.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _initialize_kalman_filter(self, track):
        """Initialize Kalman filter for motion prediction."""
        try:
            # Simple Kalman filter for 2D position and velocity
            bbox = track.get_bbox()
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # State: [x, y, vx, vy, w, h]
            initial_state = np.array([center_x, center_y, 0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1]])
            
            # Initialize Kalman filter
            kalman = cv2.KalmanFilter(6, 4)  # 6 states, 4 measurements
            
            # State transition matrix
            kalman.transitionMatrix = np.array([
                [1, 0, 1, 0, 0, 0],  # x = x + vx
                [0, 1, 0, 1, 0, 0],  # y = y + vy
                [0, 0, 1, 0, 0, 0],  # vx = vx
                [0, 0, 0, 1, 0, 0],  # vy = vy
                [0, 0, 0, 0, 1, 0],  # w = w
                [0, 0, 0, 0, 0, 1]   # h = h
            ], np.float32)
            
            # Measurement matrix
            kalman.measurementMatrix = np.array([
                [1, 0, 0, 0, 0, 0],  # measure x
                [0, 1, 0, 0, 0, 0],  # measure y
                [0, 0, 0, 0, 1, 0],  # measure w
                [0, 0, 0, 0, 0, 1]   # measure h
            ], np.float32)
            
            # Initialize state
            kalman.statePre = initial_state.astype(np.float32)
            kalman.statePost = initial_state.astype(np.float32)
            
            # Store Kalman filter
            self.kalman_filters[track.track_id] = kalman
            
        except Exception as e:
            logger.warning(f"Failed to initialize Kalman filter: {e}")
    
    def _get_active_tracks(self) -> List[Dict]:
        """Get list of active tracks in detection format."""
        active_tracks = []
        
        for track_id, track in self.tracks.items():
            if track.is_active():
                track_info = track.get_track_info()
                active_tracks.append(track_info)
        
        return active_tracks
    
    def get_performance_metrics(self) -> Dict:
        """Get tracking performance metrics."""
        metrics = self.performance_metrics.copy()
        
        if metrics['tracking_fps']:
            metrics['avg_tracking_fps'] = np.mean(metrics['tracking_fps'])
            metrics['min_tracking_fps'] = np.min(metrics['tracking_fps'])
            metrics['max_tracking_fps'] = np.max(metrics['tracking_fps'])
        
        metrics['current_tracks'] = len(self.tracks)
        metrics['total_tracks_created'] = self.next_track_id
        
        return metrics
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0
        self.kalman_filters = {}
        self.performance_metrics = {
            'tracking_fps': [],
            'total_tracks': 0,
            'id_switches': 0,
            'fragmentation': 0
        }


class Track:
    """Individual track object for Deep OC-SORT."""
    
    def __init__(self, track_id: int, detection: Dict, features: np.ndarray, 
                 camera_id: int, config: Dict):
        """Initialize new track."""
        self.track_id = track_id
        self.camera_id = camera_id
        self.config = config
        
        # Detection history
        self.detections = deque(maxlen=30)
        self.features = deque(maxlen=30)
        
        # Current state
        self.current_detection = detection
        self.current_features = features
        self.predicted_bbox = None
        
        # Tracking state
        self.hits = 1
        self.time_since_update = 0
        self.total_frames = 1
        
        # Add initial detection
        self.detections.append(detection)
        self.features.append(features)
    
    def update(self, detection: Dict, features: np.ndarray):
        """Update track with new detection."""
        self.current_detection = detection
        self.current_features = features
        self.detections.append(detection)
        self.features.append(features)
        
        self.hits += 1
        self.time_since_update = 0
        self.total_frames += 1
        
        # Clear prediction
        self.predicted_bbox = None
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
    
    def is_active(self) -> bool:
        """Check if track is active."""
        return self.hits >= self.config.get('min_hits', 3)
    
    def get_bbox(self) -> List[int]:
        """Get current bounding box."""
        return self.current_detection['bbox']
    
    def get_features(self) -> np.ndarray:
        """Get current features."""
        return self.current_features
    
    def get_predicted_bbox(self) -> Optional[List[int]]:
        """Get predicted bounding box."""
        return self.predicted_bbox
    
    def update_prediction(self, predicted_bbox: List[int]):
        """Update predicted bounding box."""
        self.predicted_bbox = predicted_bbox
    
    def get_track_info(self) -> Dict:
        """Get track information in detection format."""
        return {
            'bbox': self.current_detection['bbox'],
            'confidence': self.current_detection.get('confidence', 0.9),
            'track_id': self.track_id,
            'label': 'face',
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'camera_id': self.camera_id
        }
    
    def predict(self) -> List[int]:
        """Predict next position (placeholder for Kalman filter integration)."""
        # Simple prediction based on velocity
        if len(self.detections) >= 2:
            prev_bbox = self.detections[-2]['bbox']
            curr_bbox = self.detections[-1]['bbox']
            
            # Calculate velocity
            vx = curr_bbox[0] - prev_bbox[0]
            vy = curr_bbox[1] - prev_bbox[1]
            
            # Predict next position
            predicted_bbox = [
                curr_bbox[0] + vx,
                curr_bbox[1] + vy,
                curr_bbox[2] + vx,
                curr_bbox[3] + vy
            ]
            
            return predicted_bbox
        
        return self.current_detection['bbox']
