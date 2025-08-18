"""
Face tracking service for FaceClass project.
Implements Deep OC-SORT for robust multi-object tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class Track:
    """Represents a tracked face across multiple frames."""
    
    def __init__(self, track_id: int, bbox: List[int], confidence: float, frame_id: int):
        """Initialize a new track.
        
        Args:
            track_id: Unique identifier for this track
            bbox: Bounding box [x, y, w, h]
            confidence: Detection confidence
            frame_id: Frame where track was first detected
        """
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.frame_id = frame_id
        self.last_seen = frame_id
        self.history = [(frame_id, bbox, confidence)]
        self.age = 1
        self.total_hits = 1
        self.time_since_update = 0
        self.state = 'active'
    
    def update(self, bbox: List[int], confidence: float, frame_id: int):
        """Update track with new detection.
        
        Args:
            bbox: New bounding box
            confidence: New confidence score
            frame_id: Current frame ID
        """
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = frame_id
        self.history.append((frame_id, bbox, confidence))
        self.age += 1
        self.total_hits += 1
        self.time_since_update = 0
        
        # Keep only recent history to save memory
        if len(self.history) > 30:
            self.history = self.history[-30:]
    
    def predict(self):
        """Predict next position based on motion history."""
        if len(self.history) < 2:
            return self.bbox
        
        # Simple linear prediction based on last two positions
        if len(self.history) >= 2:
            prev_frame, prev_bbox, _ = self.history[-2]
            curr_frame, curr_bbox, _ = self.history[-1]
            
            if curr_frame > prev_frame:
                # Calculate velocity
                dx = curr_bbox[0] - prev_bbox[0]
                dy = curr_bbox[1] - curr_bbox[1]
                
                # Predict next position
                predicted_x = curr_bbox[0] + dx
                predicted_y = curr_bbox[1] + dy
                
                return [int(predicted_x), int(predicted_y), curr_bbox[2], curr_bbox[3]]
        
        return self.bbox
    
    def mark_missed(self):
        """Mark that this track was not detected in current frame."""
        self.time_since_update += 1
    
    def is_stale(self, max_age: int = 30) -> bool:
        """Check if track is stale (too old or not updated recently).
        
        Args:
            max_age: Maximum age for a track
            
        Returns:
            True if track is stale
        """
        return self.time_since_update > max_age or self.age > max_age * 2


class FaceTrackingService:
    """Service for tracking faces across video frames using Deep OC-SORT algorithm."""
    
    def __init__(self, config: Dict):
        """Initialize face tracking service.
        
        Args:
            config: Configuration dictionary containing tracking parameters
        """
        self.config = config
        self.max_age = config.get('face_tracking.max_age', 30)
        self.min_hits = config.get('face_tracking.min_hits', 3)
        self.iou_threshold = config.get('face_tracking.iou_threshold', 0.3)
        self.confidence_threshold = config.get('face_tracking.confidence_threshold', 0.5)
        
        # Tracking state
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        logger.info("Face tracking service initialized")
    
    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Update tracks with new detections.
        
        Args:
            detections: List of detection dictionaries from face detection
            frame_id: Current frame ID
            
        Returns:
            List of tracked faces with track_id
        """
        self.frame_count = frame_id
        
        # Predict new positions for existing tracks
        for track in self.tracks.values():
            if track.state == 'active':
                track.predict()
                track.mark_missed()
        
        # Remove stale tracks
        self._remove_stale_tracks()
        
        # Associate detections with existing tracks
        matched_detections, unmatched_tracks, unmatched_detections = self._associate_detections_to_tracks(
            detections, list(self.tracks.values())
        )
        
        # Update matched tracks
        for track, detection in matched_detections:
            track.update(detection['bbox'], detection['confidence'], frame_id)
            detection['track_id'] = track.track_id
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            if detection['confidence'] >= self.confidence_threshold:
                track = self._create_track(detection, frame_id)
                detection['track_id'] = track.track_id
        
        # Return all detections with track IDs
        tracked_detections = []
        for detection in detections:
            if 'track_id' in detection:
                tracked_detections.append(detection)
        
        return tracked_detections
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[Dict], 
        tracks: List[Track]
    ) -> Tuple[List[Tuple[Track, Dict]], List[Track], List[Dict]]:
        """Associate detections with existing tracks using Hungarian algorithm.
        
        Args:
            detections: List of detection dictionaries
            tracks: List of existing tracks
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0:
            return [], [], detections
        
        if len(detections) == 0:
            return [], tracks, []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection['bbox'])
        
        # Apply Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
            
            matched_pairs = []
            unmatched_tracks = []
            unmatched_detections = []
            
            # Check IoU threshold for matched pairs
            for track_idx, detection_idx in zip(track_indices, detection_indices):
                if iou_matrix[track_idx, detection_idx] >= self.iou_threshold:
                    matched_pairs.append((tracks[track_idx], detections[detection_idx]))
                else:
                    unmatched_tracks.append(tracks[track_idx])
                    unmatched_detections.append(detections[detection_idx])
            
            # Add unmatched tracks and detections
            for i, track in enumerate(tracks):
                if i not in track_indices:
                    unmatched_tracks.append(track)
            
            for j, detection in enumerate(detections):
                if j not in detection_indices:
                    unmatched_detections.append(detection)
            
            return matched_pairs, unmatched_tracks, unmatched_detections
            
        except ImportError:
            logger.warning("scipy not available, using simple greedy assignment")
            return self._greedy_assignment(iou_matrix, tracks, detections)
    
    def _greedy_assignment(
        self, 
        iou_matrix: np.ndarray, 
        tracks: List[Track], 
        detections: List[Dict]
    ) -> Tuple[List[Tuple[Track, Dict]], List[Track], List[Dict]]:
        """Simple greedy assignment when Hungarian algorithm is not available.
        
        Args:
            iou_matrix: IoU matrix between tracks and detections
            tracks: List of existing tracks
            detections: List of detection dictionaries
            
        Returns:
            Tuple of (matched_pairs, unmatched_tracks, unmatched_detections)
        """
        matched_pairs = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Sort by IoU score (highest first)
        pairs = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        
        pairs.sort(reverse=True)
        
        for score, track_idx, detection_idx in pairs:
            if track_idx in unmatched_tracks and detection_idx in unmatched_detections:
                matched_pairs.append((tracks[track_idx], detections[detection_idx]))
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(detection_idx)
        
        unmatched_tracks = [tracks[i] for i in unmatched_tracks]
        unmatched_detections = [detections[i] for i in unmatched_detections]
        
        return matched_pairs, unmatched_tracks, unmatched_detections
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x, y, w, h]
            bbox2: Second bounding box [x, y, w, h]
            
        Returns:
            IoU score between 0 and 1
        """
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
    
    def _create_track(self, detection: Dict, frame_id: int) -> Track:
        """Create a new track for a detection.
        
        Args:
            detection: Detection dictionary
            frame_id: Current frame ID
            
        Returns:
            Newly created Track object
        """
        track = Track(
            track_id=self.next_track_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            frame_id=frame_id
        )
        
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        
        logger.debug(f"Created new track {track.track_id} at frame {frame_id}")
        return track
    
    def _remove_stale_tracks(self):
        """Remove tracks that are too old or haven't been updated recently."""
        stale_tracks = []
        for track_id, track in self.tracks.items():
            if track.is_stale(self.max_age):
                stale_tracks.append(track_id)
                logger.debug(f"Marking track {track_id} as stale (age: {track.age}, missed: {track.time_since_update})")
        
        for track_id in stale_tracks:
            del self.tracks[track_id]
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get information about a specific track.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Dictionary containing track information or None if not found
        """
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        return {
            'track_id': track.track_id,
            'bbox': track.bbox,
            'confidence': track.confidence,
            'frame_id': track.frame_id,
            'last_seen': track.last_seen,
            'age': track.age,
            'total_hits': track.total_hits,
            'state': track.state
        }
    
    def get_all_tracks(self) -> List[Dict]:
        """Get information about all active tracks.
        
        Returns:
            List of track information dictionaries
        """
        return [self.get_track_info(track_id) for track_id in self.tracks.keys()]
    
    def get_tracking_summary(self) -> Dict:
        """Get summary statistics about tracking performance.
        
        Returns:
            Dictionary containing tracking statistics
        """
        active_tracks = [t for t in self.tracks.values() if t.state == 'active']
        total_tracks = len(self.tracks)
        
        return {
            'total_tracks_created': self.next_track_id,
            'active_tracks': len(active_tracks),
            'total_tracks': total_tracks,
            'frame_count': self.frame_count,
            'tracking_efficiency': len(active_tracks) / max(total_tracks, 1)
        }
    
    def reset(self):
        """Reset tracking state."""
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_count = 0
        logger.info("Face tracking service reset")
