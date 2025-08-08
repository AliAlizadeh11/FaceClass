"""
Emotion detection module for FaceClass project.
Analyzes facial expressions to detect emotions and attention states.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmotionDetector:
    """Emotion and attention detection system."""
    
    def __init__(self, config):
        """Initialize emotion detector with configuration."""
        self.config = config
        self.emotion_model = config.get('emotion_detection.model', 'affectnet')
        self.emotions = config.get('emotion_detection.emotions', 
                                 ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        self.confidence_threshold = config.get('emotion_detection.confidence_threshold', 0.3)
        
        # Attention detection settings
        self.gaze_threshold = config.get('attention_detection.gaze_threshold', 0.7)
        self.head_pose_threshold = config.get('attention_detection.head_pose_threshold', 30.0)
        
        # Load emotion detection model
        self.emotion_model = self._load_emotion_model()
        
        # Load attention detection model
        self.attention_model = self._load_attention_model()
    
    def _load_emotion_model(self):
        """Load the specified emotion detection model."""
        model_name = self.emotion_model.lower()
        
        if model_name == 'affectnet':
            return self._load_affectnet_model()
        elif model_name == 'fer2013':
            return self._load_fer2013_model()
        else:
            logger.warning(f"Unknown emotion model: {model_name}, using placeholder")
            return self._load_placeholder_model()
    
    def _load_affectnet_model(self):
        """Load AffectNet-based emotion model."""
        try:
            # This would require installing specific emotion detection libraries
            # For now, return a placeholder
            logger.warning("AffectNet model not available, using placeholder")
            return self._load_placeholder_model()
        except ImportError:
            logger.warning("AffectNet not available, using placeholder")
            return self._load_placeholder_model()
    
    def _load_fer2013_model(self):
        """Load FER2013-based emotion model."""
        try:
            # This would require installing FER2013 model
            logger.warning("FER2013 model not available, using placeholder")
            return self._load_placeholder_model()
        except ImportError:
            logger.warning("FER2013 not available, using placeholder")
            return self._load_placeholder_model()
    
    def _load_placeholder_model(self):
        """Load placeholder emotion model for testing."""
        return {'type': 'placeholder', 'model': None}
    
    def _load_attention_model(self):
        """Load attention detection model."""
        model_name = self.config.get('attention_detection.model', 'mediapipe')
        
        if model_name == 'mediapipe':
            return self._load_mediapipe_model()
        elif model_name == 'openface':
            return self._load_openface_model()
        else:
            logger.warning(f"Unknown attention model: {model_name}, using MediaPipe")
            return self._load_mediapipe_model()
    
    def _load_mediapipe_model(self):
        """Load MediaPipe face mesh for attention detection."""
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            return {'type': 'mediapipe', 'model': face_mesh}
        except ImportError:
            logger.warning("MediaPipe not available, using placeholder")
            return self._load_placeholder_attention_model()
    
    def _load_openface_model(self):
        """Load OpenFace for attention detection."""
        try:
            # This would require installing OpenFace
            logger.warning("OpenFace not available, using placeholder")
            return self._load_placeholder_attention_model()
        except ImportError:
            logger.warning("OpenFace not available, using placeholder")
            return self._load_placeholder_attention_model()
    
    def _load_placeholder_attention_model(self):
        """Load placeholder attention model for testing."""
        return {'type': 'placeholder', 'model': None}
    
    def detect_emotions(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect emotions in a face image."""
        if self.emotion_model['type'] == 'placeholder':
            return self._detect_emotions_placeholder(face_image)
        else:
            # Implement specific model detection
            logger.warning("Using placeholder emotion detection")
            return self._detect_emotions_placeholder(face_image)
    
    def _detect_emotions_placeholder(self, face_image: np.ndarray) -> Dict[str, float]:
        """Placeholder emotion detection for testing."""
        # Generate random emotion probabilities
        np.random.seed(hash(str(face_image.shape)) % 2**32)
        emotions = {}
        total = 0
        
        for emotion in self.emotions:
            prob = np.random.random()
            emotions[emotion] = prob
            total += prob
        
        # Normalize probabilities
        for emotion in emotions:
            emotions[emotion] /= total
        
        return emotions
    
    def detect_attention(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect attention state (gaze direction, head pose)."""
        if self.attention_model['type'] == 'mediapipe':
            return self._detect_attention_mediapipe(face_image)
        elif self.attention_model['type'] == 'placeholder':
            return self._detect_attention_placeholder(face_image)
        else:
            logger.warning("Using placeholder attention detection")
            return self._detect_attention_placeholder(face_image)
    
    def _detect_attention_mediapipe(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect attention using MediaPipe."""
        try:
            import mediapipe as mp
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.attention_model['model'].process(rgb_image)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Calculate head pose (simplified)
                head_pose = self._calculate_head_pose(landmarks, face_image.shape)
                
                # Calculate gaze direction (simplified)
                gaze_direction = self._calculate_gaze_direction(landmarks)
                
                return {
                    'head_pose_x': head_pose[0],
                    'head_pose_y': head_pose[1],
                    'head_pose_z': head_pose[2],
                    'gaze_x': gaze_direction[0],
                    'gaze_y': gaze_direction[1],
                    'attention_score': self._calculate_attention_score(head_pose, gaze_direction)
                }
            else:
                return self._detect_attention_placeholder(face_image)
                
        except Exception as e:
            logger.error(f"MediaPipe attention detection failed: {e}")
            return self._detect_attention_placeholder(face_image)
    
    def _detect_attention_placeholder(self, face_image: np.ndarray) -> Dict[str, float]:
        """Placeholder attention detection for testing."""
        return {
            'head_pose_x': np.random.uniform(-30, 30),
            'head_pose_y': np.random.uniform(-30, 30),
            'head_pose_z': np.random.uniform(-30, 30),
            'gaze_x': np.random.uniform(-1, 1),
            'gaze_y': np.random.uniform(-1, 1),
            'attention_score': np.random.uniform(0, 1)
        }
    
    def _calculate_head_pose(self, landmarks, image_shape) -> Tuple[float, float, float]:
        """Calculate head pose from facial landmarks."""
        # Simplified head pose calculation
        # In a real implementation, this would use 3D-2D projection
        
        # Get key facial points
        nose_tip = landmarks.landmark[4]  # Nose tip
        left_eye = landmarks.landmark[33]  # Left eye center
        right_eye = landmarks.landmark[263]  # Right eye center
        
        # Calculate eye center
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        
        # Calculate pose angles (simplified)
        pitch = (nose_tip.y - eye_center_y) * 60  # Vertical rotation
        yaw = (nose_tip.x - eye_center_x) * 60    # Horizontal rotation
        roll = 0  # Simplified roll calculation
        
        return pitch, yaw, roll
    
    def _calculate_gaze_direction(self, landmarks) -> Tuple[float, float]:
        """Calculate gaze direction from eye landmarks."""
        # Simplified gaze calculation
        # In a real implementation, this would analyze iris position
        
        left_eye_center = landmarks.landmark[33]
        right_eye_center = landmarks.landmark[263]
        
        # Calculate gaze as offset from eye center
        gaze_x = (left_eye_center.x + right_eye_center.x) / 2 - 0.5
        gaze_y = (left_eye_center.y + right_eye_center.y) / 2 - 0.5
        
        return gaze_x, gaze_y
    
    def _calculate_attention_score(self, head_pose: Tuple[float, float, float], 
                                 gaze_direction: Tuple[float, float]) -> float:
        """Calculate overall attention score."""
        pitch, yaw, roll = head_pose
        gaze_x, gaze_y = gaze_direction
        
        # Check if head pose is within acceptable range
        head_attention = 1.0
        if abs(pitch) > self.head_pose_threshold or abs(yaw) > self.head_pose_threshold:
            head_attention = 0.5
        
        # Check if gaze is focused (simplified)
        gaze_attention = 1.0 - (abs(gaze_x) + abs(gaze_y)) / 2
        
        # Combine scores
        attention_score = (head_attention + gaze_attention) / 2
        
        return max(0, min(1, attention_score))
    
    def process_faces(self, detections: List[Dict]) -> List[Dict]:
        """Process face detections and add emotion/attention information."""
        if not detections:
            return []
        
        # Group detections by track_id to avoid redundant processing
        track_groups = {}
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id not in track_groups:
                track_groups[track_id] = []
            track_groups[track_id].append(detection)
        
        # Process each track
        for track_id, track_detections in track_groups.items():
            # Use the first detection of each track for emotion/attention analysis
            first_detection = track_detections[0]
            
            # Extract face region (assuming bbox is available)
            if 'bbox' in first_detection and 'frame' in first_detection:
                face_image = self._extract_face_region(
                    first_detection['frame'], 
                    first_detection['bbox']
                )
                
                # Detect emotions
                emotions = self.detect_emotions(face_image)
                
                # Detect attention
                attention = self.detect_attention(face_image)
                
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # Add emotion and attention to all detections in this track
                for detection in track_detections:
                    detection['emotions'] = emotions
                    detection['dominant_emotion'] = dominant_emotion[0]
                    detection['emotion_confidence'] = dominant_emotion[1]
                    detection['attention'] = attention
                    detection['is_attentive'] = attention['attention_score'] > self.gaze_threshold
        
        return detections
    
    def _extract_face_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract face region from frame."""
        x1, y1, x2, y2 = bbox
        face_region = frame[y1:y2, x1:x2]
        
        # Ensure minimum size
        if face_region.shape[0] < 20 or face_region.shape[1] < 20:
            # Pad or resize if too small
            face_region = cv2.resize(face_region, (64, 64))
        
        return face_region
    
    def get_emotion_statistics(self, detections: List[Dict]) -> Dict:
        """Get emotion statistics from detections."""
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        attention_scores = []
        
        for detection in detections:
            if 'dominant_emotion' in detection:
                emotion_counts[detection['dominant_emotion']] += 1
            
            if 'attention' in detection:
                attention_scores.append(detection['attention']['attention_score'])
        
        # Calculate percentages
        total_detections = len(detections)
        emotion_percentages = {}
        for emotion, count in emotion_counts.items():
            emotion_percentages[emotion] = count / total_detections if total_detections > 0 else 0
        
        # Calculate average attention score
        avg_attention = np.mean(attention_scores) if attention_scores else 0
        
        return {
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'average_attention_score': avg_attention,
            'total_detections': total_detections
        } 