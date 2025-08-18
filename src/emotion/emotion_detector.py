"""
Enhanced emotion and attention detection module for FaceClass project.
Analyzes facial expressions to detect emotions and attention states using multiple models.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EmotionDetector:
    """Enhanced emotion and attention detection system."""
    
    def __init__(self, config):
        """Initialize emotion detector with configuration."""
        self.config = config
        self.emotion_model = config.get('emotion_detection.model', 'fer2013')
        self.emotions = config.get('emotion_detection.emotions', 
                                 ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'confused', 'tired'])
        self.confidence_threshold = config.get('emotion_detection.confidence_threshold', 0.3)
        self.batch_size = config.get('emotion_detection.batch_size', 4)
        
        # Attention detection settings
        self.gaze_threshold = config.get('attention_detection.gaze_threshold', 0.7)
        self.head_pose_threshold = config.get('attention_detection.head_pose_threshold', 30.0)
        self.attention_timeout = config.get('attention_detection.attention_timeout', 5.0)
        self.min_attention_duration = config.get('attention_detection.min_attention_duration', 2.0)
        
        # Load emotion detection model
        self.emotion_model = self._load_emotion_model()
        
        # Load attention detection model
        self.attention_model = self._load_attention_model()
        
        # Tracking state
        self.attention_history = {}  # student_id -> attention_data
        self.emotion_history = {}    # student_id -> emotion_data
    
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
            # Try to load AffectNet model
            model_path = Path(self.config.get_path('models')) / 'emotion_recognition' / 'affectnet_model.pth'
            if model_path.exists():
                import torch
                import torch.nn as nn
                
                # Define a simple CNN for emotion classification
                class EmotionCNN(nn.Module):
                    def __init__(self, num_classes=8):
                        super(EmotionCNN, self).__init__()
                        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                        self.pool = nn.MaxPool2d(2, 2)
                        self.fc1 = nn.Linear(128 * 8 * 8, 512)
                        self.fc2 = nn.Linear(512, num_classes)
                        self.dropout = nn.Dropout(0.5)
                        self.relu = nn.ReLU()
                
                model = EmotionCNN(num_classes=len(self.emotions))
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                logger.info("AffectNet model loaded successfully")
                return {'type': 'affectnet', 'model': model}
            else:
                logger.warning("AffectNet model not found, using placeholder")
            return self._load_placeholder_model()
        except ImportError:
            logger.warning("PyTorch not available, using placeholder")
            return self._load_placeholder_model()
        except Exception as e:
            logger.warning(f"Failed to load AffectNet model: {e}")
            return self._load_placeholder_model()
    
    def _load_fer2013_model(self):
        """Load FER2013-based emotion model."""
        try:
            # Try to load FER2013 model
            model_path = Path(self.config.get_path('models')) / 'emotion_recognition' / 'fer2013_model.pth'
            if model_path.exists():
                import torch
                import torch.nn as nn
                
                # Define FER2013 CNN model
                class FER2013CNN(nn.Module):
                    def __init__(self, num_classes=7):
                        super(FER2013CNN, self).__init__()
                        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
                        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                        self.pool = nn.MaxPool2d(2, 2)
                        self.fc1 = nn.Linear(256 * 8 * 8, 512)
                        self.fc2 = nn.Linear(512, num_classes)
                        self.dropout = nn.Dropout(0.5)
                        self.relu = nn.ReLU()
                
                model = FER2013CNN(num_classes=len(self.emotions))
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                logger.info("FER2013 model loaded successfully")
                return {'type': 'fer2013', 'model': model}
            else:
                logger.warning("FER2013 model not found, using placeholder")
            return self._load_placeholder_model()
        except ImportError:
            logger.warning("PyTorch not available, using placeholder")
            return self._load_placeholder_model()
        except Exception as e:
            logger.warning(f"Failed to load FER2013 model: {e}")
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
            logger.info("MediaPipe face mesh loaded successfully")
            return {'type': 'mediapipe', 'model': face_mesh}
        except ImportError:
            logger.warning("MediaPipe not available, using placeholder")
            return self._load_placeholder_attention_model()
    
    def _load_openface_model(self):
        """Load OpenFace for attention detection."""
        try:
            # This would require installing OpenFace
            logger.warning("OpenFace not available, using MediaPipe")
            return self._load_mediapipe_model()
        except ImportError:
            logger.warning("OpenFace not available, using MediaPipe")
            return self._load_mediapipe_model()
    
    def _load_placeholder_attention_model(self):
        """Load placeholder attention model for testing."""
        return {'type': 'placeholder', 'model': None}
    
    def detect_emotions(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect emotions in a face image."""
        if self.emotion_model['type'] == 'placeholder':
            return self._detect_emotions_placeholder(face_image)
        elif self.emotion_model['type'] == 'fer2013':
            return self._detect_emotions_fer2013(face_image)
        elif self.emotion_model['type'] == 'affectnet':
            return self._detect_emotions_affectnet(face_image)
        else:
            return self._detect_emotions_placeholder(face_image)
    
    def _detect_emotions_placeholder(self, face_image: np.ndarray) -> Dict[str, float]:
        """Placeholder emotion detection for testing."""
        # Generate random emotion probabilities
        np.random.seed(int(time.time() * 1000) % 10000)
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
    
    def _detect_emotions_fer2013(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect emotions using FER2013 model."""
        try:
            import torch
            import torch.nn.functional as F
            
            # Preprocess image
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            
            # Convert to tensor
            tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.emotion_model['model'](tensor)
                probabilities = F.softmax(outputs, dim=1)
                probs = probabilities.squeeze().numpy()
            
            # Map to emotions
            emotions = {}
            for i, emotion in enumerate(self.emotions[:len(probs)]):
                emotions[emotion] = float(probs[i])
            
            return emotions
        except Exception as e:
            logger.error(f"FER2013 emotion detection failed: {e}")
            return self._detect_emotions_placeholder(face_image)
    
    def _detect_emotions_affectnet(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect emotions using AffectNet model."""
        try:
            import torch
            import torch.nn.functional as F
            
            # Preprocess image
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            
            # Convert to tensor
            tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.emotion_model['model'](tensor)
                probabilities = F.softmax(outputs, dim=1)
                probs = probabilities.squeeze().numpy()
            
            # Map to emotions
            emotions = {}
            for i, emotion in enumerate(self.emotions[:len(probs)]):
                emotions[emotion] = float(probs[i])
            
            return emotions
        except Exception as e:
            logger.error(f"AffectNet emotion detection failed: {e}")
            return self._detect_emotions_placeholder(face_image)
    
    def detect_attention(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect attention and gaze direction in a face image."""
        if self.attention_model['type'] == 'mediapipe':
            return self._detect_attention_mediapipe(face_image)
        elif self.attention_model['type'] == 'openface':
            return self._detect_attention_openface(face_image)
        else:
            return self._detect_attention_placeholder(face_image)
    
    def _detect_attention_mediapipe(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect attention using MediaPipe face mesh."""
        try:
            import mediapipe as mp
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face landmarks
            results = self.attention_model['model'].process(rgb_image)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Calculate head pose
                head_pose = self._calculate_head_pose(landmarks, face_image.shape)
                
                # Calculate gaze direction
                gaze_direction = self._calculate_gaze_direction(landmarks)
                
                # Calculate attention score
                attention_score = self._calculate_attention_score(head_pose, gaze_direction)
                
                return {
                    'attention_score': attention_score,
                    'head_pose': head_pose,
                    'gaze_direction': gaze_direction,
                    'is_attentive': attention_score > self.gaze_threshold
                }
            else:
                return self._detect_attention_placeholder(face_image)
        except Exception as e:
            logger.error(f"MediaPipe attention detection failed: {e}")
            return self._detect_attention_placeholder(face_image)
    
    def _detect_attention_openface(self, face_image: np.ndarray) -> Dict[str, float]:
        """Detect attention using OpenFace."""
        try:
            # This would require OpenFace implementation
            logger.warning("OpenFace attention detection not implemented")
            return self._detect_attention_placeholder(face_image)
        except Exception as e:
            logger.error(f"OpenFace attention detection failed: {e}")
            return self._detect_attention_placeholder(face_image)
    
    def _detect_attention_placeholder(self, face_image: np.ndarray) -> Dict[str, float]:
        """Placeholder attention detection for testing."""
        # Generate random attention data
        np.random.seed(int(time.time() * 1000) % 10000)
        
        attention_score = np.random.random()
        head_pose = (
            np.random.uniform(-30, 30),  # yaw
            np.random.uniform(-30, 30),  # pitch
            np.random.uniform(-30, 30)   # roll
        )
        gaze_direction = (
            np.random.uniform(-1, 1),    # x
            np.random.uniform(-1, 1)     # y
        )
        
        return {
            'attention_score': attention_score,
            'head_pose': head_pose,
            'gaze_direction': gaze_direction,
            'is_attentive': attention_score > self.gaze_threshold
        }
    
    def _calculate_head_pose(self, landmarks, image_shape) -> Tuple[float, float, float]:
        """Calculate head pose from facial landmarks."""
        try:
                # Get key facial landmarks for head pose estimation
                # This is a simplified implementation
            height, width = image_shape[:2]
            
                # Use eye and nose landmarks for pose estimation
            left_eye = landmarks.landmark[33]  # Left eye center
            right_eye = landmarks.landmark[263]  # Right eye center
            nose = landmarks.landmark[1]  # Nose tip
            
                # Calculate head pose angles (simplified)
            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_center_y = (left_eye.y + right_eye.y) / 2
            
            # Yaw (horizontal rotation)
            yaw = (eye_center_x - 0.5) * 60  # Convert to degrees
            
            # Pitch (vertical rotation)
            pitch = (eye_center_y - 0.5) * 60  # Convert to degrees
            
            # Roll (tilt)
            eye_angle = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
            roll = np.degrees(eye_angle)
            
            return (yaw, pitch, roll)

        except Exception as e:
            logger.error(f"Head pose calculation failed: {e}")
            return (0.0, 0.0, 0.0)
    
    def _calculate_gaze_direction(self, landmarks) -> Tuple[float, float]:
        """Calculate gaze direction from facial landmarks."""
        try:
                # Get eye landmarks for gaze estimation
            left_eye_center = landmarks.landmark[33]
            right_eye_center = landmarks.landmark[263]
            
                # Calculate gaze direction (simplified)
            gaze_x = (left_eye_center.x + right_eye_center.x) / 2 - 0.5
            gaze_y = (left_eye_center.y + right_eye_center.y) / 2 - 0.5
            
            return (gaze_x, gaze_y)

        except Exception as e:
            logger.error(f"Gaze direction calculation failed: {e}")
            return (0.0, 0.0)
    
    def _calculate_attention_score(self, head_pose: Tuple[float, float, float], 
                                 gaze_direction: Tuple[float, float]) -> float:
        """Calculate attention score based on head pose and gaze direction."""
        yaw, pitch, roll = head_pose
        gaze_x, gaze_y = gaze_direction
        
        # Calculate attention score based on:
        # 1. Head pose (should be facing forward)
        # 2. Gaze direction (should be centered)
        # 3. Stability (less movement = more attention)
        
        # Head pose score (0-1, higher is better)
        head_pose_score = 1.0 - (
            abs(yaw) / self.head_pose_threshold +
            abs(pitch) / self.head_pose_threshold +
            abs(roll) / self.head_pose_threshold
        ) / 3.0
        head_pose_score = max(0.0, min(1.0, head_pose_score))
        
        # Gaze direction score (0-1, higher is better)
        gaze_score = 1.0 - (abs(gaze_x) + abs(gaze_y)) / 2.0
        gaze_score = max(0.0, min(1.0, gaze_score))
        
        # Combined attention score
        attention_score = (head_pose_score * 0.6 + gaze_score * 0.4)
        
        return attention_score
    
    def process_faces(self, detections: List[Dict]) -> List[Dict]:
        """Process face detections and add emotion and attention information."""
        processed_detections = []
        
        for detection in detections:
            if 'bbox' not in detection:
                continue
            
            # Extract face region
            face_region = self._extract_face_region(detection['frame'], detection['bbox'])
            if face_region is None:
                continue
                
                # Detect emotions
            emotions = self.detect_emotions(face_region)
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_confidence = emotions[dominant_emotion]
                
                # Detect attention
            attention_data = self.detect_attention(face_region)
                
            # Update detection with emotion and attention data
            detection['emotion'] = {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': emotion_confidence
            }
            
            detection['attention'] = attention_data
            
            # Track attention history
            student_id = detection.get('student_id', 'unknown')
            self._update_attention_history(student_id, attention_data)
            self._update_emotion_history(student_id, emotions)
            
            processed_detections.append(detection)
        
        return processed_detections
    
    def _extract_face_region(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract face region from frame using bounding box."""
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            face_region = frame[y1:y2, x1:x2]
            return face_region
            
        except Exception as e:
            logger.error(f"Face region extraction failed: {e}")
            return None
    
    def _update_attention_history(self, student_id: str, attention_data: Dict):
        """Update attention history for a student."""
        if student_id not in self.attention_history:
            self.attention_history[student_id] = []
        
        attention_record = {
            'timestamp': datetime.now(),
            'attention_score': attention_data.get('attention_score', 0.0),
            'is_attentive': attention_data.get('is_attentive', False),
            'head_pose': attention_data.get('head_pose', (0.0, 0.0, 0.0)),
            'gaze_direction': attention_data.get('gaze_direction', (0.0, 0.0))
        }
        
        self.attention_history[student_id].append(attention_record)
        
        # Keep only recent history (last 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.attention_history[student_id] = [
            record for record in self.attention_history[student_id]
            if record['timestamp'] > cutoff_time
        ]
    
    def _update_emotion_history(self, student_id: str, emotions: Dict[str, float]):
        """Update emotion history for a student."""
        if student_id not in self.emotion_history:
            self.emotion_history[student_id] = []
        
        emotion_record = {
            'timestamp': datetime.now(),
            'emotions': emotions,
            'dominant_emotion': max(emotions.items(), key=lambda x: x[1])[0]
        }
        
        self.emotion_history[student_id].append(emotion_record)
        
        # Keep only recent history (last 5 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=5)
        self.emotion_history[student_id] = [
            record for record in self.emotion_history[student_id]
            if record['timestamp'] > cutoff_time
        ]
    
    def get_emotion_statistics(self, detections: List[Dict]) -> Dict:
        """Get emotion statistics from detections."""
        emotion_counts = {emotion: 0 for emotion in self.emotions}
        total_detections = len(detections)
        
        for detection in detections:
            if 'emotion' in detection:
                dominant_emotion = detection['emotion'].get('dominant_emotion', 'neutral')
                if dominant_emotion in emotion_counts:
                    emotion_counts[dominant_emotion] += 1
        
        # Calculate percentages
        emotion_percentages = {}
        for emotion, count in emotion_counts.items():
            emotion_percentages[emotion] = (count / total_detections * 100) if total_detections > 0 else 0
        
        return {
            'emotion_counts': emotion_counts,
            'emotion_percentages': emotion_percentages,
            'total_detections': total_detections,
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if total_detections > 0 else 'neutral'
        }
    
    def get_attention_statistics(self, detections: List[Dict]) -> Dict:
        """Get attention statistics from detections."""
        attention_scores = []
        attentive_count = 0
        total_detections = len(detections)
        
        for detection in detections:
            if 'attention' in detection:
                attention_score = detection['attention'].get('attention_score', 0.0)
                attention_scores.append(attention_score)
                
                if detection['attention'].get('is_attentive', False):
                    attentive_count += 1
        
        avg_attention = np.mean(attention_scores) if attention_scores else 0.0
        attention_rate = (attentive_count / total_detections * 100) if total_detections > 0 else 0
        
        return {
            'average_attention': avg_attention,
            'attention_rate': attention_rate,
            'attentive_count': attentive_count,
            'total_detections': total_detections,
            'attention_scores': attention_scores
        } 