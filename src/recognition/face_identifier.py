"""
Face recognition module for FaceClass project.
Handles face embedding extraction and identity matching.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import os

logger = logging.getLogger(__name__)


class FaceIdentifier:
    """Face recognition and identification system."""
    
    def __init__(self, config):
        """Initialize face identifier with configuration."""
        self.config = config
        self.recognition_model = config.get('face_recognition.model', 'arcface')
        self.similarity_threshold = config.get('face_recognition.similarity_threshold', 0.6)
        self.embedding_size = config.get('face_recognition.embedding_size', 512)
        
        # Load recognition model
        self.recognizer = self._load_recognizer()
        
        # Known faces database
        self.known_faces = {}  # name -> embedding
        self.face_database_path = config.get_path('labeled_faces') / 'face_database.pkl'
        
        # Load existing face database
        self._load_face_database()
    
    def _load_recognizer(self):
        """Load the specified face recognition model."""
        model_name = self.recognition_model.lower()
        
        if model_name == 'arcface':
            return self._load_arcface_recognizer()
        elif model_name == 'facenet':
            return self._load_facenet_recognizer()
        elif model_name == 'vggface':
            return self._load_vggface_recognizer()
        else:
            logger.warning(f"Unknown recognition model: {model_name}, using OpenCV")
            return self._load_opencv_recognizer()
    
    def _load_arcface_recognizer(self):
        """Load ArcFace recognizer."""
        try:
            # This would require installing insightface
            # from insightface.app import FaceAnalysis
            # app = FaceAnalysis()
            # app.prepare(ctx_id=0, det_size=(640, 640))
            # return {'type': 'arcface', 'model': app}
            logger.warning("ArcFace not available, using OpenCV")
            return self._load_opencv_recognizer()
        except ImportError:
            logger.warning("ArcFace not available, using OpenCV")
            return self._load_opencv_recognizer()
    
    def _load_facenet_recognizer(self):
        """Load FaceNet recognizer."""
        try:
            # This would require installing facenet-pytorch
            # import torch
            # from facenet_pytorch import InceptionResnetV1
            # model = InceptionResnetV1(pretrained='vggface2').eval()
            # return {'type': 'facenet', 'model': model}
            logger.warning("FaceNet not available, using OpenCV")
            return self._load_opencv_recognizer()
        except ImportError:
            logger.warning("FaceNet not available, using OpenCV")
            return self._load_opencv_recognizer()
    
    def _load_vggface_recognizer(self):
        """Load VGGFace recognizer."""
        try:
            # This would require installing keras-vggface
            # from keras_vggface.vggface import VGGFace
            # model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
            # return {'type': 'vggface', 'model': model}
            logger.warning("VGGFace not available, using OpenCV")
            return self._load_opencv_recognizer()
        except ImportError:
            logger.warning("VGGFace not available, using OpenCV")
            return self._load_opencv_recognizer()
    
    def _load_opencv_recognizer(self):
        """Load OpenCV LBPH recognizer."""
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        return {'type': 'opencv', 'model': recognizer}
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding from face image."""
        if self.recognizer['type'] == 'opencv':
            return self._extract_opencv_embedding(face_image)
        else:
            # For other models, implement specific embedding extraction
            logger.warning("Using placeholder embedding for non-OpenCV model")
            return np.random.rand(self.embedding_size)
    
    def _extract_opencv_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding using OpenCV (simplified version)."""
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Resize to standard size
        gray = cv2.resize(gray, (128, 128))
        
        # Simple feature extraction (histogram)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        return hist
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings."""
        if self.recognizer['type'] == 'opencv':
            # Use correlation for histogram-based features
            correlation = np.corrcoef(embedding1, embedding2)[0, 1]
            return max(0, correlation)  # Ensure non-negative
        else:
            # Use cosine similarity for deep learning embeddings
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
    
    def identify_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Identify a face by comparing with known faces."""
        if not self.known_faces:
            return "Unknown", 0.0
        
        # Extract embedding
        embedding = self.extract_embedding(face_image)
        
        best_match = None
        best_similarity = 0.0
        
        # Compare with all known faces
        for name, known_embedding in self.known_faces.items():
            similarity = self.calculate_similarity(embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        # Check if similarity meets threshold
        if best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        else:
            return "Unknown", best_similarity
    
    def add_face(self, name: str, face_image: np.ndarray) -> bool:
        """Add a new face to the database."""
        try:
            embedding = self.extract_embedding(face_image)
            self.known_faces[name] = embedding
            self._save_face_database()
            logger.info(f"Added face for {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add face for {name}: {e}")
            return False
    
    def remove_face(self, name: str) -> bool:
        """Remove a face from the database."""
        if name in self.known_faces:
            del self.known_faces[name]
            self._save_face_database()
            logger.info(f"Removed face for {name}")
            return True
        return False
    
    def _load_face_database(self):
        """Load face database from file."""
        if self.face_database_path.exists():
            try:
                with open(self.face_database_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                logger.error(f"Failed to load face database: {e}")
                self.known_faces = {}
        else:
            logger.info("No existing face database found")
    
    def _save_face_database(self):
        """Save face database to file."""
        try:
            # Ensure directory exists
            self.face_database_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.face_database_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.info(f"Saved {len(self.known_faces)} known faces")
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")
    
    def process_detections(self, detections: List[Dict]) -> List[Dict]:
        """Process face detections and add identity information."""
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
            # Use the first detection of each track for identification
            first_detection = track_detections[0]
            
            # Extract face region (assuming bbox is available)
            if 'bbox' in first_detection and 'frame' in first_detection:
                face_image = self._extract_face_region(
                    first_detection['frame'], 
                    first_detection['bbox']
                )
                
                # Identify face
                identity, confidence = self.identify_face(face_image)
                
                # Add identity to all detections in this track
                for detection in track_detections:
                    detection['identity'] = identity
                    detection['identity_confidence'] = confidence
        
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
    
    def get_known_faces(self) -> List[str]:
        """Get list of known face names."""
        return list(self.known_faces.keys())
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the face database."""
        return {
            'total_faces': len(self.known_faces),
            'known_names': list(self.known_faces.keys()),
            'database_path': str(self.face_database_path)
        } 