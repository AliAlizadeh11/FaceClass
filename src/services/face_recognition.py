"""
Face recognition service for FaceClass project.
Implements multiple face recognition models for identity matching.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
import os
import time

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Service for face recognition and identification."""
    
    def __init__(self, config: Dict):
        """Initialize face recognition service.
        
        Args:
            config: Configuration dictionary containing recognition parameters
        """
        self.config = config
        self.recognition_model = config.get('face_recognition.model', 'arcface')
        self.similarity_threshold = config.get('face_recognition.similarity_threshold', 0.6)
        self.embedding_size = config.get('face_recognition.embedding_size', 512)
        
        # Load recognition model
        self.recognizer = self._load_recognizer()
        
        # Known faces database
        self.known_faces: Dict[str, np.ndarray] = {}  # student_id -> embedding
        self.face_database_path = Path(config.get('paths.labeled_faces', 'data/labeled_faces')) / 'face_database.pkl'
        
        # Load existing face database
        self._load_face_database()
        
        logger.info(f"Face recognition service initialized with {self.recognition_model} model")
    
    def _load_recognizer(self) -> Dict:
        """Load the specified face recognition model.
        
        Returns:
            Dictionary containing model type and loaded model
        """
        model_name = self.recognition_model.lower()
        
        if model_name == 'arcface':
            return self._load_arcface_recognizer()
        elif model_name == 'facenet':
            return self._load_facenet_recognizer()
        elif model_name == 'deepface':
            return self._load_deepface_recognizer()
        elif model_name == 'opencv':
            return self._load_opencv_recognizer()
        else:
            logger.warning(f"Unknown recognition model: {model_name}, using OpenCV")
            return self._load_opencv_recognizer()
    
    def _load_arcface_recognizer(self) -> Dict:
        """Load ArcFace recognizer.
        
        Returns:
            Dictionary containing ArcFace model type and loaded model
        """
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("ArcFace recognizer loaded successfully")
            return {'type': 'arcface', 'model': app}
        except ImportError:
            logger.warning("InsightFace not available, install with: pip install insightface")
            return self._load_opencv_recognizer()
        except Exception as e:
            logger.warning(f"Failed to load ArcFace: {e}")
            return self._load_opencv_recognizer()
    
    def _load_facenet_recognizer(self) -> Dict:
        """Load FaceNet recognizer.
        
        Returns:
            Dictionary containing FaceNet model type and loaded model
        """
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2').eval()
            logger.info("FaceNet recognizer loaded successfully")
            return {'type': 'facenet', 'model': model}
        except ImportError:
            logger.warning("FaceNet-PyTorch not available, install with: pip install facenet-pytorch")
            return self._load_opencv_recognizer()
        except Exception as e:
            logger.warning(f"Failed to load FaceNet: {e}")
            return self._load_opencv_recognizer()
    
    def _load_deepface_recognizer(self) -> Dict:
        """Load DeepFace recognizer.
        
        Returns:
            Dictionary containing DeepFace model type and loaded model
        """
        try:
            from deepface import DeepFace
            logger.info("DeepFace recognizer loaded successfully")
            return {'type': 'deepface', 'model': DeepFace}
        except ImportError:
            logger.warning("DeepFace not available, install with: pip install deepface")
            return self._load_opencv_recognizer()
        except Exception as e:
            logger.warning(f"Failed to load DeepFace: {e}")
            return self._load_opencv_recognizer()
    
    def _load_opencv_recognizer(self) -> Dict:
        """Load OpenCV LBPH recognizer.
        
        Returns:
            Dictionary containing OpenCV model type and loaded model
        """
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            logger.info("OpenCV LBPH recognizer loaded successfully")
            return {'type': 'opencv', 'model': recognizer}
        except Exception as e:
            logger.error(f"Failed to load OpenCV recognizer: {e}")
            raise RuntimeError(f"Could not load any face recognition model: {e}")
    
    def _load_face_database(self):
        """Load existing face database from disk."""
        try:
            if self.face_database_path.exists():
                with open(self.face_database_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logger.info(f"Loaded {len(self.known_faces)} known faces from database")
            else:
                logger.info("No existing face database found, starting with empty database")
        except Exception as e:
            logger.warning(f"Failed to load face database: {e}")
            self.known_faces = {}
    
    def _save_face_database(self):
        """Save face database to disk."""
        try:
            # Ensure directory exists
            self.face_database_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.face_database_path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            logger.info(f"Saved {len(self.known_faces)} known faces to database")
        except Exception as e:
            logger.error(f"Failed to save face database: {e}")
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from face image.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image provided for embedding extraction")
            return None
        
        try:
            if self.recognizer['type'] == 'arcface':
                return self._extract_arcface_embedding(face_image)
            elif self.recognizer['type'] == 'facenet':
                return self._extract_facenet_embedding(face_image)
            elif self.recognizer['type'] == 'deepface':
                return self._extract_deepface_embedding(face_image)
            elif self.recognizer['type'] == 'opencv':
                return self._extract_opencv_embedding(face_image)
            else:
                logger.error(f"Unknown recognizer type: {self.recognizer['type']}")
                return None
                
        except Exception as e:
            logger.error(f"Error during embedding extraction: {e}")
            return None
    
    def _extract_arcface_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using ArcFace.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Face embedding as numpy array
        """
        try:
            # Convert BGR to RGB for ArcFace
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face embedding
            faces = self.recognizer['model'].get(rgb_image)
            if faces and len(faces) > 0:
                return faces[0].embedding
            else:
                logger.warning("No faces detected by ArcFace for embedding extraction")
                return None
                
        except Exception as e:
            logger.error(f"ArcFace embedding extraction error: {e}")
            return None
    
    def _extract_facenet_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using FaceNet.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Face embedding as numpy array
        """
        try:
            import torch
            from PIL import Image
            
            # Convert BGR to RGB and resize to 160x160 for FaceNet
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image = pil_image.resize((160, 160))
            
            # Convert to tensor and normalize
            transform = torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.ReLU()
            )
            
            # This is a simplified version - in practice you'd use proper transforms
            tensor_image = torch.from_numpy(np.array(pil_image)).float().permute(2, 0, 1).unsqueeze(0)
            tensor_image = tensor_image / 255.0
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.recognizer['model'](tensor_image)
                return embedding.squeeze().numpy()
                
        except Exception as e:
            logger.error(f"FaceNet embedding extraction error: {e}")
            return None
    
    def _extract_deepface_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using DeepFace.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Face embedding as numpy array
        """
        try:
            # Save temporary image for DeepFace
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_image)
            
            # Extract embedding
            embedding = self.recognizer['model'].represent(
                img_path=temp_path,
                model_name="Facenet",
                enforce_detection=False
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if embedding is not None and len(embedding) > 0:
                return embedding[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"DeepFace embedding extraction error: {e}")
            return None
    
    def _extract_opencv_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using OpenCV.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Face embedding as numpy array
        """
        try:
            # Convert to grayscale and resize for OpenCV
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            
            # Flatten and normalize
            embedding = gray.flatten().astype(np.float32) / 255.0
            return embedding
            
        except Exception as e:
            logger.error(f"OpenCV embedding extraction error: {e}")
            return None
    
    def identify_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify a face by comparing with known faces.
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Tuple of (student_id, confidence) or (None, 0.0) if no match
        """
        # Extract embedding
        embedding = self.extract_embedding(face_image)
        if embedding is None:
            return None, 0.0
        
        if not self.known_faces:
            logger.debug("No known faces in database")
            return None, 0.0
        
        # Compare with known faces
        best_match = None
        best_similarity = 0.0
        
        for student_id, known_embedding in self.known_faces.items():
            similarity = self._calculate_similarity(embedding, known_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = student_id
        
        if best_match:
            logger.debug(f"Face identified as {best_match} with confidence {best_similarity:.3f}")
            return best_match, best_similarity
        else:
            logger.debug(f"No match found, best similarity: {best_similarity:.3f}")
            return None, 0.0
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def add_face(self, student_id: str, face_image: np.ndarray) -> bool:
        """Add a new face to the database.
        
        Args:
            student_id: Student identifier
            face_image: Face image as numpy array
            
        Returns:
            True if face was added successfully
        """
        try:
            # Extract embedding
            embedding = self.extract_embedding(face_image)
            if embedding is None:
                logger.error(f"Failed to extract embedding for student {student_id}")
                return False
            
            # Add to database
            self.known_faces[student_id] = embedding
            
            # Save database
            self._save_face_database()
            
            logger.info(f"Added face for student {student_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face for student {student_id}: {e}")
            return False
    
    def remove_face(self, student_id: str) -> bool:
        """Remove a face from the database.
        
        Args:
            student_id: Student identifier
            
        Returns:
            True if face was removed successfully
        """
        try:
            if student_id in self.known_faces:
                del self.known_faces[student_id]
                self._save_face_database()
                logger.info(f"Removed face for student {student_id}")
                return True
            else:
                logger.warning(f"Student {student_id} not found in database")
                return False
                
        except Exception as e:
            logger.error(f"Error removing face for student {student_id}: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get information about the face database.
        
        Returns:
            Dictionary containing database information
        """
        return {
            'total_faces': len(self.known_faces),
            'student_ids': list(self.known_faces.keys()),
            'database_path': str(self.face_database_path),
            'model_type': self.recognizer['type'],
            'similarity_threshold': self.similarity_threshold
        }
    
    def clear_database(self):
        """Clear all faces from the database."""
        self.known_faces.clear()
        self._save_face_database()
        logger.info("Face database cleared")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded recognition model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.recognizer['type'],
            'similarity_threshold': self.similarity_threshold,
            'embedding_size': self.embedding_size
        }
