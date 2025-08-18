"""
Face detection service for FaceClass project.
Implements multiple face detection models with consistent API.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """Service for face detection using various models."""
    
    def __init__(self, config: Dict):
        """Initialize face detection service.
        
        Args:
            config: Configuration dictionary containing detection parameters
        """
        self.config = config
        self.model_type = config.get('face_detection.model', 'yolo')
        self.confidence_threshold = config.get('face_detection.confidence_threshold', 0.5)
        self.nms_threshold = config.get('face_detection.nms_threshold', 0.4)
        self.min_face_size = config.get('face_detection.min_face_size', 20)
        self.max_faces = config.get('face_detection.max_faces', 50)
        
        # Initialize detection model
        self.detector = self._load_detector()
        logger.info(f"Face detection service initialized with {self.model_type} model")
    
    def _load_detector(self) -> Dict:
        """Load the specified face detection model.
        
        Returns:
            Dictionary containing model type and loaded model
        """
        model_name = self.model_type.lower()
        
        if model_name == 'yolo':
            return self._load_yolo_detector()
        elif model_name == 'mtcnn':
            return self._load_mtcnn_detector()
        elif model_name == 'opencv':
            return self._load_opencv_detector()
        elif model_name == 'retinaface':
            logger.warning("RetinaFace not available due to TensorFlow compatibility issues")
            logger.info("Falling back to YOLO detector")
            return self._load_yolo_detector()
        else:
            logger.warning(f"Unknown detection model: {model_name}, using YOLO")
            return self._load_yolo_detector()
    
    def _load_yolo_detector(self) -> Dict:
        """Load YOLOv8-based face detector.
        
        Returns:
            Dictionary containing YOLO model type and loaded model
        """
        try:
            from ultralytics import YOLO
            
            # Check if YOLO model file exists
            model_path = Path(self.config.get('paths.models', 'models')) / 'face_detection' / 'yolov8n-face.pt'
            
            if model_path.exists():
                model = YOLO(str(model_path))
                logger.info(f"YOLOv8 model loaded from {model_path}")
                return {'type': 'yolo', 'model': model}
            else:
                # Try to use pre-trained YOLOv8n model
                model = YOLO('yolov8n.pt')
                logger.info("Using pre-trained YOLOv8n model for face detection")
                return {'type': 'yolo', 'model': model}
                
        except ImportError:
            logger.warning("Ultralytics not available, install with: pip install ultralytics")
            return self._load_opencv_detector()
        except Exception as e:
            logger.warning(f"Failed to load YOLO detector: {e}")
            return self._load_opencv_detector()
    
    def _load_mtcnn_detector(self) -> Dict:
        """Load MTCNN detector.
        
        Returns:
            Dictionary containing MTCNN model type and loaded model
        """
        try:
            from mtcnn import MTCNN
            detector = MTCNN(
                min_face_size=self.min_face_size,
                scale_factor=0.709,
                factor=0.709
            )
            logger.info("MTCNN detector loaded successfully")
            return {'type': 'mtcnn', 'model': detector}
        except ImportError:
            logger.warning("MTCNN not available, install with: pip install mtcnn")
            return self._load_opencv_detector()
    
    def _load_opencv_detector(self) -> Dict:
        """Load OpenCV Haar cascade detector.
        
        Returns:
            Dictionary containing OpenCV model type and loaded model
        """
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            
            if cascade.empty():
                logger.error("Failed to load OpenCV Haar cascade")
                raise ValueError("OpenCV cascade file not found")
            
            logger.info("OpenCV Haar cascade detector loaded successfully")
            return {'type': 'opencv', 'model': cascade}
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV detector: {e}")
            raise RuntimeError(f"Could not load any face detection model: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with bbox, confidence, and frame_id
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided for face detection")
            return []
        
        try:
            if self.detector['type'] == 'yolo':
                return self._detect_yolo(frame)
            elif self.detector['type'] == 'mtcnn':
                return self._detect_mtcnn(frame)
            elif self.detector['type'] == 'opencv':
                return self._detect_opencv(frame)
            else:
                logger.error(f"Unknown detector type: {self.detector['type']}")
                return []
                
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using YOLOv8.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Run YOLO inference
            results = self.detector['model'](frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter by confidence and class (person class for YOLOv8n)
                        if confidence >= self.confidence_threshold and class_id == 0:
                            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                            detections.append({
                                'bbox': bbox,
                                'confidence': float(confidence),
                                'class_id': class_id
                            })
            
            # Apply NMS if needed
            if len(detections) > self.max_faces:
                detections = self._apply_nms(detections)
                detections = detections[:self.max_faces]
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run MTCNN detection
            faces = self.detector['model'].detect_faces(rgb_frame)
            
            detections = []
            for face in faces:
                confidence = face['confidence']
                if confidence >= self.confidence_threshold:
                    bbox = face['box']
                    x, y, w, h = bbox
                    
                    detections.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': float(confidence),
                        'class_id': 1  # Face class
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar cascade.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Convert to grayscale for OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Run detection
            faces = self.detector['model'].detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            for (x, y, w, h) in faces:
                # OpenCV doesn't provide confidence, use default
                confidence = 0.8
                
                detections.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': confidence,
                    'class_id': 1  # Face class
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return []
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections after NMS
        """
        if len(detections) <= 1:
            return detections
        
        # Convert to numpy arrays for NMS
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
    
    def process_video(self, video_path: str) -> List[Dict]:
        """Process video file and detect faces in all frames.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            List of detection dictionaries with frame_id, bbox, and confidence
        """
        logger.info(f"Processing video: {video_path}")
        
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        frame_count = 0
        all_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces in current frame
                detections = self.detect_faces(frame)
                
                # Add frame information
                for detection in detections:
                    detection['frame_id'] = frame_count
                    detection['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                
                all_detections.extend(detections)
                frame_count += 1
                
                # Log progress every 100 frames
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames, found {len(all_detections)} detections")
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
        
        finally:
            cap.release()
        
        logger.info(f"Video processing completed. Total frames: {frame_count}, Total detections: {len(all_detections)}")
        return all_detections
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded detection model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.detector['type'],
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'min_face_size': self.min_face_size,
            'max_faces': self.max_faces
        }
