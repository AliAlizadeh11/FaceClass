"""
FaceClass Services Package
Provides face detection, tracking, recognition, and video processing services.
"""

from .face_detection import FaceDetectionService
from .face_tracking import FaceTrackingService
from .face_recognition import FaceRecognitionService
from .video_processor import VideoProcessor

__all__ = [
    'FaceDetectionService',
    'FaceTrackingService', 
    'FaceRecognitionService',
    'VideoProcessor'
]
