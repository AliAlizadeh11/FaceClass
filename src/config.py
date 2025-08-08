"""
Configuration module for FaceClass project.
Manages all settings, paths, and parameters for the face analysis pipeline.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for FaceClass project."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from file or use defaults."""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or create default config."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'paths': {
                'data_dir': 'data',
                'raw_videos': 'data/raw_videos',
                'labeled_faces': 'data/labeled_faces',
                'heatmaps': 'data/heatmaps',
                'outputs': 'data/outputs',
                'models': 'models',
                'reports': 'reports'
            },
            'face_detection': {
                'model': 'opencv',  # Options: yolo, retinaface, mtcnn, opencv
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'min_face_size': 20
            },
            'face_recognition': {
                'model': 'arcface',  # Options: arcface, facenet, vggface
                'similarity_threshold': 0.6,
                'embedding_size': 512
            },
            'emotion_detection': {
                'model': 'affectnet',  # Options: affectnet, fer2013
                'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
                'confidence_threshold': 0.3
            },
            'attention_detection': {
                'model': 'mediapipe',  # Options: mediapipe, openface
                'gaze_threshold': 0.7,
                'head_pose_threshold': 30.0
            },
            'video_processing': {
                'fps': 30,
                'max_resolution': (1920, 1080),
                'batch_size': 4
            },
            'heatmap': {
                'resolution': (100, 100),
                'blur_radius': 5,
                'color_map': 'hot'
            },
            'dashboard': {
                'port': 8080,
                'host': 'localhost',
                'refresh_rate': 1.0
            },
            'logging': {
                'level': 'INFO',
                'file': 'faceclass.log'
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to YAML file."""
        save_path = path or self.config_path or 'config.yaml'
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get_path(self, key: str) -> Path:
        """Get path configuration as Path object."""
        path_str = self.get(f'paths.{key}')
        return Path(path_str) if path_str else None
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        for key in ['data_dir', 'raw_videos', 'labeled_faces', 'heatmaps', 'outputs', 'models', 'reports']:
            path = self.get_path(key)
            if path:
                path.mkdir(parents=True, exist_ok=True) 