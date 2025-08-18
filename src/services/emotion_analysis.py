"""
Emotion analysis service (Team 2).

Uses DeepFace if available, otherwise falls back to a lightweight heuristic.
Target classes: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "confused", "tired"].

API:
    analyze_emotions(face_images: List[np.ndarray]) -> List[str]
"""

from typing import List
import numpy as np

try:
    from deepface import DeepFace  # type: ignore
    _DEEPFACE_AVAILABLE = True
except Exception:
    _DEEPFACE_AVAILABLE = False


_TARGET_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "confused", "tired"]


def _deepface_predict(face_img: np.ndarray) -> str:
    # DeepFace expects BGR or RGB numpy array; we pass as-is
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        # DeepFace may return dict or list depending on version
        if isinstance(result, list) and result:
            result = result[0]
        emotion_dict = result.get('emotion') or {}
        if not emotion_dict:
            return 'neutral'
        # choose highest score
        label = max(emotion_dict.items(), key=lambda kv: kv[1])[0].lower()
        # map to target classes if needed
        if label not in _TARGET_CLASSES:
            if label == 'surprised':
                label = 'surprise'
            elif label == 'fearful':
                label = 'fear'
            elif label == 'disgusted':
                label = 'disgust'
            elif label == 'angry' or label == 'anger':
                label = 'angry'
            elif label == 'sad' or label == 'sadness':
                label = 'sad'
            elif label == 'happy' or label == 'happiness':
                label = 'happy'
            elif label == 'neutral' or label == 'calm':
                label = 'neutral'
        return label if label in _TARGET_CLASSES else 'neutral'
    except Exception:
        return 'neutral'


def _heuristic_predict(face_img: np.ndarray) -> str:
    # Enhanced heuristic fallback using intensity/contrast and edge analysis
    try:
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        
        # Calculate additional features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Enhanced emotion mapping for all 9 classes
        if edge_density > 0.15 and mean_intensity > 160:
            return 'surprise'
        elif edge_density > 0.12 and mean_intensity > 150:
            return 'fear'
        elif std_intensity > 40 and mean_intensity < 120:
            return 'angry'
        elif std_intensity > 35 and mean_intensity < 100:
            return 'disgust'
        elif mean_intensity > 140 and std_intensity < 30:
            return 'happy'
        elif mean_intensity < 90 and std_intensity < 25:
            return 'sad'
        elif 120 < mean_intensity < 150 and 25 < std_intensity < 40:
            return 'confused'
        elif mean_intensity < 110 and std_intensity < 20:
            return 'tired'
        elif 100 <= mean_intensity <= 140:
            return 'neutral'
        else:
            # Fallback
            if mean_intensity > 150:
                return 'happy'
            elif mean_intensity < 100:
                return 'sad'
            else:
                return 'neutral'
    except Exception:
        return 'neutral'


def analyze_emotions(face_images: List[np.ndarray]) -> List[str]:
    """Analyze emotions in face images.
    
    Args:
        face_images: List of face images as numpy arrays
        
    Returns:
        List of emotion labels for each face image
    """
    predictions: List[str] = []
    if not face_images:
        return predictions
        
    if _DEEPFACE_AVAILABLE:
        for img in face_images:
            predictions.append(_deepface_predict(img))
        return predictions
    
    # Fallback to heuristic
    for img in face_images:
        predictions.append(_heuristic_predict(img))
    return predictions


