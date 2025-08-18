"""
Emotion analysis service (Team 2).

Uses DeepFace if available, otherwise falls back to a lightweight heuristic.
Target classes: ["happy", "sad", "angry", "surprise", "fear", "neutral", "disgust"].

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


_TARGET_CLASSES = ["happy", "sad", "angry", "surprise", "fear", "neutral", "disgust"]


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
        return label if label in _TARGET_CLASSES else 'neutral'
    except Exception:
        return 'neutral'


def _heuristic_predict(face_img: np.ndarray) -> str:
    # Very lightweight heuristic fallback using intensity/contrast
    try:
        import cv2
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = face_img if face_img.ndim == 2 else face_img[..., 0]
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if mean > 170 and std > 45:
        return 'surprise'
    if mean > 150:
        return 'happy'
    if mean < 60:
        return 'sad'
    return 'neutral'


def analyze_emotions(face_images: List[np.ndarray]) -> List[str]:
    predictions: List[str] = []
    if not face_images:
        return predictions
    if _DEEPFACE_AVAILABLE:
        for img in face_images:
            predictions.append(_deepface_predict(img))
        return predictions
    # Fallback
    for img in face_images:
        predictions.append(_heuristic_predict(img))
    return predictions


