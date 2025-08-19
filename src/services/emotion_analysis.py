"""
Emotion analysis service (Team 2).

Uses DeepFace if available, otherwise falls back to a lightweight heuristic.
Target classes (extended):
    ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "confused", "tired"].

API:
    analyze_emotions(face_images: List[np.ndarray]) -> List[str]
"""

from typing import List
import numpy as np
import cv2  # type: ignore

try:
    from deepface import DeepFace  # type: ignore
    _DEEPFACE_AVAILABLE = True
except Exception:
    _DEEPFACE_AVAILABLE = False


_TARGET_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "confused", "tired"]


def _deepface_predict(face_img: np.ndarray) -> str:
    # DeepFace expects BGR or RGB numpy array; we pass as-is
    try:
        # DeepFace accepts BGR; we keep as-is. We request only emotion.
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        # DeepFace may return dict or list depending on version
        if isinstance(result, list) and result:
            result = result[0]
        emotion_dict = result.get('emotion') or {}
        if not emotion_dict:
            return 'neutral'
        # choose highest score
        sorted_items = sorted(((k.lower(), float(v)) for k, v in emotion_dict.items()), key=lambda kv: kv[1], reverse=True)
        label, top_score = sorted_items[0]
        second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        # map to target classes if needed
        if label not in _TARGET_CLASSES:
            if label == 'surprised':
                label = 'surprise'
            elif label == 'fearful':
                label = 'fear'
            elif label == 'disgusted':
                label = 'disgust'
        # Heuristic refinement for extended classes
        # "confused": when top two emotions are close (ambiguous) and involve negative/uncertain affects
        negative_set = {"fear", "sad", "disgust", "surprise", "angry"}
        if (label in negative_set) and (top_score - second_score) < 5.0:
            return 'confused'
        # "tired": neutral-looking faces with low contrast/brightness
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = face_img if face_img.ndim == 2 else face_img[..., 0]
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if label == 'neutral' and (mean < 110.0) and (std < 25.0):
            return 'tired'
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
    # Very simple rules to cover extended classes
    if mean > 175 and std > 45:
        return 'surprise'
    if mean > 155:
        return 'happy'
    if mean < 55 and std < 25:
        return 'tired'
    if 55 <= mean < 90 and std < 35:
        return 'sad'
    if 120 <= mean <= 150 and 20 <= std <= 35:
        return 'confused'
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


