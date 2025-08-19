"""
Attention analysis service (Team 2).

Uses MediaPipe Face Mesh if available (or returns defaults) to estimate head pose and gaze direction.

API:
    analyze_attention(face_landmarks: Any) -> Dict
        Returns {"is_attentive": bool, "yaw": float, "pitch": float, "roll": float}

Attention rule:
    Not attentive if |yaw| > 25° or |pitch| > 20° for >3 consecutive frames (tracking handled upstream).
    This module only computes per-frame yaw/pitch/roll; the consecutive-frame logic is aggregated upstream.
"""

from typing import Any, Dict
import math


def analyze_attention(face_landmarks: Any) -> Dict:
    # This function assumes caller passes already-computed head pose angles if available.
    # If only sparse landmarks are passed, caller should estimate angles and include in face_landmarks.
    # Here we consume keys: 'yaw', 'pitch', 'roll'.
    try:
        yaw = float(face_landmarks.get('yaw', 0.0))
        pitch = float(face_landmarks.get('pitch', 0.0))
        roll = float(face_landmarks.get('roll', 0.0))
    except Exception:
        yaw, pitch, roll = 0.0, 0.0, 0.0
    # Normalize/clip extreme angles and smooth minor noise
    yaw = float(max(-90.0, min(90.0, yaw)))
    pitch = float(max(-90.0, min(90.0, pitch)))
    roll = float(max(-90.0, min(90.0, roll)))
    if abs(yaw) < 2.0:
        yaw = 0.0
    if abs(pitch) < 2.0:
        pitch = 0.0
    # Simple gaze direction heuristic from yaw/pitch
    # Categories: front, left, right, up, down
    if pitch >= 12.0:
        gaze_direction = 'down'
    elif pitch <= -12.0:
        gaze_direction = 'up'
    elif yaw <= -10.0:
        gaze_direction = 'left'
    elif yaw >= 10.0:
        gaze_direction = 'right'
    else:
        gaze_direction = 'front'
    # Per-frame attentiveness; consecutive-frame rule applied upstream
    is_attentive = (abs(yaw) <= 25.0) and (abs(pitch) <= 20.0) and (gaze_direction == 'front')
    return {
        'is_attentive': bool(is_attentive),
        'engaged': bool(is_attentive),  # compatibility with legacy callers
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'gaze_direction': gaze_direction
    }


