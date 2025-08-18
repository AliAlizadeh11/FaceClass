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


def analyze_attention(face_landmarks: Any) -> Dict:
    """Classify attentiveness from head pose angles.

    Expects a mapping with numeric keys 'yaw', 'pitch', 'roll' in degrees. Returns a
    dictionary with boolean 'is_attentive' and the angles echoed back.
    """
    # This function assumes caller passes already-computed head pose angles if available.
    # If only sparse landmarks are passed, caller should estimate angles and include in face_landmarks.
    # Here we consume keys: 'yaw', 'pitch', 'roll'.
    try:
        yaw = float(face_landmarks.get('yaw', 0.0))
        pitch = float(face_landmarks.get('pitch', 0.0))
        roll = float(face_landmarks.get('roll', 0.0))
    except Exception:
        yaw, pitch, roll = 0.0, 0.0, 0.0
    is_attentive = (abs(yaw) <= 25.0) and (abs(pitch) <= 20.0)
    return {
        'is_attentive': bool(is_attentive),
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll
    }


