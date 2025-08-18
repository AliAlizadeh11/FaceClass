"""
Attention analysis service (Team 2).

Uses MediaPipe Face Mesh if available (or returns defaults) to estimate head pose and gaze direction.

API:
    analyze_attention(face_landmarks: Any) -> Dict
        Returns {"engaged": bool, "gaze_direction": str, "head_pose": {"yaw": float, "pitch": float, "roll": float}}

Attention rule:
    Not attentive if |yaw| > 25° or |pitch| > 20° for >3 consecutive frames (tracking handled upstream).
    This module only computes per-frame yaw/pitch/roll; the consecutive-frame logic is aggregated upstream.
"""

from typing import Any, Dict


def analyze_attention(face_landmarks: Any) -> Dict:
    """Analyze attention based on head pose angles.
    
    Args:
        face_landmarks: Dict with 'yaw', 'pitch', 'roll' keys, or dict with 'head_pose' key
        
    Returns:
        Dict with attention analysis results
    """
    try:
        # Handle different input formats
        if isinstance(face_landmarks, dict):
            if 'head_pose' in face_landmarks:
                # Format: {'head_pose': {'yaw': x, 'pitch': y, 'roll': z}}
                head_pose = face_landmarks['head_pose']
                yaw = float(head_pose.get('yaw', 0.0))
                pitch = float(head_pose.get('pitch', 0.0))
                roll = float(head_pose.get('roll', 0.0))
            else:
                # Format: {'yaw': x, 'pitch': y, 'roll': z}
                yaw = float(face_landmarks.get('yaw', 0.0))
                pitch = float(face_landmarks.get('pitch', 0.0))
                roll = float(face_landmarks.get('roll', 0.0))
        else:
            yaw, pitch, roll = 0.0, 0.0, 0.0
            
        # Apply attention thresholds: yaw ±25°, pitch ±20°
        yaw_engaged = abs(yaw) <= 25.0
        pitch_engaged = abs(pitch) <= 20.0
        engaged = yaw_engaged and pitch_engaged
        
        # Determine gaze direction based on yaw
        if abs(yaw) < 10.0:
            gaze_direction = 'center'
        elif yaw > 10.0:
            gaze_direction = 'right'
        else:
            gaze_direction = 'left'
            
        return {
            'engaged': bool(engaged),
            'gaze_direction': gaze_direction,
            'head_pose': {
                'yaw': round(yaw, 2),
                'pitch': round(pitch, 2),
                'roll': round(roll, 2)
            }
        }
        
    except Exception as e:
        # Return default values on error
        return {
            'engaged': False,
            'gaze_direction': 'unknown',
            'head_pose': {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        }


