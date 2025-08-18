"""
Attendance manager (Team 3): per-session presence and duration tracking.
"""
from typing import Dict


class AttendanceManager:
    def __init__(self):
        # session_id -> { student_id -> frames_present }
        self._presence: Dict[str, Dict[str, int]] = {}

    def add_presence(self, session_id: str, student_id: str, frames: int = 1) -> None:
        sess = self._presence.setdefault(session_id, {})
        sess[student_id] = sess.get(student_id, 0) + frames

    def get_session_summary(self, session_id: str, fps: float) -> Dict[str, float]:
        # returns student_id -> seconds_present
        sess = self._presence.get(session_id, {})
        if fps <= 0:
            fps = 1.0
        return {sid: frames / fps for sid, frames in sess.items()}

    def get_all(self) -> Dict[str, Dict[str, int]]:
        return self._presence


