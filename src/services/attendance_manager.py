"""
Attendance manager (Team 3): per-session presence and duration tracking.
"""
from typing import Dict


class AttendanceManager:
    def __init__(self):
        """Initialize attendance storage.

        Maintains an in-memory mapping from session IDs to per-student
        frame counts indicating how many frames each student was present.
        Structure: { session_id: { student_id: frames_present } }
        """
        # session_id -> { student_id -> frames_present }
        self._presence: Dict[str, Dict[str, int]] = {}

    def add_presence(self, session_id: str, student_id: str, frames: int = 1) -> None:
        """Accumulate presence for a student within a session.

        Args:
            session_id: Unique identifier for a class session.
            student_id: Unique identifier for the student.
            frames: Number of frames to add to the student's presence counter.
        """
        sess = self._presence.setdefault(session_id, {})
        sess[student_id] = sess.get(student_id, 0) + frames

    def get_session_summary(self, session_id: str, fps: float) -> Dict[str, float]:
        """Summarize presence in seconds per student for a session.

        Args:
            session_id: The session identifier to summarize.
            fps: Video frames per second used to convert frames to seconds. If
                non-positive, defaults to 1.0 to avoid division by zero.

        Returns:
            Mapping of student_id to seconds present (float).
        """
        sess = self._presence.get(session_id, {})
        if fps <= 0:
            fps = 1.0
        return {sid: frames / fps for sid, frames in sess.items()}

    def get_all(self) -> Dict[str, Dict[str, int]]:
        """Return the raw presence map for all sessions."""
        return self._presence


