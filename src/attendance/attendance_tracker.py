"""
Attendance tracking module for FaceClass project.
Handles student attendance, absence tracking, and attendance statistics.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttendanceRecord:
    """Data class for attendance records."""
    student_id: str
    timestamp: datetime
    duration: float  # seconds
    confidence: float
    emotion: str
    attention_score: float
    location: Tuple[int, int]  # x, y coordinates
    seat_id: Optional[str] = None


@dataclass
class StudentAttendance:
    """Data class for student attendance summary."""
    student_id: str
    total_sessions: int
    attended_sessions: int
    total_duration: float  # seconds
    average_attention: float
    dominant_emotion: str
    preferred_seat: Optional[str] = None
    attendance_rate: float = 0.0


class AttendanceTracker:
    """Comprehensive attendance tracking system."""
    
    def __init__(self, config):
        """Initialize attendance tracker with configuration."""
        self.config = config
        self.min_detection_duration = config.get('attendance.min_detection_duration', 3.0)
        self.max_absence_duration = config.get('attendance.max_absence_duration', 300.0)
        self.auto_mark_absent = config.get('attendance.auto_mark_absent', True)
        self.attendance_threshold = config.get('attendance.attendance_threshold', 0.7)
        
        # Attendance data storage
        self.attendance_records: List[AttendanceRecord] = []
        self.student_sessions: Dict[str, List[AttendanceRecord]] = {}
        self.current_session: Dict[str, Dict] = {}  # student_id -> session_data
        self.session_start_time: Optional[datetime] = None
        
        # Statistics
        self.attendance_statistics: Dict[str, StudentAttendance] = {}
        
        # Load existing attendance data
        self._load_attendance_data()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new attendance tracking session."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.session_start_time = datetime.now()
        self.current_session = {}
        
        logger.info(f"Started attendance session: {session_id}")
        return session_id
    
    def end_session(self) -> Dict:
        """End the current session and return summary."""
        if not self.session_start_time:
            logger.warning("No active session to end")
            return {}
        
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        session_summary = self._generate_session_summary(session_duration)
        
        # Save session data
        self._save_session_data(session_summary)
        
        logger.info(f"Ended attendance session. Duration: {session_duration:.2f}s")
        return session_summary
    
    def process_detections(self, detections: List[Dict], timestamp: datetime) -> List[Dict]:
        """Process face detections and update attendance records."""
        if not self.session_start_time:
            logger.warning("No active session. Call start_session() first.")
            return detections
        
        updated_detections = []
        
        for detection in detections:
            student_id = detection.get('student_id', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            # Update current session tracking
            if student_id not in self.current_session:
                self.current_session[student_id] = {
                    'start_time': timestamp,
                    'last_seen': timestamp,
                    'total_duration': 0.0,
                    'detection_count': 0,
                    'attention_scores': [],
                    'emotions': []
                }
            
            # Update session data
            session_data = self.current_session[student_id]
            session_data['last_seen'] = timestamp
            session_data['detection_count'] += 1
            
            # Add attention and emotion data
            if 'attention' in detection:
                session_data['attention_scores'].append(
                    detection['attention'].get('attention_score', 0.0)
                )
            
            if 'emotion' in detection:
                session_data['emotions'].append(
                    detection['emotion'].get('dominant_emotion', 'neutral')
                )
            
            # Calculate duration
            duration = (timestamp - session_data['start_time']).total_seconds()
            session_data['total_duration'] = duration
            
            # Create attendance record if duration meets threshold
            if duration >= self.min_detection_duration:
                attendance_record = AttendanceRecord(
                    student_id=student_id,
                    timestamp=timestamp,
                    duration=duration,
                    confidence=confidence,
                    emotion=detection.get('emotion', {}).get('dominant_emotion', 'neutral'),
                    attention_score=detection.get('attention', {}).get('attention_score', 0.0),
                    location=detection.get('bbox', [0, 0, 0, 0])[:2],  # x, y coordinates
                    seat_id=detection.get('seat_id')
                )
                
                self.attendance_records.append(attendance_record)
                
                # Update detection with attendance info
                detection['attendance'] = {
                    'recorded': True,
                    'duration': duration,
                    'attendance_score': self._calculate_attendance_score(attendance_record)
                }
            
            updated_detections.append(detection)
        
        return updated_detections
    
    def _calculate_attendance_score(self, record: AttendanceRecord) -> float:
        """Calculate attendance score based on duration, confidence, and attention."""
        duration_score = min(record.duration / 60.0, 1.0)  # Normalize to 1 minute
        confidence_score = record.confidence
        attention_score = record.attention_score
        
        # Weighted average
        attendance_score = (
            duration_score * 0.4 +
            confidence_score * 0.3 +
            attention_score * 0.3
        )
        
        return min(attendance_score, 1.0)
    
    def _generate_session_summary(self, session_duration: float) -> Dict:
        """Generate summary for the current session."""
        summary = {
            'session_id': f"session_{int(self.session_start_time.timestamp())}",
            'start_time': self.session_start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': session_duration,
            'total_students': len(self.current_session),
            'attendance_records': len(self.attendance_records),
            'student_summaries': {}
        }
        
        for student_id, session_data in self.current_session.items():
            duration = session_data['total_duration']
            attendance_rate = duration / session_duration if session_duration > 0 else 0.0
            
            avg_attention = np.mean(session_data['attention_scores']) if session_data['attention_scores'] else 0.0
            dominant_emotion = self._get_dominant_emotion(session_data['emotions'])
            
            summary['student_summaries'][student_id] = {
                'duration': duration,
                'attendance_rate': attendance_rate,
                'average_attention': avg_attention,
                'dominant_emotion': dominant_emotion,
                'detection_count': session_data['detection_count']
            }
        
        return summary
    
    def _get_dominant_emotion(self, emotions: List[str]) -> str:
        """Get the most frequent emotion from a list."""
        if not emotions:
            return 'neutral'
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def get_attendance_statistics(self, student_id: Optional[str] = None) -> Dict:
        """Get attendance statistics for all students or a specific student."""
        if student_id:
            return self._get_student_statistics(student_id)
        
        # Calculate statistics for all students
        student_ids = set(record.student_id for record in self.attendance_records)
        
        for sid in student_ids:
            self.attendance_statistics[sid] = self._get_student_statistics(sid)
        
        return {
            'total_students': len(student_ids),
            'total_sessions': len(set(record.timestamp.date() for record in self.attendance_records)),
            'total_records': len(self.attendance_records),
            'students': self.attendance_statistics
        }
    
    def _get_student_statistics(self, student_id: str) -> StudentAttendance:
        """Get attendance statistics for a specific student."""
        student_records = [r for r in self.attendance_records if r.student_id == student_id]
        
        if not student_records:
            return StudentAttendance(
                student_id=student_id,
                total_sessions=0,
                attended_sessions=0,
                total_duration=0.0,
                average_attention=0.0,
                dominant_emotion='neutral'
            )
        
        # Calculate statistics
        total_duration = sum(r.duration for r in student_records)
        average_attention = np.mean([r.attention_score for r in student_records])
        dominant_emotion = self._get_dominant_emotion([r.emotion for r in student_records])
        
        # Count sessions (unique dates)
        session_dates = set(r.timestamp.date() for r in student_records)
        total_sessions = len(session_dates)
        attended_sessions = total_sessions  # All sessions with records are attended
        
        # Calculate attendance rate
        attendance_rate = attended_sessions / total_sessions if total_sessions > 0 else 0.0
        
        # Find preferred seat
        seat_counts = {}
        for record in student_records:
            if record.seat_id:
                seat_counts[record.seat_id] = seat_counts.get(record.seat_id, 0) + 1
        
        preferred_seat = max(seat_counts.items(), key=lambda x: x[1])[0] if seat_counts else None
        
        return StudentAttendance(
            student_id=student_id,
            total_sessions=total_sessions,
            attended_sessions=attended_sessions,
            total_duration=total_duration,
            average_attention=average_attention,
            dominant_emotion=dominant_emotion,
            preferred_seat=preferred_seat,
            attendance_rate=attendance_rate
        )
    
    def get_attendance_report(self, start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None) -> Dict:
        """Generate comprehensive attendance report."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Filter records by date range
        filtered_records = [
            r for r in self.attendance_records
            if start_date <= r.timestamp <= end_date
        ]
        
        if not filtered_records:
            return {
                'period': f"{start_date.date()} to {end_date.date()}",
                'total_records': 0,
                'students': {},
                'summary': {}
            }
        
        # Generate report
        report = {
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_records': len(filtered_records),
            'students': {},
            'summary': self._generate_report_summary(filtered_records)
        }
        
        # Add student-specific data
        student_ids = set(r.student_id for r in filtered_records)
        for student_id in student_ids:
            student_records = [r for r in filtered_records if r.student_id == student_id]
            report['students'][student_id] = self._generate_student_report(student_records)
        
        return report
    
    def _generate_report_summary(self, records: List[AttendanceRecord]) -> Dict:
        """Generate summary statistics for the report."""
        if not records:
            return {}
        
        total_duration = sum(r.duration for r in records)
        avg_attention = np.mean([r.attention_score for r in records])
        
        # Emotion distribution
        emotions = [r.emotion for r in records]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            'total_duration': total_duration,
            'average_attention': avg_attention,
            'emotion_distribution': emotion_counts,
            'unique_students': len(set(r.student_id for r in records)),
            'date_range': {
                'start': min(r.timestamp for r in records).date().isoformat(),
                'end': max(r.timestamp for r in records).date().isoformat()
            }
        }
    
    def _generate_student_report(self, records: List[AttendanceRecord]) -> Dict:
        """Generate report for a specific student."""
        if not records:
            return {}
        
        total_duration = sum(r.duration for r in records)
        avg_attention = np.mean([r.attention_score for r in records])
        dominant_emotion = self._get_dominant_emotion([r.emotion for r in records])
        
        return {
            'total_duration': total_duration,
            'average_attention': avg_attention,
            'dominant_emotion': dominant_emotion,
            'attendance_rate': len(records) / len(set(r.timestamp.date() for r in records)),
            'records_count': len(records)
        }
    
    def _load_attendance_data(self):
        """Load existing attendance data from file."""
        data_path = Path(self.config.get_path('outputs')) / 'attendance_data.json'
        if data_path.exists():
            try:
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                # Load attendance records
                self.attendance_records = [
                    AttendanceRecord(**record) for record in data.get('records', [])
                ]
                
                logger.info(f"Loaded {len(self.attendance_records)} attendance records")
            except Exception as e:
                logger.error(f"Failed to load attendance data: {e}")
    
    def _save_session_data(self, session_summary: Dict):
        """Save session data to file."""
        data_path = Path(self.config.get_path('outputs')) / 'attendance_data.json'
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load existing data
            existing_data = {}
            if data_path.exists():
                with open(data_path, 'r') as f:
                    existing_data = json.load(f)
            
            # Add new session data
            if 'sessions' not in existing_data:
                existing_data['sessions'] = []
            
            existing_data['sessions'].append(session_summary)
            
            # Update records
            existing_data['records'] = [asdict(record) for record in self.attendance_records]
            
            # Save updated data
            with open(data_path, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)
            
            logger.info(f"Saved session data: {session_summary['session_id']}")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def export_attendance_csv(self, output_path: Optional[str] = None) -> str:
        """Export attendance data to CSV format."""
        if output_path is None:
            output_path = Path(self.config.get_path('outputs')) / 'attendance_report.csv'
        
        # Convert records to DataFrame
        data = []
        for record in self.attendance_records:
            data.append({
                'student_id': record.student_id,
                'timestamp': record.timestamp,
                'duration': record.duration,
                'confidence': record.confidence,
                'emotion': record.emotion,
                'attention_score': record.attention_score,
                'location_x': record.location[0],
                'location_y': record.location[1],
                'seat_id': record.seat_id
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported attendance data to: {output_path}")
        return str(output_path) 