"""
Enhanced Database Manager for Face Recognition - Team 1
Manages student face database with lighting, angle variations, and quality assessment
"""

import cv2
import numpy as np
import json
import sqlite3
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import hashlib
import os

logger = logging.getLogger(__name__)


class FaceDatabaseManager:
    """
    Enhanced database manager for face recognition with advanced features.
    
    Features:
    - Multi-variant face storage (lighting, angles, expressions)
    - Quality assessment integration
    - Automatic database optimization
    - Backup and recovery
    - Performance monitoring
    """
    
    def __init__(self, config: Dict = None, db_path: str = None):
        """Initialize face database manager."""
        self.config = config or {}
        self.db_path = db_path or self._get_default_db_path()
        
        # Database configuration
        self.max_faces_per_person = self.config.get('max_faces_per_person', 20)
        self.min_quality_score = self.config.get('min_quality_score', 0.7)
        self.enable_auto_optimization = self.config.get('enable_auto_optimization', True)
        self.backup_interval = self.config.get('backup_interval', 24)  # hours
        
        # Initialize database
        self._init_database()
        
        # Performance monitoring
        self.performance_metrics = {
            'query_times': [],
            'insert_times': [],
            'total_queries': 0,
            'total_inserts': 0
        }
        
        # Last backup time
        self.last_backup = self._load_last_backup_time()
        
        # Auto-optimization timer
        self.last_optimization = self._load_last_optimization_time()
    
    def _get_default_db_path(self) -> str:
        """Get default database path."""
        db_dir = Path(__file__).parent.parent.parent / 'data' / 'face_database'
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / 'face_recognition.db')
    
    def _init_database(self):
        """Initialize database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create students table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        email TEXT,
                        department TEXT,
                        enrollment_date TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create face_encodings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_encodings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        encoding_data BLOB NOT NULL,
                        face_image_path TEXT,
                        quality_score REAL,
                        lighting_condition TEXT,
                        pose_angles TEXT,
                        expression_type TEXT,
                        capture_date TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id)
                    )
                ''')
                
                # Create face_variants table for different conditions
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_variants (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        variant_type TEXT NOT NULL,
                        variant_data TEXT,
                        quality_metrics TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id)
                    )
                ''')
                
                # Create performance_logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_type TEXT NOT NULL,
                        execution_time REAL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        details TEXT
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_student_id ON face_encodings(student_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality_score ON face_encodings(quality_score)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_variant_type ON face_variants(variant_type)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_student(self, student_id: str, name: str, email: str = None, 
                   department: str = None) -> bool:
        """Add a new student to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO students 
                    (student_id, name, email, department, enrollment_date, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (student_id, name, email, department, 
                     datetime.now().isoformat(), datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"Student {student_id} added/updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add student {student_id}: {e}")
            return False
    
    def add_face_encoding(self, student_id: str, encoding: np.ndarray, 
                         face_image_path: str = None, quality_score: float = None,
                         lighting_condition: str = None, pose_angles: Dict = None,
                         expression_type: str = None) -> bool:
        """Add face encoding with metadata to database."""
        start_time = datetime.now()
        
        try:
            # Validate student exists
            if not self._student_exists(student_id):
                logger.error(f"Student {student_id} not found in database")
                return False
            
            # Check if we've reached the maximum faces per person
            if self._get_face_count(student_id) >= self.max_faces_per_person:
                logger.warning(f"Maximum faces reached for student {student_id}, removing oldest")
                self._remove_oldest_face(student_id)
            
            # Serialize encoding
            encoding_blob = pickle.dumps(encoding)
            
            # Serialize pose angles
            pose_angles_json = json.dumps(pose_angles) if pose_angles else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO face_encodings 
                    (student_id, encoding_data, face_image_path, quality_score,
                     lighting_condition, pose_angles, expression_type, capture_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (student_id, encoding_blob, face_image_path, quality_score,
                     lighting_condition, pose_angles_json, expression_type,
                     datetime.now().isoformat()))
                
                conn.commit()
                
                # Log performance
                execution_time = (datetime.now() - start_time).total_seconds()
                self._log_performance('insert_face', execution_time)
                
                logger.info(f"Face encoding added for student {student_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add face encoding for student {student_id}: {e}")
            return False
    
    def add_face_variant(self, student_id: str, variant_type: str, 
                        variant_data: Dict, quality_metrics: Dict = None) -> bool:
        """Add face variant information (lighting, pose, expression variations)."""
        try:
            # Validate student exists
            if not self._student_exists(student_id):
                logger.error(f"Student {student_id} not found in database")
                return False
            
            # Serialize data
            variant_data_json = json.dumps(variant_data)
            quality_metrics_json = json.dumps(quality_metrics) if quality_metrics else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO face_variants 
                    (student_id, variant_type, variant_data, quality_metrics)
                    VALUES (?, ?, ?, ?)
                ''', (student_id, variant_type, variant_data_json, quality_metrics_json))
                
                conn.commit()
                logger.info(f"Face variant {variant_type} added for student {student_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add face variant for student {student_id}: {e}")
            return False
    
    def get_face_encodings(self, student_id: str, 
                          min_quality: float = None) -> List[Tuple[np.ndarray, Dict]]:
        """Get face encodings for a student with optional quality filtering."""
        start_time = datetime.now()
        
        try:
            min_quality = min_quality or self.min_quality_score
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT encoding_data, quality_score, lighting_condition, 
                           pose_angles, expression_type, capture_date
                    FROM face_encodings 
                    WHERE student_id = ? AND quality_score >= ?
                    ORDER BY quality_score DESC, capture_date DESC
                '''
                
                cursor.execute(query, (student_id, min_quality))
                results = cursor.fetchall()
                
                # Deserialize encodings and create metadata
                encodings = []
                for row in results:
                    encoding = pickle.loads(row[0])
                    metadata = {
                        'quality_score': row[1],
                        'lighting_condition': row[2],
                        'pose_angles': json.loads(row[3]) if row[3] else None,
                        'expression_type': row[4],
                        'capture_date': row[5]
                    }
                    encodings.append((encoding, metadata))
                
                # Log performance
                execution_time = (datetime.now() - start_time).total_seconds()
                self._log_performance('query_encodings', execution_time)
                
                logger.info(f"Retrieved {len(encodings)} face encodings for student {student_id}")
                return encodings
                
        except Exception as e:
            logger.error(f"Failed to get face encodings for student {student_id}: {e}")
            return []
    
    def get_all_face_encodings(self, min_quality: float = None) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
        """Get all face encodings from database with quality filtering."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all students
                cursor.execute('SELECT student_id FROM students')
                student_ids = [row[0] for row in cursor.fetchall()]
                
                # Get encodings for each student
                all_encodings = {}
                for student_id in student_ids:
                    encodings = self.get_face_encodings(student_id, min_quality)
                    if encodings:
                        all_encodings[student_id] = encodings
                
                logger.info(f"Retrieved encodings for {len(all_encodings)} students")
                return all_encodings
                
        except Exception as e:
            logger.error(f"Failed to get all face encodings: {e}")
            return {}
    
    def search_similar_faces(self, query_encoding: np.ndarray, 
                           threshold: float = 0.6, max_results: int = 10) -> List[Dict]:
        """Search for similar faces in the database."""
        start_time = datetime.now()
        
        try:
            all_encodings = self.get_all_face_encodings()
            similar_faces = []
            
            for student_id, encodings in all_encodings.items():
                for encoding, metadata in encodings:
                    # Calculate similarity (cosine similarity)
                    similarity = self._calculate_similarity(query_encoding, encoding)
                    
                    if similarity >= threshold:
                        similar_faces.append({
                            'student_id': student_id,
                            'similarity': similarity,
                            'metadata': metadata
                        })
            
            # Sort by similarity and limit results
            similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
            similar_faces = similar_faces[:max_results]
            
            # Log performance
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_performance('search_similar', execution_time)
            
            logger.info(f"Found {len(similar_faces)} similar faces")
            return similar_faces
            
        except Exception as e:
            logger.error(f"Failed to search for similar faces: {e}")
            return []
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT student_id, name, email, department, enrollment_date, 
                           created_at, updated_at
                    FROM students 
                    WHERE student_id = ?
                ''', (student_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'student_id': row[0],
                        'name': row[1],
                        'email': row[2],
                        'department': row[3],
                        'enrollment_date': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get student info for {student_id}: {e}")
            return None
    
    def get_database_stats(self) -> Dict:
        """Get database statistics and performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count students
                cursor.execute('SELECT COUNT(*) FROM students')
                student_count = cursor.fetchone()[0]
                
                # Count face encodings
                cursor.execute('SELECT COUNT(*) FROM face_encodings')
                encoding_count = cursor.fetchone()[0]
                
                # Count variants
                cursor.execute('SELECT COUNT(*) FROM face_variants')
                variant_count = cursor.fetchone()[0]
                
                # Average quality score
                cursor.execute('SELECT AVG(quality_score) FROM face_encodings')
                avg_quality = cursor.fetchone()[0] or 0.0
                
                # Database size
                db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
                
                stats = {
                    'total_students': student_count,
                    'total_face_encodings': encoding_count,
                    'total_variants': variant_count,
                    'average_quality_score': round(avg_quality, 3),
                    'database_size_mb': round(db_size, 2),
                    'performance_metrics': self.performance_metrics.copy(),
                    'last_backup': self.last_backup.isoformat() if self.last_backup else None,
                    'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def optimize_database(self) -> bool:
        """Optimize database performance and clean up old data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analyze tables for better query planning
                cursor.execute('ANALYZE')
                
                # Clean up old performance logs (keep last 1000)
                cursor.execute('''
                    DELETE FROM performance_logs 
                    WHERE id NOT IN (
                        SELECT id FROM performance_logs 
                        ORDER BY timestamp DESC 
                        LIMIT 1000
                    )
                ''')
                
                # Remove low-quality encodings if too many exist
                cursor.execute('''
                    DELETE FROM face_encodings 
                    WHERE id NOT IN (
                        SELECT id FROM face_encodings 
                        ORDER BY quality_score DESC, capture_date DESC 
                        LIMIT ?
                    )
                ''', (self.max_faces_per_person * 100,))  # Keep reasonable number
                
                # Vacuum database to reclaim space
                cursor.execute('VACUUM')
                
                conn.commit()
                
                # Update optimization time
                self.last_optimization = datetime.now()
                self._save_last_optimization_time()
                
                logger.info("Database optimization completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Create database backup."""
        try:
            if backup_path is None:
                backup_dir = Path(self.db_path).parent / 'backups'
                backup_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = str(backup_dir / f'face_recognition_backup_{timestamp}.db')
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            # Update backup time
            self.last_backup = datetime.now()
            self._save_last_backup_time()
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            # Validate backup file
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Create current backup before restore
            self.backup_database()
            
            # Restore from backup
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def _student_exists(self, student_id: str) -> bool:
        """Check if student exists in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM students WHERE student_id = ?', (student_id,))
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def _get_face_count(self, student_id: str) -> int:
        """Get number of face encodings for a student."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM face_encodings WHERE student_id = ?', (student_id,))
                return cursor.fetchone()[0]
        except Exception:
            return 0
    
    def _remove_oldest_face(self, student_id: str):
        """Remove oldest face encoding for a student."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM face_encodings 
                    WHERE id = (
                        SELECT id FROM face_encodings 
                        WHERE student_id = ? 
                        ORDER BY capture_date ASC 
                        LIMIT 1
                    )
                ''', (student_id,))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to remove oldest face for student {student_id}: {e}")
    
    def _calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate cosine similarity between two encodings."""
        try:
            # Normalize encodings
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(encoding1, encoding2) / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    def _log_performance(self, operation_type: str, execution_time: float):
        """Log performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_logs (operation_type, execution_time)
                    VALUES (?, ?)
                ''', (operation_type, execution_time))
                conn.commit()
            
            # Update in-memory metrics
            if operation_type == 'query_encodings':
                self.performance_metrics['query_times'].append(execution_time)
                self.performance_metrics['total_queries'] += 1
            elif operation_type == 'insert_face':
                self.performance_metrics['insert_times'].append(execution_time)
                self.performance_metrics['total_inserts'] += 1
            
            # Keep only last 100 measurements
            if len(self.performance_metrics['query_times']) > 100:
                self.performance_metrics['query_times'] = self.performance_metrics['query_times'][-100:]
            if len(self.performance_metrics['insert_times']) > 100:
                self.performance_metrics['insert_times'] = self.performance_metrics['insert_times'][-100:]
                
        except Exception as e:
            logger.warning(f"Failed to log performance: {e}")
    
    def _load_last_backup_time(self) -> Optional[datetime]:
        """Load last backup time from file."""
        try:
            backup_time_file = Path(self.db_path).parent / 'last_backup.txt'
            if backup_time_file.exists():
                with open(backup_time_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    return datetime.fromisoformat(timestamp_str)
        except Exception:
            pass
        return None
    
    def _save_last_backup_time(self):
        """Save last backup time to file."""
        try:
            backup_time_file = Path(self.db_path).parent / 'last_backup.txt'
            with open(backup_time_file, 'w') as f:
                f.write(self.last_backup.isoformat())
        except Exception as e:
            logger.warning(f"Failed to save backup time: {e}")
    
    def _load_last_optimization_time(self) -> Optional[datetime]:
        """Load last optimization time from file."""
        try:
            opt_time_file = Path(self.db_path).parent / 'last_optimization.txt'
            if opt_time_file.exists():
                with open(opt_time_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    return datetime.fromisoformat(timestamp_str)
        except Exception:
            pass
        return None
    
    def _save_last_optimization_time(self):
        """Save last optimization time to file."""
        try:
            opt_time_file = Path(self.db_path).parent / 'last_optimization.txt'
            with open(opt_time_file, 'w') as f:
                f.write(self.last_optimization.isoformat())
        except Exception as e:
            logger.warning(f"Failed to save optimization time: {e}")
    
    def auto_maintenance(self):
        """Perform automatic database maintenance tasks."""
        try:
            current_time = datetime.now()
            
            # Check if backup is needed
            if (self.last_backup is None or 
                (current_time - self.last_backup).total_seconds() > self.backup_interval * 3600):
                logger.info("Performing automatic database backup")
                self.backup_database()
            
            # Check if optimization is needed
            if (self.enable_auto_optimization and 
                (self.last_optimization is None or 
                 (current_time - self.last_optimization).total_seconds() > 24 * 3600)):  # Daily
                logger.info("Performing automatic database optimization")
                self.optimize_database()
                
        except Exception as e:
            logger.error(f"Automatic maintenance failed: {e}")


def create_database_manager(config: Dict = None, db_path: str = None) -> FaceDatabaseManager:
    """Factory function to create face database manager."""
    return FaceDatabaseManager(config, db_path)


if __name__ == "__main__":
    # Example usage
    config = {
        'max_faces_per_person': 20,
        'min_quality_score': 0.7,
        'enable_auto_optimization': True,
        'backup_interval': 24
    }
    
    db_manager = create_database_manager(config)
    print("Face Database Manager created successfully!")
    
    # Get database stats
    stats = db_manager.get_database_stats()
    print(f"Database stats: {stats}")
