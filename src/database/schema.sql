-- FaceClass Database Schema
-- Defines tables for storing face detection, tracking, and recognition results

-- Students table to store known student information
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    class VARCHAR(50),
    enrollment_date DATE,
    face_embedding_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processed results table to store detection and tracking results
CREATE TABLE IF NOT EXISTS processed_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(100) NOT NULL,
    frame_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    student_id VARCHAR(50),
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    confidence REAL NOT NULL,
    recognition_confidence REAL,
    timestamp REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_session_frame (session_id, frame_id),
    INDEX idx_track_id (track_id),
    INDEX idx_student_id (student_id),
    INDEX idx_timestamp (timestamp)
);

-- Processing sessions table to track video processing runs
CREATE TABLE IF NOT EXISTS processing_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    video_path VARCHAR(255) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    total_frames INTEGER,
    frames_processed INTEGER,
    total_detections INTEGER,
    total_tracks INTEGER,
    total_recognitions INTEGER,
    processing_time REAL,
    status VARCHAR(20) DEFAULT 'running',
    config_snapshot TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face embeddings table to store face feature vectors
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id VARCHAR(50) NOT NULL,
    embedding_data BLOB NOT NULL,
    embedding_type VARCHAR(20) NOT NULL,
    embedding_size INTEGER NOT NULL,
    source_image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    INDEX idx_student_id (student_id)
);

-- Recognition logs table to track recognition attempts
CREATE TABLE IF NOT EXISTS recognition_logs (
    id INTEGER PRIMARY KEY AUTOINESTAMP,
    session_id VARCHAR(100) NOT NULL,
    frame_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    input_embedding_size INTEGER,
    best_match_student_id VARCHAR(50),
    best_match_confidence REAL,
    processing_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_session_frame (session_id, frame_id),
    INDEX idx_track_id (track_id)
);

-- Create views for common queries
CREATE VIEW IF NOT EXISTS v_detection_summary AS
SELECT 
    session_id,
    COUNT(DISTINCT track_id) as unique_tracks,
    COUNT(*) as total_detections,
    AVG(confidence) as avg_confidence,
    MIN(timestamp) as first_detection,
    MAX(timestamp) as last_detection
FROM processed_results
GROUP BY session_id;

CREATE VIEW IF NOT EXISTS v_student_attendance AS
SELECT 
    pr.session_id,
    pr.student_id,
    s.name,
    COUNT(*) as frames_present,
    MIN(pr.timestamp) as first_seen,
    MAX(pr.timestamp) as last_seen,
    AVG(pr.recognition_confidence) as avg_recognition_confidence
FROM processed_results pr
LEFT JOIN students s ON pr.student_id = s.student_id
WHERE pr.student_id IS NOT NULL
GROUP BY pr.session_id, pr.student_id;

-- Insert sample data for testing
INSERT OR IGNORE INTO students (student_id, name, email, class) VALUES
('STU001', 'John Doe', 'john.doe@university.edu', 'Computer Science'),
('STU002', 'Jane Smith', 'jane.smith@university.edu', 'Computer Science'),
('STU003', 'Bob Johnson', 'bob.johnson@university.edu', 'Computer Science');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_processed_results_session_track ON processed_results(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_processed_results_student_time ON processed_results(student_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_processing_sessions_status ON processing_sessions(status);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_type ON face_embeddings(embedding_type);
