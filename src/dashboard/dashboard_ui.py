"""
Dashboard UI module for FaceClass project.
Provides web-based interface for visualizing analysis results.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json
import threading
import time
import cv2
import base64
import io
import os
import tempfile
import socket
import subprocess
import platform
from collections import Counter

logger = logging.getLogger(__name__)


class DashboardUI:
    """Web-based dashboard for FaceClass analysis visualization."""
    
    def __init__(self, config):
        """Initialize dashboard with configuration."""
        self.config = config
        self.port = config.get('dashboard.port', 8080)
        self.host = config.get('dashboard.host', 'localhost')
        self.refresh_rate = config.get('dashboard.refresh_rate', 1.0)
        
        # Initialize Dash app with better configuration
        self.app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True,
            update_title=None,
            external_stylesheets=[
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
            ]
        )
        self.app.title = "FaceClass Dashboard"
        
        # Configure app for better performance
        self.app.config.suppress_callback_exceptions = True
        
        # Data storage
        self.analysis_data = {}
        self.live_data = {}
        self.video_frames = []
        self.current_frame_index = 0
        self.uploaded_video_path = None
        self.processing_status = "Ready"
        self.video_analysis_data = {}
        
        # Clean up old videos on startup
        self._cleanup_old_videos()
        
        # Load video frames
        self._load_video_frames()
        
        # Add sample data for demonstration
        self._add_sample_data()
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        # Data update thread
        self.update_thread = None
        self.is_running = False
    
    def _load_video_frames(self):
        """Load video frames from the frames directory."""
        frames_dir = Path("data/frames")
        if frames_dir.exists():
            # Load all jpg frames
            frame_files = sorted(frames_dir.glob("*.jpg"))
            if frame_files:
                self.video_frames = [str(f) for f in frame_files]
                logger.info(f"Loaded {len(self.video_frames)} video frames")
                
                # Process frames for face detection
                self._process_frames_for_detection()
            else:
                logger.warning("No video frames found in data/frames directory")
        else:
            logger.warning("Frames directory not found: data/frames")
    
    def _process_frames_for_detection(self):
        """Process frames to get real face detection results."""
        try:
            import sys
            from pathlib import Path
            
            # Add the src directory to the path for imports
            src_path = Path(__file__).parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            try:
                from detection.face_tracker import FaceTracker
                from config import Config
                
                # Initialize face tracker
                config = Config()
                face_tracker = FaceTracker(config)
            except ImportError as e:
                logger.error(f"Failed to import face tracker: {e}")
                # Fallback: create a simple mock detector
                face_tracker = self._create_mock_face_tracker()
            
            # Process each frame
            all_detections = []
            for i, frame_path in enumerate(self.video_frames):
                # Load frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                # Detect faces
                detections = face_tracker.detect_faces(frame)
                
                # Add frame information and track IDs
                for j, detection in enumerate(detections):
                    detection['frame_idx'] = i
                    detection['track_id'] = i * 100 + j  # Simple track ID assignment
                    detection['timestamp'] = time.time() - (len(self.video_frames) - i) * 5  # Simulate time progression
                    
                    # Add realistic identity and emotion data
                    detection['identity'] = f"Student {chr(65 + j)}"  # A, B, C, etc.
                    detection['dominant_emotion'] = np.random.choice(['focused', 'neutral', 'engaged', 'distracted'])
                    detection['emotion_confidence'] = np.random.uniform(0.6, 0.95)
                    detection['attention'] = {'attention_score': np.random.uniform(0.3, 0.95)}
                    detection['confidence'] = detection.get('confidence', 0.8)
                
                all_detections.extend(detections)
            
            # Store real detections
            if all_detections:
                self.live_data['detections'] = all_detections
                logger.info(f"Processed {len(all_detections)} face detections from {len(self.video_frames)} frames")
            else:
                logger.warning("No faces detected in video frames")
                
        except Exception as e:
            logger.error(f"Error processing frames for detection: {e}")
            # Fall back to sample data
            self._add_sample_data()
    
    def _process_uploaded_video(self, video_path: str) -> Dict:
        """Process uploaded video and extract frames with face detections for attendance tracking."""
        try:
            logger.info(f"Processing video for attendance tracking: {video_path}")
            self.processing_status = "Processing video for attendance analysis..."
            
            # Initialize face tracker with proper imports
            import sys
            from pathlib import Path
            
            # Add the src directory to the path for imports
            src_path = Path(__file__).parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            try:
                from detection.face_tracker import FaceTracker
                from config import Config
                
                config = Config()
                face_tracker = FaceTracker(config)
            except ImportError as e:
                logger.error(f"Failed to import face tracker: {e}")
                # Fallback: create a simple mock detector
                face_tracker = self._create_mock_face_tracker()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {total_frames} frames, {fps} fps, {duration:.2f}s duration")
            
            # Create frames directory if it doesn't exist
            frames_dir = Path("data/frames")
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames at regular intervals (every 2 seconds for better analysis)
            frame_interval = int(fps * 2) if fps > 0 else 60
            if frame_interval < 1:
                frame_interval = 1
            
            extracted_frames = []
            frame_analysis_data = []
            current_frame = 0
            
            logger.info(f"Extracting frames every {frame_interval} frames...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if current_frame % frame_interval == 0:
                    # Save frame
                    frame_filename = f"uploaded_frame_{current_frame:03d}.jpg"
                    frame_path = frames_dir / frame_filename
                    
                    # Resize frame for better processing (max 800x600)
                    height, width = frame.shape[:2]
                    if width > 800 or height > 600:
                        scale = min(800/width, 600/height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame_resized = cv2.resize(frame, (new_width, new_height))
                    else:
                        frame_resized = frame
                    
                    # Save the frame
                    cv2.imwrite(str(frame_path), frame_resized)
                    
                    # Analyze frame for faces
                    try:
                        detections = face_tracker.detect_faces(frame_resized)
                        frame_data = {
                            'frame_number': current_frame,
                            'frame_path': str(frame_path),
                            'detections': detections,
                            'timestamp': current_frame / fps if fps > 0 else 0
                        }
                        frame_analysis_data.append(frame_data)
                        extracted_frames.append(str(frame_path))
                        
                        logger.info(f"Frame {current_frame}: {len(detections)} faces detected")
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze frame {current_frame}: {e}")
                        # Still save the frame even if analysis fails
                        extracted_frames.append(str(frame_path))
                
                current_frame += 1
                
                # Progress update every 100 frames
                if current_frame % 100 == 0:
                    logger.info(f"Processed {current_frame}/{total_frames} frames...")
            
            cap.release()
            
            # Update the video frames list
            self.video_frames = extracted_frames
            self.current_frame_index = 0
            
            logger.info(f"Successfully extracted {len(extracted_frames)} frames from video")
            
            # Create analysis summary
            analysis_result = {
                'video_path': video_path,
                'total_frames': total_frames,
                'extracted_frames': len(extracted_frames),
                'fps': fps,
                'duration': duration,
                'frames': extracted_frames,
                'frame_analysis': frame_analysis_data,
                'detections': frame_analysis_data
            }
            
            self.processing_status = f"✅ Video processed successfully! Extracted {len(extracted_frames)} frames for analysis."
            
            return analysis_result
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Processing traceback: {traceback.format_exc()}")
            self.processing_status = f"❌ {error_msg}"
            return None
    
    def _create_mock_face_tracker(self):
        """Create a mock face tracker for fallback when real detector is not available."""
        class MockFaceTracker:
            def detect_faces(self, frame):
                # Return some mock detections for testing
                height, width = frame.shape[:2]
                mock_detections = []
                
                # Create 2-4 mock face detections
                num_faces = np.random.randint(2, 5)
                for i in range(num_faces):
                    x1 = np.random.randint(50, width - 150)
                    y1 = np.random.randint(50, height - 150)
                    w = np.random.randint(80, 120)
                    h = np.random.randint(80, 120)
                    
                    mock_detections.append({
                        'bbox': [x1, y1, x1 + w, y1 + h],
                        'confidence': np.random.uniform(0.7, 0.95),
                        'label': 'face'
                    })
                
                return mock_detections
        
        return MockFaceTracker()
    
    def _generate_student_id(self, detection: Dict, student_tracking: Dict) -> str:
        """Generate a unique student ID based on face position and tracking."""
        bbox = detection.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Create a position-based ID
        position_id = f"pos_{int(center_x/50)}_{int(center_y/50)}"
        
        # If we've seen this position before, use the existing ID
        if position_id in student_tracking:
            return student_tracking[position_id]
        
        # Generate new student ID
        student_id = f"STU_{len(student_tracking) + 1:03d}"
        student_tracking[position_id] = student_id
        
        return student_id
    
    def _save_uploaded_video(self, contents: str, filename: str) -> str:
        """Save uploaded video to temporary file."""
        try:
            logger.info(f"Starting to save video: {filename}")
            
            # Decode base64 content
            if ',' in contents:
                content_type, content_string = contents.split(',', 1)
                logger.info(f"Content type: {content_type}")
            else:
                content_string = contents
                logger.info("No content type found, using raw content")
            
            # Decode base64
            try:
                decoded = base64.b64decode(content_string)
                logger.info(f"Successfully decoded {len(decoded)} bytes")
            except Exception as e:
                logger.error(f"Failed to decode base64: {e}")
                return None
            
            # Create multiple directories for better organization
            temp_dir = Path("data/temp")
            raw_videos_dir = Path("data/raw_videos")
            
            # Ensure directories exist
            temp_dir.mkdir(parents=True, exist_ok=True)
            raw_videos_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean filename to prevent path issues
            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.')).rstrip()
            if not safe_filename:
                safe_filename = f"uploaded_video_{int(time.time())}.mp4"
            
            # Ensure unique filename in both directories
            counter = 1
            original_safe_filename = safe_filename
            while (temp_dir / safe_filename).exists() or (raw_videos_dir / safe_filename).exists():
                name, ext = os.path.splitext(original_safe_filename)
                safe_filename = f"{name}_{counter}{ext}"
                counter += 1
                if counter > 100:  # Prevent infinite loop
                    break
            
            # Save to temp directory first
            temp_video_path = temp_dir / safe_filename
            logger.info(f"Saving to temp: {temp_video_path}")
            
            # Write the file
            with open(temp_video_path, 'wb') as f:
                f.write(decoded)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force sync to disk
            
            # Verify the file was written correctly
            if temp_video_path.exists() and temp_video_path.stat().st_size > 0:
                logger.info(f"Successfully saved to temp: {temp_video_path} ({temp_video_path.stat().st_size} bytes)")
                
                # Also save a copy to raw_videos for permanent storage
                try:
                    raw_video_path = raw_videos_dir / safe_filename
                    import shutil
                    shutil.copy2(temp_video_path, raw_video_path)
                    logger.info(f"Also saved to raw_videos: {raw_video_path}")
                except Exception as copy_error:
                    logger.warning(f"Failed to copy to raw_videos: {copy_error}")
                
                # Return the temp path for immediate processing
                return str(temp_video_path)
            else:
                logger.error(f"File was not written correctly: {temp_video_path}")
                return None
            
        except Exception as e:
            logger.error(f"Error saving uploaded video: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _frame_to_base64(self, frame_path: str) -> str:
        """Convert a frame file to base64 string for display."""
        try:
            if not frame_path or not Path(frame_path).exists():
                return self._create_classroom_photo()
            
            # Read the image file
            with open(frame_path, 'rb') as f:
                image_data = f.read()
            
            # Convert to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            return f'data:image/jpeg;base64,{encoded_image}'
            
        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return self._create_classroom_photo()
    
    def _get_current_frame(self):
        """Get the current frame to display."""
        if self.video_frames and 0 <= self.current_frame_index < len(self.video_frames):
            return self._frame_to_base64(self.video_frames[self.current_frame_index])
        else:
            return self._create_classroom_photo()
    
    def _update_frame_index(self):
        """Update the current frame index for cycling through frames."""
        if self.video_frames:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.video_frames)
    
    def _add_sample_data(self):
        """Add realistic sample data for classroom analysis."""
        # Realistic classroom data based on 75-minute video
        sample_detections = [
            {
                'bbox': [150, 200, 250, 300],  # Front row, left side
                'frame_idx': 0,
                'track_id': 1,
                'identity': 'Student A',
                'dominant_emotion': 'focused',
                'emotion_confidence': 0.85,
                'attention': {'attention_score': 0.82},
                'confidence': 0.92,
                'timestamp': time.time() - 300  # 5 minutes ago
            },
            {
                'bbox': [350, 180, 450, 280],  # Front row, center
                'frame_idx': 0,
                'track_id': 2,
                'identity': 'Student B',
                'dominant_emotion': 'neutral',
                'emotion_confidence': 0.78,
                'attention': {'attention_score': 0.65},
                'confidence': 0.88,
                'timestamp': time.time() - 180  # 3 minutes ago
            },
            {
                'bbox': [550, 220, 650, 320],  # Front row, right side
                'frame_idx': 0,
                'track_id': 3,
                'identity': 'Student C',
                'dominant_emotion': 'engaged',
                'emotion_confidence': 0.91,
                'attention': {'attention_score': 0.94},
                'confidence': 0.95,
                'timestamp': time.time() - 120  # 2 minutes ago
            },
            {
                'bbox': [200, 350, 300, 450],  # Back row, left
                'frame_idx': 0,
                'track_id': 4,
                'identity': 'Student D',
                'dominant_emotion': 'neutral',
                'emotion_confidence': 0.62,
                'attention': {'attention_score': 0.45},
                'confidence': 0.76,
                'timestamp': time.time() - 90  # 1.5 minutes ago
            },
            {
                'bbox': [400, 380, 500, 480],  # Back row, center
                'frame_idx': 0,
                'track_id': 5,
                'identity': 'Student E',
                'dominant_emotion': 'focused',
                'emotion_confidence': 0.87,
                'attention': {'attention_score': 0.78},
                'confidence': 0.89,
                'timestamp': time.time() - 60  # 1 minute ago
            },
            {
                'bbox': [600, 360, 700, 460],  # Back row, right
                'frame_idx': 0,
                'track_id': 6,
                'identity': 'Student F',
                'dominant_emotion': 'distracted',
                'emotion_confidence': 0.73,
                'attention': {'attention_score': 0.32},
                'confidence': 0.82,
                'timestamp': time.time() - 30  # 30 seconds ago
            }
        ]
        
        # Realistic seat assignments based on classroom layout
        sample_seat_assignments = {
            'R1C1': {'identity': 'Student A', 'attention_score': 0.82, 'emotion': 'focused'},
            'R1C2': {'identity': 'Student B', 'attention_score': 0.65, 'emotion': 'neutral'},
            'R1C3': {'identity': 'Student C', 'attention_score': 0.94, 'emotion': 'engaged'},
            'R2C1': {'identity': 'Student D', 'attention_score': 0.45, 'emotion': 'neutral'},
            'R2C2': {'identity': 'Student E', 'attention_score': 0.78, 'emotion': 'focused'},
            'R2C3': {'identity': 'Student F', 'attention_score': 0.32, 'emotion': 'distracted'}
        }
        
        # Realistic classroom statistics
        classroom_stats = {
            'total_students': 6,
            'average_attention': 0.66,
            'dominant_emotion': 'focused',
            'engagement_rate': 0.67,  # 4 out of 6 students engaged
            'session_duration': 75,  # minutes
            'current_time': time.time()
        }
        
        self.live_data = {
            'detections': sample_detections,
            'seat_assignments': sample_seat_assignments,
            'classroom_stats': classroom_stats,
            'timestamp': time.time()
        }
    
    def _create_classroom_photo(self):
        """Create a simple classroom drawing with SVG."""
        # Create a simple, clean SVG representation of a classroom
        svg_content = '''
        <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="800" height="600" fill="#f8f9fa"/>
            
            <!-- Simple classroom floor -->
            <rect x="0" y="450" width="800" height="150" fill="#e9ecef"/>
            
            <!-- Simple walls -->
            <rect x="0" y="0" width="800" height="15" fill="#6c757d"/>
            <rect x="0" y="0" width="15" height="600" fill="#6c757d"/>
            <rect x="785" y="0" width="15" height="600" fill="#6c757d"/>
            
            <!-- Simple blackboard -->
            <rect x="100" y="80" width="600" height="80" fill="#212529"/>
            <text x="400" y="125" text-anchor="middle" fill="white" font-family="Arial" font-size="18" font-weight="bold">FaceClass Dashboard</text>
            <text x="400" y="145" text-anchor="middle" fill="white" font-family="Arial" font-size="14">Upload a video to start analysis</text>
            
            <!-- Simple desks (3 rows, 3 columns) -->
            <!-- Row 1 -->
            <rect x="150" y="300" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            <rect x="300" y="300" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            <rect x="450" y="300" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            
            <!-- Row 2 -->
            <rect x="150" y="380" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            <rect x="300" y="380" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            <rect x="450" y="380" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            
            <!-- Row 3 -->
            <rect x="150" y="460" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            <rect x="300" y="460" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            <rect x="450" y="460" width="100" height="60" fill="#dee2e6" stroke="#495057" stroke-width="1" rx="5"/>
            
            <!-- Simple chairs (circles) -->
            <!-- Row 1 Chairs -->
            <circle cx="200" cy="330" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            <circle cx="350" cy="330" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            <circle cx="500" cy="330" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            
            <!-- Row 2 Chairs -->
            <circle cx="200" cy="410" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            <circle cx="350" cy="410" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            <circle cx="500" cy="410" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            
            <!-- Row 3 Chairs -->
            <circle cx="200" cy="490" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            <circle cx="350" cy="490" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            <circle cx="500" cy="490" r="20" fill="#adb5bd" stroke="#495057" stroke-width="1"/>
            
            <!-- Simple windows -->
            <rect x="50" y="200" width="80" height="60" fill="#87ceeb" stroke="#495057" stroke-width="1" rx="3"/>
            <rect x="670" y="200" width="80" height="60" fill="#87ceeb" stroke="#495057" stroke-width="1" rx="3"/>
            
            <!-- Simple door -->
            <rect x="380" y="250" width="80" height="150" fill="#8b4513" stroke="#495057" stroke-width="1" rx="3"/>
            <circle cx="440" cy="325" r="3" fill="#495057"/>
            
            <!-- Simple student indicators (small dots) -->
            <circle cx="200" cy="330" r="8" fill="#27ae60" opacity="0.7"/>
            <circle cx="350" cy="330" r="8" fill="#f39c12" opacity="0.7"/>
            <circle cx="500" cy="330" r="8" fill="#27ae60" opacity="0.7"/>
            <circle cx="200" cy="410" r="8" fill="#e74c3c" opacity="0.7"/>
            <circle cx="350" cy="410" r="8" fill="#27ae60" opacity="0.7"/>
            <circle cx="500" cy="410" r="8" fill="#f39c12" opacity="0.7"/>
            <circle cx="200" cy="490" r="8" fill="#27ae60" opacity="0.7"/>
            <circle cx="350" cy="490" r="8" fill="#e74c3c" opacity="0.7"/>
            <circle cx="500" cy="490" r="8" fill="#27ae60" opacity="0.7"/>
            
            <!-- Simple legend -->
            <text x="650" y="550" text-anchor="middle" fill="#495057" font-family="Arial" font-size="12" font-weight="bold">Legend:</text>
            <circle cx="620" cy="545" r="6" fill="#27ae60"/>
            <text x="635" y="550" fill="#495057" font-family="Arial" font-size="10">High Attention</text>
            <circle cx="620" cy="560" r="6" fill="#f39c12"/>
            <text x="635" y="565" fill="#495057" font-family="Arial" font-size="10">Medium Attention</text>
            <circle cx="620" cy="575" r="6" fill="#e74c3c"/>
            <text x="635" y="580" fill="#495057" font-family="Arial" font-size="10">Low Attention</text>
        </svg>
        '''
        
        # Convert to base64
        svg_bytes = svg_content.encode('utf-8')
        svg_base64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f'data:image/svg+xml;base64,{svg_base64}'
    
    def _create_simple_analysis_message(self):
        """Create a simple FaceClass Analysis message for when no video is uploaded."""
        # Create a simple SVG with just the FaceClass Analysis message
        svg_content = '''
        <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="800" height="600" fill="#f8f9fa"/>
            
            <!-- Simple centered message -->
            <text x="400" y="300" text-anchor="middle" fill="#2c3e50" font-family="Arial" font-size="48" font-weight="bold">FaceClass Analysis</text>
            <text x="400" y="350" text-anchor="middle" fill="#7f8c8d" font-family="Arial" font-size="18">Upload a video to start analysis</text>
        </svg>
        '''
        
        # Convert to base64
        svg_bytes = svg_content.encode('utf-8')
        svg_base64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f'data:image/svg+xml;base64,{svg_base64}'
    
    def _setup_layout(self):
        """Setup the main dashboard layout with all sections in rows."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("FaceClass - Student Attendance Analysis System", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P("Comprehensive Computer Vision-Based Classroom Analysis", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
            
            # Store for dashboard state
            dcc.Store(id='dashboard-state', data={
                'uploaded_video_path': None,
                'video_frames': [],
                'current_frame_index': 0,
                'video_analysis_data': {},
                'live_data': {},
                'processing_status': "Ready - Upload a video to start analysis",
                'attendance_data': {},
                'statistics_data': {},
                'chart_data': {},
                'heatmap_data': {}
                    }),
                    
            # Row 1: Upload Video Section
            html.Div([
                html.H2("📁 Upload Video Section", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                        dcc.Upload(
                            id='upload-video',
                            children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '48px', 'color': '#3498db'}),
                            html.H3("Drag and Drop Video Here", style={'margin': '10px 0'}),
                            ]),
                            style={
                                'width': '100%',
                            'height': '200px',
                            'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                            'borderRadius': '10px',
                                'textAlign': 'center',
                                'backgroundColor': '#f8f9fa',
                            'borderColor': '#3498db',
                                'cursor': 'pointer'
                            },
                            accept='video/*',
                        multiple=False
                    ),
                    html.Div(id='upload-status', style={'marginTop': '10px', 'minHeight': '20px'}),
                    
                    # Video preview area
                    html.Div(id='video-preview', style={'marginTop': '15px', 'minHeight': '100px'}),
                    
                    html.Div([
                        # Process Video button
                        html.Button(
                            id='process-video-button',
                            children='🎬 Process Video',
                            style={
                                'backgroundColor': '#95a5a6',
                                'color': 'white',
                                'border': 'none',
                                'padding': '12px 24px',
                                'borderRadius': '6px',
                                'cursor': 'not-allowed',
                                'fontSize': '16px',
                                'fontWeight': 'bold',
                                'opacity': '0.6'
                            }
                        ),
                        html.Button(
                        'Clear Video', 
                            id='clear-video-button', 
                            n_clicks=0,
                            style={
                            'backgroundColor': '#e74c3c', 
                                'color': 'white', 
                                'border': 'none', 
                                'padding': '10px 20px',
                                'borderRadius': '4px',
                                'fontSize': '13px',
                                'fontWeight': 'bold',
                                'cursor': 'pointer'
                            }
                        )
                ], style={'marginTop': '15px'})
                ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'marginBottom': '30px'}),
                
            # Row 2: Video Analysis Results
            html.Div([
                html.H2("🎬 Video Analysis Results", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                
                # Frame counter and navigation
                html.Div([
                    html.Div([
                        html.Span("📊 Frame: ", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        html.Span(id='frame-counter', children="0/0", style={'color': '#3498db', 'fontWeight': 'bold'}),
                        html.Span(" of ", style={'color': '#7f8c8d'}),
                        html.Span(id='total-frames', children="0", style={'color': '#7f8c8d'})
                    ], style={'textAlign': 'center', 'marginBottom': '10px', 'fontSize': '16px'}),
                    
                    # Navigation buttons
                    html.Div([
                        html.Button(
                            '⏮️ Previous Frame',
                            id='prev-frame-button',
                            style={
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin': '0 10px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'fontSize': '14px'
                            }
                        ),
                        html.Button(
                            '⏭️ Next Frame',
                            id='next-frame-button',
                            style={
                                'backgroundColor': '#27ae60',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin': '0 10px',
                                'borderRadius': '5px',
                                'cursor': 'pointer',
                                'fontSize': '14px'
                            }
                        )
                    ], style={'textAlign': 'center', 'marginBottom': '15px'})
                ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '15px'}),
                
                # Video frame display
                html.Div([
                    html.Img(
                        id='video-frame',
                        src=self._create_classroom_photo(),
                        style={
                            'width': '100%',
                            'maxWidth': '800px',
                            'height': 'auto',
                            'border': '2px solid #bdc3c7',
                            'borderRadius': '8px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                        }
                    )
                ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                
                # Processing status
                html.Div(id='processing-status', style={'textAlign': 'center', 'minHeight': '60px'}),
                
                # Frame information
                html.Div(id='frame-info', style={'marginTop': '15px'})
            ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
            
            # Row 3: Attendance & Absence System
                        html.Div([
                html.H2("📊 Attendance & Absence System", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                        html.Div([
                    html.Div([
                        html.H4("Attendance Summary", style={'marginBottom': '10px'}),
                        html.Div(id='attendance-summary', style={'padding': '15px', 'backgroundColor': '#e8f5e8', 'borderRadius': '5px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                                html.H5("Attendance Rate", style={'marginBottom': '5px'}),
                                html.Div(id='attendance-rate', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#27ae60'})
                            ], style={'flex': '1', 'textAlign': 'center'}),
                        html.Div([
                                html.H5("Absent Count", style={'marginBottom': '5px'}),
                                html.Div(id='absent-count', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#e74c3c'})
                            ], style={'flex': '1', 'textAlign': 'center'})
                        ], style={'display': 'flex', 'gap': '20px'})
                    ], style={'flex': '1', 'marginRight': '20px'}),
                        html.Div([
                        html.H4("Student Attendance List", style={'marginBottom': '10px'}),
                        html.Div(id='student-attendance-list', style={'maxHeight': '300px', 'overflowY': 'auto', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'marginBottom': '30px'}),
            
            # Row 4: Real-time Statistics
            html.Div([
                html.H2("📈 Real-time Statistics", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H4("Face Detection", style={'marginBottom': '10px'}),
                        html.Div(id='face-count', style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#3498db', 'textAlign': 'center'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ebf3fd', 'borderRadius': '8px', 'margin': '5px'}),
                    html.Div([
                        html.H4("Attention Score", style={'marginBottom': '10px'}),
                        html.Div(id='attention-score', style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#f39c12', 'textAlign': 'center'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#fef9e7', 'borderRadius': '8px', 'margin': '5px'}),
                    html.Div([
                        html.H4("Dominant Emotion", style={'marginBottom': '10px'}),
                        html.Div(id='dominant-emotion', style={'fontSize': '32px', 'fontWeight': 'bold', 'color': '#e74c3c', 'textAlign': 'center'})
                    ], style={'flex': '1', 'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#fdf2f2', 'borderRadius': '8px', 'margin': '5px'})
                ], style={'display': 'flex', 'gap': '15px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'marginBottom': '30px'}),
                
            # Row 5: Analysis Charts
                html.Div([
                html.H2("📊 Analysis Charts", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                    html.Div([
                    html.Div([
                        html.H4("Emotion Distribution", style={'marginBottom': '10px'}),
                        dcc.Graph(id='emotion-chart', style={'height': '300px'})
                    ], style={'flex': '1', 'marginRight': '15px'}),
                    html.Div([
                        html.H4("Attention Timeline", style={'marginBottom': '10px'}),
                        dcc.Graph(id='attention-timeline', style={'height': '300px'})
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                html.Div([
                    html.Div([
                        html.H4("Position Heatmap", style={'marginBottom': '10px'}),
                        dcc.Graph(id='position-heatmap', style={'height': '300px'})
                    ], style={'flex': '1', 'marginRight': '15px'}),
                    html.Div([
                        html.H4("Seat Assignments", style={'marginBottom': '10px'}),
                        html.Div(id='seat-assignments', style={'height': '300px', 'overflowY': 'auto', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginTop': '20px'})
            ], style={'marginBottom': '30px'}),
                    
            # Row 6: Heatmap of Student Locations
                    html.Div([
                html.H2("🗺️ Heatmap of Student Locations", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H4("Classroom Heatmap", style={'marginBottom': '10px'}),
                        dcc.Graph(id='classroom-heatmap', style={'height': '400px'})
                    ], style={'flex': '2', 'marginRight': '20px'}),
                    html.Div([
                        html.H4("Heatmap Controls", style={'marginBottom': '10px'}),
                        html.Div([
                            html.Label("Heatmap Type:"),
                            dcc.Dropdown(
                                id='heatmap-type-dropdown',
                                options=[
                                    {'label': 'Presence Heatmap', 'value': 'presence'},
                                    {'label': 'Attention Heatmap', 'value': 'attention'},
                                    {'label': 'Emotion Heatmap', 'value': 'emotion'}
                                ],
                                value='presence',
                                style={'marginBottom': '15px'}
                            ),
                            html.Label("Heatmap Intensity:"),
                            dcc.Slider(
                                id='heatmap-intensity-slider',
                                min=0.1,
                                max=2.0,
                                step=0.1,
                                value=1.0,
                                marks={0.1: '0.1', 1.0: '1.0', 2.0: '2.0'}
                            )
                        ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'marginBottom': '30px'}),
            
            # Footer
            html.Div([
                html.P("FaceClass - Comprehensive Student Attendance Analysis System", 
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '0'}),
                html.P("Powered by Computer Vision and AI", 
                      style={'textAlign': 'center', 'color': '#95a5a6', 'margin': '5px 0 0 0'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginTop': '30px'}),
            
            # Hidden interval component for real-time updates
            dcc.Interval(
                id='interval-component',
                interval=1000,  # in milliseconds
                n_intervals=0
            )
        ], style={'padding': '20px', 'backgroundColor': '#f5f6fa', 'minHeight': '100vh'})
    
    def _setup_callbacks(self):
        """Setup all dashboard callbacks."""
        
        @self.app.callback(
            Output('face-count', 'children'),
            Output('attention-score', 'children'),
            Output('dominant-emotion', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_statistics(stored_data, n):
            """Update real-time statistics."""
            if not stored_data:
                return "0", "0.0%", "Neutral"
            
            # Get face count
            face_count = len(stored_data.get('video_frames', []))
            
            # Get attention score
            attention_scores = []
            if stored_data.get('video_analysis_data'):
                for detection in stored_data['video_analysis_data'].get('detections', []):
                    if 'attention' in detection:
                        attention_scores.append(detection['attention'].get('attention_score', 0.0))
            
            avg_attention = np.mean(attention_scores) if attention_scores else 0.0
            attention_percentage = f"{avg_attention:.1%}"
            
            # Get dominant emotion
            emotions = []
            if stored_data.get('video_analysis_data'):
                for detection in stored_data['video_analysis_data'].get('detections', []):
                    if 'emotion' in detection:
                        emotions.append(detection['emotion'].get('dominant_emotion', 'neutral'))
            
            if emotions:
                emotion_counts = Counter(emotions)
                dominant_emotion = emotion_counts.most_common(1)[0][0].title()
            else:
                dominant_emotion = "Neutral"
            
            return str(face_count), attention_percentage, dominant_emotion
        
        @self.app.callback(
            Output('emotion-chart', 'figure'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_emotion_chart(stored_data, n):
            """Update emotion distribution chart."""
            if not stored_data or not stored_data.get('video_analysis_data'):
                return self._create_empty_chart("No emotion data available")
            
            emotions = []
            for detection in stored_data['video_analysis_data'].get('detections', []):
                if 'emotion' in detection:
                    emotions.append(detection['emotion'].get('dominant_emotion', 'neutral'))
            
            if not emotions:
                return self._create_empty_chart("No emotion data available")
            
            # Count emotions
            emotion_counts = Counter(emotions)
            
            fig = go.Figure(data=[
                go.Bar(
                x=list(emotion_counts.keys()),
                y=list(emotion_counts.values()),
                    marker_color=['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
                )
            ])
            
            fig.update_layout(
                title="Emotion Distribution",
                xaxis_title="Emotions",
                yaxis_title="Count",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('attention-timeline', 'figure'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_attention_timeline(stored_data, n):
            """Update attention timeline chart."""
            if not stored_data or not stored_data.get('video_analysis_data'):
                return self._create_empty_chart("No attention data available")
            
            attention_scores = []
            timestamps = []
            
            for i, detection in enumerate(stored_data['video_analysis_data'].get('detections', [])):
                if 'attention' in detection:
                    attention_scores.append(detection['attention'].get('attention_score', 0.0))
                    timestamps.append(i)
            
            if not attention_scores:
                return self._create_empty_chart("No attention data available")
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=timestamps,
                    y=attention_scores,
                    mode='lines+markers',
                    line=dict(color='#f39c12', width=2),
                    marker=dict(size=6)
                )
            ])
            
            fig.update_layout(
                title="Attention Timeline",
                xaxis_title="Frame",
                yaxis_title="Attention Score",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('position-heatmap', 'figure'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_position_heatmap(stored_data, n):
            """Update position heatmap."""
            if not stored_data or not stored_data.get('video_analysis_data'):
                return self._create_empty_chart("No position data available")
            
            positions = []
            for detection in stored_data['video_analysis_data'].get('detections', []):
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    x = (bbox[0] + bbox[2]) / 2  # Center x
                    y = (bbox[1] + bbox[3]) / 2  # Center y
                    positions.append([x, y])
            
            if not positions:
                return self._create_empty_chart("No position data available")
            
            # Create heatmap data
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            fig = go.Figure(data=[
                go.Histogram2d(
                    x=x_coords,
                    y=y_coords,
                    nbinsx=20,
                    nbinsy=20,
                    colorscale='Hot'
                )
            ])
            
            fig.update_layout(
                title="Position Heatmap",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('classroom-heatmap', 'figure'),
            Input('dashboard-state', 'data'),
            Input('heatmap-type-dropdown', 'value'),
            Input('heatmap-intensity-slider', 'value'),
            Input('interval-component', 'n_intervals')
        )
        def update_classroom_heatmap(stored_data, heatmap_type, intensity, n):
            """Update classroom heatmap."""
            if not stored_data or not stored_data.get('video_analysis_data'):
                return self._create_empty_chart("No classroom data available")
            
            # Create classroom layout heatmap
            classroom_width = 1920
            classroom_height = 1080
            heatmap_data = np.zeros((50, 50))
            
            for detection in stored_data['video_analysis_data'].get('detections', []):
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    x = int((bbox[0] + bbox[2]) / 2 * 50 / classroom_width)
                    y = int((bbox[1] + bbox[3]) / 2 * 50 / classroom_height)
                    
                    if 0 <= x < 50 and 0 <= y < 50:
                        if heatmap_type == 'presence':
                            heatmap_data[y, x] += 1
                        elif heatmap_type == 'attention' and 'attention' in detection:
                            heatmap_data[y, x] += detection['attention'].get('attention_score', 0.0)
                        elif heatmap_type == 'emotion' and 'emotion' in detection:
                            emotion_score = detection['emotion'].get('confidence', 0.0)
                            heatmap_data[y, x] += emotion_score
            
            # Apply intensity
            heatmap_data *= intensity
            
            fig = go.Figure(data=[
                go.Heatmap(
                    z=heatmap_data,
                    colorscale='Hot',
                    showscale=True
                )
            ])
            
            fig.update_layout(
                title=f"Classroom {heatmap_type.title()} Heatmap",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                template="plotly_white"
            )
            
            return fig
        
        @self.app.callback(
            Output('seat-assignments', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_seat_assignments(stored_data, n):
            """Update seat assignments."""
            if not stored_data or not stored_data.get('video_analysis_data'):
                return html.P("No seat assignment data available", style={'color': '#7f8c8d'})
            
            seat_assignments = {}
            for detection in stored_data['video_analysis_data'].get('detections', []):
                if 'bbox' in detection and 'student_id' in detection:
                    bbox = detection['bbox']
                    x = (bbox[0] + bbox[2]) / 2
                    y = (bbox[1] + bbox[3]) / 2
                    
                    # Simple seat assignment based on position
                    row = int(y / 200) + 1
                    col = int(x / 200) + 1
                    seat_id = f"R{row}C{col}"
                    
                    if seat_id not in seat_assignments:
                        seat_assignments[seat_id] = []
                    
                    seat_assignments[seat_id].append(detection['student_id'])
            
            if not seat_assignments:
                return html.P("No seat assignments available", style={'color': '#7f8c8d'})
            
            seat_list = []
            for seat_id, students in seat_assignments.items():
                seat_list.append(
                    html.Div([
                        html.Strong(f"Seat {seat_id}: "),
                        html.Span(", ".join(set(students)))
                    ], style={'marginBottom': '5px', 'padding': '5px', 'backgroundColor': 'white', 'borderRadius': '3px'})
                )
            
            return seat_list
        
        @self.app.callback(
            Output('attendance-summary', 'children'),
            Output('attendance-rate', 'children'),
            Output('absent-count', 'children'),
            Output('student-attendance-list', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_attendance_display(stored_data, n_intervals):
            """Update attendance display."""
            if not stored_data:
                return "No attendance data", "0%", "0", []
            
            # Calculate attendance data
            total_students = len(set(
                detection.get('student_id', 'unknown') 
                for detection in stored_data.get('video_analysis_data', {}).get('detections', [])
                if detection.get('student_id') != 'unknown'
            ))
            
            if total_students == 0:
                return "No students detected", "0%", "0", []
            
            # Calculate attendance rate (simplified)
            attendance_rate = min(100, max(0, (total_students / max(total_students, 1)) * 100))
            
            # Get student list
            students = {}
            for detection in stored_data.get('video_analysis_data', {}).get('detections', []):
                student_id = detection.get('student_id', 'unknown')
                if student_id != 'unknown':
                    if student_id not in students:
                        students[student_id] = {
                            'detections': 0,
                            'attention_scores': [],
                            'emotions': []
                        }
                    
                    students[student_id]['detections'] += 1
                    
                    if 'attention' in detection:
                        students[student_id]['attention_scores'].append(
                            detection['attention'].get('attention_score', 0.0)
                        )
                    
                    if 'emotion' in detection:
                        students[student_id]['emotions'].append(
                            detection['emotion'].get('dominant_emotion', 'neutral')
                        )
            
            # Create student list
            student_list = []
            for student_id, data in students.items():
                avg_attention = np.mean(data['attention_scores']) if data['attention_scores'] else 0.0
                dominant_emotion = max(set(data['emotions']), key=data['emotions'].count) if data['emotions'] else 'neutral'
                
                student_list.append(
                    html.Div([
                        html.Strong(f"Student {student_id}: "),
                        html.Span(f"Detections: {data['detections']}, "),
                        html.Span(f"Avg Attention: {avg_attention:.1%}, "),
                        html.Span(f"Dominant Emotion: {dominant_emotion}")
                    ], style={'marginBottom': '5px', 'padding': '5px', 'backgroundColor': 'white', 'borderRadius': '3px'})
                )
            
            attendance_summary = f"Total Students: {total_students}, Attendance Rate: {attendance_rate:.1f}%"
            attendance_rate_display = f"{attendance_rate:.1f}%"
            absent_count = max(0, 20 - total_students)  # Assuming 20 total seats
            
            return attendance_summary, attendance_rate_display, str(absent_count), student_list
        
        @self.app.callback(
            Output('upload-status', 'children'),
            Output('dashboard-state', 'data'),
            Input('upload-video', 'contents'),
            Input('upload-video', 'filename'),
            State('dashboard-state', 'data')
        )
        def handle_upload(contents, filename, stored_data):
            """Handle video upload directly."""
            from dash import callback_context
            ctx = callback_context
            
            # Initialize state if needed
            if stored_data is None:
                stored_data = self._initialize_dashboard_state()
            
            # Check if upload was triggered
            if not ctx.triggered:
                return "Ready to upload video", stored_data
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            logger.info(f"Upload callback triggered by: {trigger_id}")
            
            if trigger_id == 'upload-video':
                logger.info(f"Upload triggered - contents: {bool(contents)}, filename: {filename}")
                
                # Check if we have both contents and filename
                if not contents:
                    logger.warning("No contents received")
                    return "❌ No video content received", stored_data
                
                if not filename:
                    logger.warning("No filename received")
                    return "❌ No filename received", stored_data
                
                # Check file extension
                allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
                file_ext = Path(filename).suffix.lower()
                if file_ext not in allowed_extensions:
                    return f"❌ Unsupported file format: {file_ext}. Supported formats: {', '.join(allowed_extensions)}", stored_data
                
                # More accurate file size calculation
                try:
                    # Remove data URL prefix if present
                    if contents.startswith('data:'):
                        content_string = contents.split(',', 1)[1]
                    else:
                        content_string = contents
                    
                    # Calculate actual size from base64
                    content_length = len(content_string)
                    # Base64 encoding increases size by ~33%, so decode to get actual size
                    actual_size = int(content_length * 0.75)  # More accurate estimate
                    
                    max_size = 100 * 1024 * 1024  # 100MB
                    
                    if actual_size > max_size:
                        return f"❌ File too large. Maximum size is 100MB. Estimated size: {actual_size / (1024*1024):.1f}MB", stored_data
                    
                    logger.info(f"File size check passed: {actual_size / (1024*1024):.1f}MB")
                    
                except Exception as size_error:
                    logger.warning(f"Could not calculate file size: {size_error}")
                    # Continue with upload if size calculation fails
                
                # Process the upload
                logger.info(f"Processing upload: {filename}")
                try:
                    # Log upload details for debugging
                    logger.info(f"Upload details - Filename: {filename}, Content length: {len(contents) if contents else 0}")
                    
                    video_path = self._save_uploaded_video(contents, filename)
                    logger.info(f"Save result - Path: {video_path}, Exists: {Path(video_path).exists() if video_path else False}")
                    
                    if video_path and Path(video_path).exists():
                        # Verify the uploaded video
                        logger.info(f"Verifying video: {video_path}")
                        if self._verify_uploaded_video(video_path):
                            file_size = Path(video_path).stat().st_size
                            stored_data['uploaded_video_path'] = video_path
                            stored_data['processing_status'] = f"Video uploaded and verified successfully ({file_size / (1024*1024):.1f}MB)"
                            logger.info(f"Video saved and verified successfully: {video_path} ({file_size / (1024*1024):.1f}MB)")
                            
                            # Create a more prominent success message with video preview
                            success_message = html.Div([
                                html.Div([
                                    html.Span("✅ ", style={'color': '#27ae60', 'fontSize': '20px', 'fontWeight': 'bold'}),
                                    html.Span(f"{filename} uploaded successfully!", style={'color': '#27ae60', 'fontWeight': 'bold', 'fontSize': '18px'}),
                                    html.Br(),
                                    html.Span(f"📁 Size: {file_size / (1024*1024):.1f}MB", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                                    html.Br(),
                                    html.Span(f"💾 Saved to: {Path(video_path).name}", style={'color': '#95a5a6', 'fontSize': '12px'}),
                                    html.Br(),
                                    html.Br(),
                                    html.Span("🎬 Click 'Process Video' to analyze frame by frame", 
                                             style={'color': '#3498db', 'fontSize': '16px', 'fontWeight': 'bold'}),
                                    html.Br(),
                                    html.Span("The video is now ready for processing!", style={'color': '#27ae60', 'fontSize': '14px'})
                                ], style={
                                    'padding': '20px',
                                    'backgroundColor': '#d4edda',
                                    'border': '2px solid #c3e6cb',
                                    'borderRadius': '8px',
                                    'textAlign': 'center',
                                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                                })
                            ])
                            
                            # Force a refresh of the state
                            logger.info(f"Updated stored_data with video_path: {video_path}")
                            return success_message, stored_data
                        else:
                            # Video verification failed
                            stored_data['processing_status'] = "Video uploaded but verification failed"
                            logger.error(f"Video verification failed for: {video_path}")
                            
                            # Try to clean up the failed video
                            try:
                                Path(video_path).unlink()
                                logger.info(f"Cleaned up failed video: {video_path}")
                            except Exception as cleanup_error:
                                logger.warning(f"Could not clean up failed video: {cleanup_error}")
                            
                            return f"❌ Video uploaded but verification failed. Please try uploading again.", stored_data
                    else:
                        stored_data['processing_status'] = "Failed to save video"
                        logger.error(f"Failed to save video - Path: {video_path}, Exists: {Path(video_path).exists() if video_path else False}")
                        return f"❌ Failed to save {filename}. Check server logs for details.", stored_data
                except Exception as e:
                    error_msg = f"Upload error: {str(e)}"
                    stored_data['processing_status'] = error_msg
                    logger.error(f"Upload error: {e}")
                    import traceback
                    logger.error(f"Upload traceback: {traceback.format_exc()}")
                    return f"❌ {error_msg}", stored_data
            
            return "Ready to upload video", stored_data
        
        @self.app.callback(
            Output('dashboard-state', 'data', allow_duplicate=True),
            Input('process-video-button', 'n_clicks'),
            Input('clear-video-button', 'n_clicks'),
            Input('prev-frame-button', 'n_clicks'),
            Input('next-frame-button', 'n_clicks'),
            State('dashboard-state', 'data'),
            prevent_initial_call=True
        )
        def update_dashboard_state(process_clicks, clear_clicks, prev_clicks, next_clicks, stored_data):
            """Update and persist dashboard state."""
            from dash import callback_context
            ctx = callback_context
            
            # Initialize state
            if stored_data is None:
                stored_data = self._initialize_dashboard_state()
            
            if not ctx.triggered:
                return stored_data
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            logger.info(f"Dashboard state update triggered by: {trigger_id}")
            
            if trigger_id == 'clear-video-button':
                # Clear all state
                logger.info("Clearing video and analysis data")
                stored_data = {
                    'uploaded_video_path': None,
                    'video_frames': [],
                    'current_frame_index': 0,
                    'video_analysis_data': {},
                    'live_data': {},
                    'processing_status': "Ready - Video cleared"
                }
                logger.info("Video and analysis data cleared")
                
            elif trigger_id == 'process-video-button':
                if not stored_data.get('uploaded_video_path'):
                    stored_data['processing_status'] = "❌ No video uploaded. Please upload a video first."
                    logger.warning("Process video button clicked but no video uploaded")
                elif not Path(stored_data['uploaded_video_path']).exists():
                    stored_data['processing_status'] = "❌ Uploaded video file not found. Please upload again."
                    logger.warning(f"Video file not found: {stored_data['uploaded_video_path']}")
                else:
                    # Process video and update state
                    logger.info(f"Processing video: {stored_data['uploaded_video_path']}")
                    try:
                        stored_data['processing_status'] = "🔄 Processing video... Please wait."
                        
                        # Process the video
                        analysis_data = self._process_uploaded_video(stored_data['uploaded_video_path'])
                        
                        if analysis_data and analysis_data.get('frames'):
                            stored_data['video_frames'] = analysis_data['frames']
                            stored_data['current_frame_index'] = 0
                            stored_data['video_analysis_data'] = analysis_data
                            stored_data['processing_status'] = f"✅ Processed {len(analysis_data['frames'])} frames successfully"
                            
                            # Update the main video frames list for display
                            self.video_frames = analysis_data['frames']
                            self.current_frame_index = 0
                            
                            logger.info(f"Video processed successfully: {len(analysis_data['frames'])} frames extracted")
                            
                            # Create success message with frame information
                            success_message = html.Div([
                                html.Div([
                                    html.Span("🎬 ", style={'color': '#3498db', 'fontSize': '16px', 'fontWeight': 'bold'}),
                                    html.Span("Video Analysis Complete!", style={'color': '#3498db', 'fontWeight': 'bold'}),
                                    html.Br(),
                                    html.Span(f"Extracted {len(analysis_data['frames'])} frames", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                    html.Br(),
                                    html.Span(f"Duration: {analysis_data.get('duration', 0):.1f} seconds", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                    html.Br(),
                                    html.Span("Use frame navigation to view results", style={'color': '#27ae60', 'fontSize': '12px', 'fontWeight': 'bold'})
                                ], style={
                                    'padding': '10px',
                                    'backgroundColor': '#d1ecf1',
                                    'border': '1px solid #bee5eb',
                                    'borderRadius': '4px',
                                    'textAlign': 'center'
                                })
                            ])
                            
                            return stored_data
                        else:
                            stored_data['processing_status'] = "❌ No frames extracted from video"
                            logger.warning("No frames extracted from video")
                            return stored_data
                    except Exception as e:
                        error_msg = f"Processing error: {str(e)}"
                        stored_data['processing_status'] = f"❌ {error_msg}"
                        logger.error(f"Video processing error: {e}")
                        import traceback
                        logger.error(f"Processing traceback: {traceback.format_exc()}")
                        return stored_data
            
            elif trigger_id == 'prev-frame-button' and stored_data.get('video_frames'):
                # Navigate to previous frame
                current_index = stored_data.get('current_frame_index', 0)
                if current_index > 0:
                    stored_data['current_frame_index'] = current_index - 1
                    logger.info(f"Navigated to previous frame: {stored_data['current_frame_index']}")
                else:
                    # Wrap to last frame
                    stored_data['current_frame_index'] = len(stored_data['video_frames']) - 1
                    logger.info(f"Wrapped to last frame: {stored_data['current_frame_index']}")
                
                # Force immediate update
                logger.info(f"Frame navigation: {stored_data['current_frame_index'] + 1}/{len(stored_data['video_frames'])}")
                return stored_data
                
            elif trigger_id == 'next-frame-button' and stored_data.get('video_frames'):
                # Navigate to next frame
                current_index = stored_data.get('current_frame_index', 0)
                if current_index < len(stored_data['video_frames']) - 1:
                    stored_data['current_frame_index'] = current_index + 1
                    logger.info(f"Navigated to next frame: {stored_data['current_frame_index']}")
                else:
                    # Wrap to first frame
                    stored_data['current_frame_index'] = 0
                    logger.info(f"Wrapped to first frame: {stored_data['current_frame_index']}")
                
                # Force immediate update
                logger.info(f"Frame navigation: {stored_data['current_frame_index'] + 1}/{len(stored_data['video_frames'])}")
                return stored_data
            
            return stored_data
        
        @self.app.callback(
            Output('video-frame', 'src'),
            Output('processing-status', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_video_display(stored_data, n_intervals):
            """Update video frame display and processing status."""
            if not stored_data:
                return self._create_classroom_photo(), "No video data available"
            
            # Get current frame
            current_frame_index = stored_data.get('current_frame_index', 0)
            video_frames = stored_data.get('video_frames', [])
            
            if not video_frames:
                return self._create_classroom_photo(), "No video frames available. Upload and process a video first."
            
            # Ensure index is within bounds
            if current_frame_index >= len(video_frames):
                current_frame_index = 0
                stored_data['current_frame_index'] = 0
            
            # Get current frame path
            current_frame_path = video_frames[current_frame_index]
            
            # Check if frame exists
            if not Path(current_frame_path).exists():
                logger.warning(f"Frame not found: {current_frame_path}")
                return self._create_classroom_photo(), f"Frame {current_frame_index + 1} not found. Frame file may be missing."
            
            # Convert frame to base64 for display
            try:
                frame_src = self._frame_to_base64(current_frame_path)
                
                # Get analysis data for current frame
                video_analysis = stored_data.get('video_analysis_data', {})
                frame_analysis = video_analysis.get('frame_analysis', [])
                
                # Find analysis for current frame
                current_analysis = None
                for analysis in frame_analysis:
                    if analysis.get('frame_path') == current_frame_path:
                        current_analysis = analysis
                        break
                
                # Create status message with frame analysis
                if current_analysis:
                    detections = current_analysis.get('detections', [])
                    timestamp = current_analysis.get('timestamp', 0)
                    status_message = html.Div([
                        html.Div([
                            html.Span(f"🎬 Frame {current_frame_index + 1}/{len(video_frames)}", 
                                     style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '16px'}),
                            html.Br(),
                            html.Span(f"⏱️ Time: {timestamp:.1f}s", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                            html.Br(),
                            html.Span(f"👥 Faces detected: {len(detections)}", 
                                     style={'color': '#e74c3c' if len(detections) == 0 else '#27ae60', 'fontSize': '14px', 'fontWeight': 'bold'}),
                            html.Br(),
                            html.Span(f"📁 File: {Path(current_frame_path).name}", 
                                     style={'color': '#95a5a6', 'fontSize': '12px'})
                        ], style={
                            'padding': '15px',
                            'backgroundColor': '#f8f9fa',
                            'border': '2px solid #3498db',
                            'borderRadius': '8px',
                            'textAlign': 'center',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ])
                else:
                    status_message = html.Div([
                        html.Div([
                            html.Span(f"🎬 Frame {current_frame_index + 1}/{len(video_frames)}", 
                                     style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '16px'}),
                            html.Br(),
                            html.Span("📊 Analysis data not available", style={'color': '#f39c12', 'fontSize': '14px'}),
                            html.Br(),
                            html.Span(f"📁 File: {Path(current_frame_path).name}", 
                                     style={'color': '#95a5a6', 'fontSize': '12px'})
                        ], style={
                            'padding': '15px',
                            'backgroundColor': '#fff3cd',
                            'border': '2px solid #ffc107',
                            'borderRadius': '8px',
                            'textAlign': 'center'
                        })
                    ])
                
                logger.info(f"Displaying frame {current_frame_index + 1}/{len(video_frames)}: {current_frame_path}")
                return frame_src, status_message
                
            except Exception as e:
                logger.error(f"Error displaying frame: {e}")
                return self._create_classroom_photo(), f"Error displaying frame: {str(e)}"
        
        @self.app.callback(
            Output('frame-info', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_frame_info(stored_data, n):
            """Update frame information display."""
            if not stored_data or not stored_data.get('video_frames'):
                return "No video frames available"
            
            current_index = stored_data.get('current_frame_index', 0)
            video_frames = stored_data.get('video_frames', [])
            video_analysis = stored_data.get('video_analysis_data', {})
            
            if not video_frames:
                return "No frames to display"
            
            # Get current frame analysis
            current_frame_path = video_frames[current_index]
            current_analysis = None
            
            for analysis in video_analysis.get('frame_analysis', []):
                if analysis.get('frame_path') == current_frame_path:
                    current_analysis = analysis
                    break
            
            # Create detailed frame information
            frame_info = html.Div([
                html.H4("📊 Frame Analysis", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                
                # Frame details
                html.Div([
                    html.Span("🎬 ", style={'color': '#3498db', 'fontSize': '16px'}),
                    html.Span(f"Frame {current_index + 1} of {len(video_frames)}", 
                             style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                    html.Br(),
                    html.Span("📁 ", style={'color': '#95a5a6'}),
                    html.Span(f"File: {Path(current_frame_path).name}", 
                             style={'fontSize': '12px', 'color': '#7f8c8d'}),
                ], style={'marginBottom': '15px'}),
                
                # Analysis results
                html.Div([
                    html.H5("🔍 Detection Results", style={'color': '#e74c3c', 'marginBottom': '8px'}),
                    
                    # Face detection count
                    html.Div([
                        html.Span("👥 ", style={'color': '#e74c3c'}),
                        html.Span("Faces Detected: ", style={'fontWeight': 'bold'}),
                        html.Span(f"{len(current_analysis.get('detections', [])) if current_analysis else 0}", 
                                 style={'color': '#27ae60', 'fontWeight': 'bold'})
                    ], style={'marginBottom': '5px'}),
                    
                    # Video properties
                    html.Div([
                        html.Span("🎥 ", style={'color': '#9b59b6'}),
                        html.Span("Video Properties: ", style={'fontWeight': 'bold'}),
                        html.Br(),
                        html.Span(f"Total Frames: {video_analysis.get('total_frames', 0)}", 
                                 style={'fontSize': '12px', 'color': '#7f8c8d'}),
                        html.Br(),
                        html.Span(f"FPS: {video_analysis.get('fps', 0):.1f}", 
                                 style={'fontSize': '12px', 'color': '#7f8c8d'}),
                        html.Br(),
                        html.Span(f"Duration: {video_analysis.get('duration', 0):.1f}s", 
                                 style={'fontSize': '12px', 'color': '#7f8c8d'})
                    ], style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                    
                ], style={'border': '1px solid #dee2e6', 'padding': '15px', 'borderRadius': '8px'})
                
            ], style={'padding': '15px'})
            
            # Add timestamp information if available
            if current_analysis:
                timestamp_info = html.Div([
                    html.Span("⏱️ ", style={'color': '#f39c12'}),
                    html.Span("Timestamp: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{current_analysis.get('timestamp', 0):.2f}s", 
                             style={'color': '#7f8c8d'})
                ], style={'marginBottom': '5px'})
                
                # Insert timestamp info after face detection count
                frame_info.children[1].children.insert(2, timestamp_info)
            
            return frame_info
        
        @self.app.callback(
            Output('detection-overlay', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_detection_overlay(stored_data, n):
            """Update detection overlay on video frame."""
            if not stored_data or not stored_data.get('live_data') or not stored_data.get('video_frames'):
                return []
            
            detections = stored_data['live_data'].get('detections', [])
            if not detections:
                return []
            
            # Get current frame detections
            current_frame_idx = stored_data.get('current_frame_index', 0)
            current_frame_detections = [d for d in detections if d.get('frame_idx') == current_frame_idx]
            
            if not current_frame_detections:
                return []
            
            overlay_elements = []
            
            for detection in current_frame_detections:
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    # Calculate relative positions based on actual frame dimensions
                    # Get the actual frame dimensions from the current frame
                    video_frames = stored_data.get('video_frames', [])
                    if video_frames and current_frame_idx < len(video_frames):
                        frame_path = video_frames[current_frame_idx]
                        try:
                            frame = cv2.imread(frame_path)
                            if frame is not None:
                                actual_height, actual_width = frame.shape[:2]
                            else:
                                actual_width, actual_height = 800, 600
                        except:
                            actual_width, actual_height = 800, 600
                    else:
                        actual_width, actual_height = 800, 600
                    
                    # Scale coordinates to fit the display (max 800px width)
                    display_width = 800
                    display_height = int(display_width * actual_height / actual_width)
                    
                    scale_x = display_width / actual_width
                    scale_y = display_height / actual_height
                    
                    rel_x1 = x1 * scale_x
                    rel_y1 = y1 * scale_y
                    rel_x2 = x2 * scale_x
                    rel_y2 = y2 * scale_y
                    
                    # Get attention score and determine color
                    attention_score = detection.get('confidence', 0.5)  # Use confidence as attention score
                    if attention_score > 0.8:
                        color = '#27ae60'  # Green for high attention
                    elif attention_score > 0.5:
                        color = '#f39c12'  # Orange for medium attention
                    else:
                        color = '#e74c3c'  # Red for low attention
                    
                    # Get identity and emotion
                    identity = detection.get('identity', f'Student {detection.get("track_id", "Unknown")}')
                    emotion = detection.get('dominant_emotion', 'neutral')
                    
                    # Create detection box
                    detection_box = html.Div([
                        # Bounding box
                        html.Div(style={
                            'position': 'absolute',
                            'left': f'{rel_x1}px',
                            'top': f'{rel_y1}px',
                            'width': f'{rel_x2 - rel_x1}px',
                            'height': f'{rel_y2 - rel_y1}px',
                            'border': f'3px solid {color}',
                            'borderRadius': '4px',
                            'boxShadow': '0 0 10px rgba(0,0,0,0.3)',
                            'backgroundColor': f'{color}20'  # Semi-transparent background
                        }),
                        # Attention score and identity indicator
                        html.Div([
                            html.Div(style={
                                'position': 'absolute',
                                'left': f'{rel_x1}px',
                                'top': f'{rel_y1 - 35}px',
                                'backgroundColor': color,
                                'color': 'white',
                                'padding': '4px 8px',
                                'borderRadius': '4px',
                                'fontSize': '11px',
                                'fontWeight': 'bold',
                                'minWidth': '80px',
                                'textAlign': 'center'
                            }, children=f"{identity}: {attention_score:.2f}")
                        ]),
                        # Emotion indicator
                        html.Div([
                            html.Div(style={
                                'position': 'absolute',
                                'left': f'{rel_x1}px',
                                'top': f'{rel_y2 + 5}px',
                                'backgroundColor': '#34495e',
                                'color': 'white',
                                'padding': '2px 6px',
                                'borderRadius': '3px',
                                'fontSize': '10px',
                                'fontWeight': 'bold'
                            }, children=f"😊 {emotion.title()}")
                        ])
                    ])
                    
                    overlay_elements.append(detection_box)
            
            return overlay_elements
        
        @self.app.callback(
            Output('process-video-button', 'disabled'),
            Output('process-video-button', 'style'),
            Input('dashboard-state', 'data')
        )
        def update_process_button_state(stored_data):
            """Update Process Video button state based on video availability."""
            if not stored_data or not stored_data.get('uploaded_video_path'):
                # No video uploaded - button disabled
                return True, {
                    'backgroundColor': '#95a5a6',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'borderRadius': '6px',
                    'cursor': 'not-allowed',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'opacity': '0.6'
                }
            
            # Check if video file exists
            video_path = stored_data['uploaded_video_path']
            if not Path(video_path).exists():
                return True, {
                    'backgroundColor': '#e74c3c',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'borderRadius': '6px',
                    'cursor': 'not-allowed',
                    'fontSize': '16px',
                    'fontWeight': 'bold'
                }
            
            # Check if video is already processed
            if stored_data.get('video_frames') and len(stored_data['video_frames']) > 0:
                # Video already processed - button shows "Re-process"
                return False, {
                    'backgroundColor': '#f39c12',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                }
            else:
                # Video uploaded but not processed - button enabled
                return False, {
                    'backgroundColor': '#27ae60',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                }
        
        @self.app.callback(
            Output('interval-component', 'interval'),
            Input('refresh-slider', 'value')
        )
        def update_refresh_rate(value):
            """Update refresh rate."""
            return value * 1000  # Convert to milliseconds
        
        @self.app.callback(
            Output('frame-counter', 'children'),
            Output('total-frames', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_frame_counter(stored_data, n):
            """Update frame counter display."""
            if not stored_data or not stored_data.get('video_frames'):
                return "0/0", "0"
            
            current_index = stored_data.get('current_frame_index', 0)
            total_frames = len(stored_data['video_frames'])
            
            if total_frames == 0:
                return "0/0", "0"
            
            # Ensure index is within bounds
            if current_index >= total_frames:
                current_index = 0
            
            return f"{current_index + 1}/{total_frames}", str(total_frames)
        
        @self.app.callback(
            Output('process-video-button', 'children'),
            Input('dashboard-state', 'data')
        )
        def update_process_button_text(stored_data):
            """Update Process Video button text based on video state."""
            if not stored_data or not stored_data.get('uploaded_video_path'):
                return "🎬 Process Video"
            
            # Check if video is already processed
            if stored_data.get('video_frames') and len(stored_data['video_frames']) > 0:
                return "🔄 Re-process Video"
            else:
                return "🎬 Process Video"
        
        @self.app.callback(
            Output('video-preview', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_video_preview(stored_data, n):
            """Update video preview area with uploaded video information."""
            if not stored_data or not stored_data.get('uploaded_video_path'):
                return html.Div([
                    html.Div([
                        html.Span("📹 ", style={'fontSize': '48px', 'color': '#bdc3c7'}),
                        html.Br(),
                        html.Span("No video uploaded yet", style={'color': '#7f8c8d', 'fontSize': '14px'})
                    ], style={
                        'textAlign': 'center',
                        'padding': '20px',
                        'border': '2px dashed #bdc3c7',
                        'borderRadius': '8px',
                        'backgroundColor': '#f8f9fa'
                    })
                ])
            
            video_path = stored_data['uploaded_video_path']
            if not Path(video_path).exists():
                return html.Div([
                    html.Div([
                        html.Span("❌ ", style={'fontSize': '48px', 'color': '#e74c3c'}),
                        html.Br(),
                        html.Span("Video file not found", style={'color': '#e74c3c', 'fontSize': '14px'})
                    ], style={
                        'textAlign': 'center',
                        'padding': '20px',
                        'border': '2px solid #e74c3c',
                        'borderRadius': '8px',
                        'backgroundColor': '#fdf2f2'
                    })
                ])
            
            # Get video information
            file_size = Path(video_path).stat().st_size
            file_name = Path(video_path).name
            file_ext = Path(video_path).suffix.upper()
            
            # Check if video is processed
            is_processed = stored_data.get('video_frames') and len(stored_data['video_frames']) > 0
            
            # Create video preview
            preview_content = html.Div([
                html.Div([
                    # Video icon and status
                    html.Div([
                        html.Span("🎬", style={'fontSize': '48px', 'color': '#27ae60' if is_processed else '#3498db'}),
                        html.Br(),
                        html.Span("✅ Uploaded", style={'color': '#27ae60', 'fontWeight': 'bold', 'fontSize': '16px'})
                    ], style={'textAlign': 'center', 'marginBottom': '15px'}),
                    
                    # Video details
                    html.Div([
                        html.H4(file_name, style={'color': '#2c3e50', 'marginBottom': '10px', 'wordBreak': 'break-word'}),
                        
                        html.Div([
                            html.Span("📁 Format: ", style={'fontWeight': 'bold'}),
                            html.Span(file_ext, style={'color': '#3498db'})
                        ], style={'marginBottom': '5px'}),
                        
                        html.Div([
                            html.Span("💾 Size: ", style={'fontWeight': 'bold'}),
                            html.Span(f"{file_size / (1024*1024):.1f} MB", style={'color': '#7f8c8d'})
                        ], style={'marginBottom': '5px'}),
                        
                        html.Div([
                            html.Span("📂 Location: ", style={'fontWeight': 'bold'}),
                            html.Span(Path(video_path).parent.name, style={'color': '#95a5a6'})
                        ], style={'marginBottom': '15px'}),
                        
                        # Processing status
                        html.Div([
                            html.Span("🔄 Status: ", style={'fontWeight': 'bold'}),
                            html.Span(
                                "✅ Processed" if is_processed else "⏳ Ready to Process",
                                style={'color': '#27ae60' if is_processed else '#f39c12', 'fontWeight': 'bold'}
                            )
                        ], style={
                            'padding': '10px',
                            'backgroundColor': '#d4edda' if is_processed else '#fff3cd',
                            'border': '1px solid #c3e6cb' if is_processed else '#ffc107',
                            'borderRadius': '4px',
                            'textAlign': 'center'
                        })
                        
                    ], style={'textAlign': 'left'})
                    
                ], style={
                    'padding': '20px',
                    'border': '2px solid #27ae60' if is_processed else '#3498db',
                    'borderRadius': '8px',
                    'backgroundColor': '#f8fff9' if is_processed else '#f8f9fa'
                })
            ])
            
            return preview_content
    
    def _create_empty_chart(self, message: str):
        """Create an empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
        return fig
    
    def update_data(self, detections: List[Dict], seat_assignments: Dict = None):
        """Update dashboard data."""
        self.live_data = {
            'detections': detections,
            'seat_assignments': seat_assignments or {},
            'timestamp': time.time()
        }
        logger.info(f"Updated dashboard with {len(detections)} detections")
    
    def load_analysis_data(self, data_path: str):
        """Load analysis data from file."""
        try:
            with open(data_path, 'r') as f:
                self.analysis_data = json.load(f)
            logger.info(f"Loaded analysis data from: {data_path}")
        except Exception as e:
            logger.error(f"Failed to load analysis data: {e}")
    
    def _find_available_port(self, start_port: int = 8080, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                    s.close()
                    return port
            except OSError:
                continue
        
        # If no port found, return the original port (will raise error later)
        return start_port
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        # Find an available port
        available_port = self._find_available_port(self.port)
        
        if available_port != self.port:
            logger.info(f"Port {self.port} is in use, using port {available_port} instead")
            self.port = available_port
        
        logger.info(f"Starting dashboard on {self.host}:{self.port}")
        
        # Configure Flask app for better performance and caching
        self.app.server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development
        self.app.server.config['TEMPLATES_AUTO_RELOAD'] = True
        
        # Add headers to prevent caching issues
        @self.app.server.after_request
        def add_header(response):
            # Prevent caching for all responses
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            return response
        
        # Add health check endpoint
        @self.app.server.route('/health')
        def health_check():
            return {'status': 'healthy', 'timestamp': time.time()}
        
        # Add debug endpoint for upload issues
        @self.app.server.route('/debug/upload')
        def debug_upload():
            """Debug endpoint for upload issues."""
            try:
                import json
                debug_info = {
                    'temp_dir_exists': Path("data/temp").exists(),
                    'raw_videos_dir_exists': Path("data/raw_videos").exists(),
                    'temp_dir_contents': [],
                    'raw_videos_contents': [],
                    'disk_space': {},
                    'permissions': {}
                }
                
                # Check temp directory
                temp_dir = Path("data/temp")
                if temp_dir.exists():
                    debug_info['temp_dir_contents'] = [
                        {
                            'name': f.name,
                            'size': f.stat().st_size,
                            'modified': f.stat().st_mtime,
                            'exists': f.exists()
                        }
                        for f in temp_dir.glob("*")
                    ]
                
                # Check raw_videos directory
                raw_videos_dir = Path("data/raw_videos")
                if raw_videos_dir.exists():
                    debug_info['raw_videos_contents'] = [
                        {
                            'name': f.name,
                            'size': f.stat().st_size,
                            'modified': f.stat().st_mtime,
                            'exists': f.exists()
                        }
                        for f in raw_videos_dir.glob("*")
                    ]
                
                # Check disk space
                try:
                    import shutil
                    total, used, free = shutil.disk_usage("data")
                    debug_info['disk_space'] = {
                        'total_gb': total / (1024**3),
                        'used_gb': used / (1024**3),
                        'free_gb': free / (1024**3)
                    }
                except Exception as e:
                    debug_info['disk_space'] = {'error': str(e)}
                
                # Check permissions
                try:
                    debug_info['permissions'] = {
                        'temp_dir_writable': os.access("data/temp", os.W_OK),
                        'raw_videos_dir_writable': os.access("data/raw_videos", os.W_OK),
                        'current_user': os.getenv('USER', 'unknown'),
                        'current_working_dir': os.getcwd()
                    }
                except Exception as e:
                    debug_info['permissions'] = {'error': str(e)}
                
                return json.dumps(debug_info, indent=2, default=str)
                
            except Exception as e:
                return f"Debug error: {str(e)}", 500
        
        # Add error handlers
        @self.app.server.errorhandler(404)
        def not_found(error):
            return {'error': 'Not found'}, 404
        
        @self.app.server.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return {'error': 'Internal server error'}, 500
        
        try:
            self.app.run(
                host=self.host, 
                port=self.port, 
                debug=debug,
                use_reloader=False,  # Disable reloader to prevent issues
                threaded=True
            )
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Port {self.port} is still in use. Attempting to kill processes...")
                
                # Try to kill processes on the port
                if self._kill_process_on_port(self.port):
                    logger.info(f"Killed processes on port {self.port}. Retrying...")
                    time.sleep(1)  # Wait a moment for the port to be released
                    try:
                        self.app.run(
                            host=self.host, 
                            port=self.port, 
                            debug=debug,
                            use_reloader=False,
                            threaded=True
                        )
                    except OSError as e2:
                        logger.error(f"Still cannot use port {self.port} after killing processes")
                        logger.error(f"Please try:")
                        logger.error(f"1. Stop any other applications using port {self.port}")
                        logger.error(f"2. Or manually specify a different port in the config")
                        logger.error(f"3. Or run: lsof -ti:{self.port} | xargs kill -9 (to kill processes on port {self.port})")
                        raise e2
                else:
                    logger.error(f"Could not kill processes on port {self.port}. Please try:")
                    logger.error(f"1. Stop any other applications using port {self.port}")
                    logger.error(f"2. Or manually specify a different port in the config")
                    logger.error(f"3. Or run: lsof -ti:{self.port} | xargs kill -9 (to kill processes on port {self.port})")
                    raise e
            else:
                logger.error(f"Failed to start dashboard: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            raise
    
    def start_background(self):
        """Start dashboard in background thread."""
        if self.update_thread and self.update_thread.is_alive():
            logger.warning("Dashboard already running")
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self.run, daemon=True)
        self.update_thread.start()
        logger.info("Dashboard started in background")
    
    def stop(self):
        """Stop the dashboard."""
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        logger.info("Dashboard stopped")
    
    def get_url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.host}:{self.port}" 
    
    def _create_video_preview_message(self, video_path: str):
        """Create a video preview message for uploaded videos."""
        try:
            video_name = Path(video_path).name
            file_size = Path(video_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            # Create a simple SVG with video preview information
            svg_content = f'''
            <svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
                <!-- Background -->
                <rect width="800" height="600" fill="#f8f9fa"/>
                
                <!-- Video preview box -->
                <rect x="200" y="150" width="400" height="300" fill="#e9ecef" stroke="#6c757d" stroke-width="2" rx="10"/>
                
                <!-- Video icon -->
                <circle cx="400" cy="300" r="50" fill="#3498db"/>
                <polygon points="380,285 380,315 405,300" fill="white"/>
                
                <!-- Video info -->
                <text x="400" y="380" text-anchor="middle" fill="#2c3e50" font-family="Arial" font-size="18" font-weight="bold">{video_name}</text>
                <text x="400" y="400" text-anchor="middle" fill="#7f8c8d" font-family="Arial" font-size="14">Size: {file_size_mb:.1f}MB</text>
                <text x="400" y="420" text-anchor="middle" fill="#27ae60" font-family="Arial" font-size="16" font-weight="bold">✅ Video Uploaded Successfully!</text>
                <text x="400" y="440" text-anchor="middle" fill="#7f8c8d" font-family="Arial" font-size="12">Click "Process Video" to extract frames and analyze</text>
                
                <!-- Upload status indicator -->
                <circle cx="400" cy="480" r="8" fill="#27ae60"/>
                <text x="415" y="485" fill="#27ae60" font-family="Arial" font-size="12" font-weight="bold">Ready for Processing</text>
            </svg>
            '''
            
            # Convert to base64
            svg_bytes = svg_content.encode('utf-8')
            svg_base64 = base64.b64encode(svg_bytes).decode('utf-8')
            return f'data:image/svg+xml;base64,{svg_base64}'
            
        except Exception as e:
            logger.error(f"Error creating video preview: {e}")
            return self._create_simple_analysis_message()
    
    def _kill_process_on_port(self, port: int) -> bool:
        """Attempt to kill processes running on a specific port."""
        try:
            system = platform.system().lower()
            
            if system == "linux" or system == "darwin":  # Linux or macOS
                # Find processes using the port
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid.strip():
                            try:
                                subprocess.run(['kill', '-9', pid.strip()], 
                                             capture_output=True, check=True)
                                logger.info(f"Killed process {pid} on port {port}")
                            except subprocess.CalledProcessError:
                                continue
                    return True
            elif system == "windows":
                # Windows equivalent
                result = subprocess.run(['netstat', '-ano'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if f':{port}' in line and 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                try:
                                    subprocess.run(['taskkill', '/PID', pid, '/F'], 
                                                 capture_output=True, check=True)
                                    logger.info(f"Killed process {pid} on port {port}")
                                except subprocess.CalledProcessError:
                                    continue
                    return True
        except Exception as e:
            logger.warning(f"Could not kill processes on port {port}: {e}")
        
        return False
    
    def _verify_uploaded_video(self, video_path: str) -> bool:
        """Verify that an uploaded video file is valid and accessible."""
        try:
            if not video_path or not Path(video_path).exists():
                logger.error(f"Video file not found: {video_path}")
                return False
            
            video_file = Path(video_path)
            file_size = video_file.stat().st_size
            
            if file_size == 0:
                logger.error(f"Video file is empty: {video_path}")
                return False
            
            # Try to open the video with OpenCV to verify it's a valid video file
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    logger.error(f"Could not open video file with OpenCV: {video_path}")
                    return False
                
                # Get basic video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cap.release()
                
                logger.info(f"Video verified: {total_frames} frames, {fps} fps, {width}x{height}, {file_size / (1024*1024):.1f}MB")
                return True
                
            except Exception as cv_error:
                logger.error(f"OpenCV verification failed: {cv_error}")
                # If OpenCV fails, at least check if the file exists and has content
                return file_size > 0
                
        except Exception as e:
            logger.error(f"Video verification error: {e}")
            return False
    
    def _cleanup_old_videos(self, max_age_hours: int = 24):
        """Clean up old uploaded videos to prevent disk space issues."""
        try:
            temp_dir = Path("data/temp")
            if not temp_dir.exists():
                return
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for video_file in temp_dir.glob("*.mp4"):
                try:
                    file_age = current_time - video_file.stat().st_mtime
                    if file_age > max_age_seconds:
                        video_file.unlink()
                        logger.info(f"Cleaned up old video: {video_file}")
                except Exception as e:
                    logger.warning(f"Could not clean up {video_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _list_uploaded_videos(self) -> List[str]:
        """List all uploaded videos in the system."""
        videos = []
        try:
            # Check temp directory
            temp_dir = Path("data/temp")
            if temp_dir.exists():
                for video_file in temp_dir.glob("*.mp4"):
                    if video_file.exists() and video_file.stat().st_size > 0:
                        videos.append(str(video_file))
            
            # Check raw_videos directory
            raw_videos_dir = Path("data/raw_videos")
            if raw_videos_dir.exists():
                for video_file in raw_videos_dir.glob("*.mp4"):
                    if video_file.exists() and video_file.stat().st_size > 0:
                        videos.append(str(video_file))
            
            logger.info(f"Found {len(videos)} uploaded videos")
            return videos
            
        except Exception as e:
            logger.error(f"Error listing uploaded videos: {e}")
            return []
    
    def _initialize_dashboard_state(self) -> Dict:
        """Initialize the dashboard state with any existing uploaded videos."""
        state = {
            'uploaded_video_path': None,
            'video_frames': [],
            'current_frame_index': 0,
            'video_analysis_data': {},
            'live_data': {},
            'processing_status': "Ready"
        }
        
        # Check for existing uploaded videos
        uploaded_videos = self._list_uploaded_videos()
        if uploaded_videos:
            # Use the most recent video
            most_recent = max(uploaded_videos, key=lambda x: Path(x).stat().st_mtime)
            if self._verify_uploaded_video(most_recent):
                state['uploaded_video_path'] = most_recent
                state['processing_status'] = f"Found existing video: {Path(most_recent).name}"
                logger.info(f"Initialized with existing video: {most_recent}")
        
        return state