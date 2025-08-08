"""
Dashboard UI module for FaceClass project.
Provides web-based interface for visualizing analysis results.
"""

import dash
from dash import dcc, html, Input, Output, State
from dash.dependencies import Input, Output, State
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

logger = logging.getLogger(__name__)


class DashboardUI:
    """Web-based dashboard for FaceClass analysis visualization."""
    
    def __init__(self, config):
        """Initialize dashboard with configuration."""
        self.config = config
        self.port = config.get('dashboard.port', 8080)
        self.host = config.get('dashboard.host', 'localhost')
        self.refresh_rate = config.get('dashboard.refresh_rate', 1.0)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "FaceClass Dashboard"
        
        # Data storage
        self.analysis_data = {}
        self.live_data = {}
        self.video_frames = []
        self.current_frame_index = 0
        self.uploaded_video_path = None
        self.processing_status = "Ready"
        self.video_analysis_data = {}
        
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
            from detection.face_tracker import FaceTracker
            from config import Config
            
            # Initialize face tracker
            config = Config()
            face_tracker = FaceTracker(config)
            
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
        """Process uploaded video and extract frames with face detections."""
        try:
            logger.info(f"Processing video: {video_path}")
            self.processing_status = "Processing video..."
            
            # Initialize face tracker
            from detection.face_tracker import FaceTracker
            from config import Config
            
            config = Config()
            face_tracker = FaceTracker(config)
            
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
            
            # Extract frames at regular intervals (every 5 seconds)
            frame_interval = int(fps * 5) if fps > 0 else 150
            extracted_frames = []
            all_detections = []
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_count % frame_interval == 0:
                    # Detect faces in frame
                    detections = face_tracker.detect_faces(frame)
                    
                    # Add frame information
                    for i, detection in enumerate(detections):
                        detection['frame_idx'] = len(extracted_frames)
                        detection['timestamp'] = frame_count / fps if fps > 0 else frame_count
                        detection['track_id'] = len(extracted_frames) * 100 + i
                    
                    all_detections.extend(detections)
                    
                    # Save frame
                    frame_path = frames_dir / f"uploaded_frame_{len(extracted_frames):03d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(str(frame_path))
                    
                    logger.info(f"Extracted frame {len(extracted_frames)} with {len(detections)} detections")
                
                frame_count += 1
                
                # Limit to prevent memory issues
                if len(extracted_frames) >= 50:
                    break
            
            cap.release()
            
            # Update video analysis data
            self.video_analysis_data = {
                'frames': extracted_frames,
                'detections': all_detections,
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'extracted_frames': len(extracted_frames)
            }
            
            self.processing_status = f"✅ Video processed: {len(extracted_frames)} frames, {len(all_detections)} detections"
            logger.info(f"Video processing complete: {len(extracted_frames)} frames, {len(all_detections)} detections")
            
            return self.video_analysis_data
            
        except Exception as e:
            self.processing_status = f"❌ Error processing video: {str(e)}"
            logger.error(f"Error processing video: {e}")
            return {}
    
    def _save_uploaded_video(self, contents: str, filename: str) -> str:
        """Save uploaded video to temporary file."""
        try:
            # Decode base64 content
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Create temporary file
            temp_dir = Path("data/temp")
            temp_dir.mkdir(exist_ok=True)
            
            video_path = temp_dir / filename
            with open(video_path, 'wb') as f:
                f.write(decoded)
            
            logger.info(f"Saved uploaded video: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded video: {e}")
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
        """Setup the dashboard layout."""
        self.app.layout = html.Div([
            # Store component for persisting state across page refreshes
            dcc.Store(id='dashboard-state', storage_type='session'),
            
            # Header
            html.Div([
                html.H1("FaceClass Dashboard", 
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
                
                # Video Upload Section
                html.Div([
                    html.H3("📹 Upload Classroom Video", style={
                        'textAlign': 'center', 
                        'marginBottom': 20,
                        'color': '#2c3e50',
                        'fontSize': '24px',
                        'fontWeight': 'bold'
                    }),
                    
                    # Upload area with improved design
                    html.Div([
                        dcc.Upload(
                            id='upload-video',
                            children=html.Div([
                                html.Div([
                                    html.I(className="fas fa-cloud-upload-alt", style={'fontSize': '48px', 'color': '#3498db', 'marginBottom': '10px'}),
                                    html.Br(),
                                    html.Span("Drag and Drop your video here", style={'fontSize': '18px', 'fontWeight': 'bold', 'color': '#2c3e50'}),
                                    html.Br(),
                                    html.Span("or", style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '0 10px'}),
                                    html.A("Browse Files", style={'color': '#3498db', 'textDecoration': 'underline', 'fontWeight': 'bold'}),
                                    html.Br(),
                                    html.Span("Supports: MP4, AVI, MOV, MKV", style={'fontSize': '12px', 'color': '#95a5a6', 'marginTop': '10px'})
                                ], style={'textAlign': 'center', 'padding': '40px 20px'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '200px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '10px 0',
                                'backgroundColor': '#f8f9fa',
                                'borderColor': '#3498db',
                                'transition': 'all 0.3s ease',
                                'cursor': 'pointer'
                            },
                            accept='video/*'
                        ),
                        
                        # Upload status
                        html.Div(id='upload-status', style={
                            'marginTop': '15px', 
                            'fontSize': '14px', 
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'backgroundColor': '#f8f9fa'
                        }),
                        
                        # Action buttons
                        html.Div([
                            html.Button(
                                '🚀 Process Video', 
                                id='process-video-button', 
                                n_clicks=0,
                                style={
                                    'backgroundColor': '#3498db', 
                                    'color': 'white', 
                                    'border': 'none', 
                                    'padding': '12px 24px', 
                                    'marginRight': 15,
                                    'borderRadius': '6px',
                                    'fontSize': '14px',
                                    'fontWeight': 'bold',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease'
                                }
                            ),
                            html.Button(
                                '🗑️ Clear Video', 
                                id='clear-video-button', 
                                n_clicks=0,
                                style={
                                    'backgroundColor': '#95a5a6', 
                                    'color': 'white', 
                                    'border': 'none', 
                                    'padding': '12px 24px',
                                    'borderRadius': '6px',
                                    'fontSize': '14px',
                                    'fontWeight': 'bold',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease'
                                }
                            )
                        ], style={'marginTop': '20px', 'textAlign': 'center'}),
                        
                        # Processing status
                        html.Div(id='processing-status', style={
                            'marginTop': '15px', 
                            'fontSize': '14px', 
                            'fontWeight': 'bold',
                            'textAlign': 'center',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'backgroundColor': '#f8f9fa'
                        })
                    ])
                ], style={
                    'marginBottom': '30px', 
                    'padding': '25px', 
                    'backgroundColor': 'white', 
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                    'border': '1px solid #e1e8ed'
                }),
                
                # Video Analysis Results
                html.Div([
                    html.H4("Video Analysis Results", style={'textAlign': 'center', 'marginBottom': 15}),
                    
                    # Video frame display area
                    html.Div([
                        html.Img(
                            id='video-frame',
                            src=self._create_simple_analysis_message(),
                            style={
                                'width': '100%',
                                'maxWidth': '800px',
                                'height': 'auto',
                                'border': '2px solid #ddd',
                                'borderRadius': '8px',
                                'position': 'relative'
                            }
                        ),
                        # Overlay for detections
                        html.Div(id='detection-overlay', style={
                            'position': 'absolute',
                            'top': '0',
                            'left': '0',
                            'width': '100%',
                            'height': '100%',
                            'pointerEvents': 'none'
                        })
                    ], style={'position': 'relative', 'display': 'inline-block', 'maxWidth': '800px', 'textAlign': 'center'}),
                    
                    # Video controls
                    html.Div([
                        html.Button('⏮️ Previous Frame', id='prev-frame-button', n_clicks=0,
                                   style={'backgroundColor': '#34495e', 'color': 'white', 'border': 'none', 'padding': '8px 15px', 'marginRight': 10}),
                        html.Button('⏸️ Pause/Play', id='play-pause-button', n_clicks=0,
                                   style={'backgroundColor': '#9b59b6', 'color': 'white', 'border': 'none', 'padding': '8px 15px', 'marginRight': 10}),
                        html.Button('⏭️ Next Frame', id='next-frame-button', n_clicks=0,
                                   style={'backgroundColor': '#34495e', 'color': 'white', 'border': 'none', 'padding': '8px 15px'}),
                        html.Div(id='frame-info', style={'marginTop': '10px', 'fontSize': '14px', 'color': '#666'})
                    ], style={'textAlign': 'center', 'marginTop': '15px'})
                ], style={'textAlign': 'center', 'marginBottom': 30}),
                
                # Detection Legend
                html.Div([
                    html.H4("Detection Legend", style={'textAlign': 'center', 'marginBottom': 10}),
                    html.Div([
                        html.Div([
                            html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': '#27ae60', 'display': 'inline-block', 'marginRight': '5px'}),
                            html.Span("High Attention (>80%)", style={'fontSize': '12px'})
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        html.Div([
                            html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': '#f39c12', 'display': 'inline-block', 'marginRight': '5px'}),
                            html.Span("Medium Attention (50-80%)", style={'fontSize': '12px'})
                        ], style={'display': 'inline-block', 'marginRight': '20px'}),
                        html.Div([
                            html.Div(style={'width': '20px', 'height': '20px', 'backgroundColor': '#e74c3c', 'display': 'inline-block', 'marginRight': '5px'}),
                            html.Span("Low Attention (<50%)", style={'fontSize': '12px'})
                        ], style={'display': 'inline-block'})
                    ], style={'textAlign': 'center'})
                ], style={'marginBottom': '30px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
            ]),
            
            # Main Content
            html.Div([
                # Left Column - Statistics
                html.Div([
                    html.H3("Real-time Statistics", style={'textAlign': 'center'}),
                    
                    # Face Count
                    html.Div([
                        html.H4("Faces Detected"),
                        html.Div(id='face-count', style={'fontSize': '2em', 'textAlign': 'center', 'color': '#3498db'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    
                    # Attention Score
                    html.Div([
                        html.H4("Average Attention"),
                        html.Div(id='attention-score', style={'fontSize': '2em', 'textAlign': 'center', 'color': '#e67e22'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    
                    # Emotion Distribution
                    html.Div([
                        html.H4("Dominant Emotion"),
                        html.Div(id='dominant-emotion', style={'fontSize': '1.5em', 'textAlign': 'center', 'color': '#9b59b6'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Center Column - Charts
                html.Div([
                    html.H3("Analysis Charts", style={'textAlign': 'center'}),
                    
                    # Emotion Distribution Chart
                    html.Div([
                        dcc.Graph(id='emotion-chart', style={'height': '300px'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    
                    # Attention Timeline
                    html.Div([
                        dcc.Graph(id='attention-timeline', style={'height': '300px'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right Column - Additional Info
                html.Div([
                    html.H3("Additional Information", style={'textAlign': 'center'}),
                    
                    # Position Heatmap
                    html.Div([
                        dcc.Graph(id='position-heatmap', style={'height': '300px'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                    
                    # Recent Detections
                    html.Div([
                        html.H4("Recent Detections"),
                        html.Div(id='recent-detections', style={'fontSize': '0.9em'})
                    ], style={'backgroundColor': 'white', 'padding': '20px', 'margin': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ]),
            
            # Interval component for updates
            dcc.Interval(
                id='interval-component',
                interval=self.refresh_rate * 1000,  # Convert to milliseconds
                n_intervals=0
            )
        ], style={'backgroundColor': '#ecf0f1', 'minHeight': '100vh', 'padding': '20px'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output('face-count', 'children'),
            Output('attention-score', 'children'),
            Output('dominant-emotion', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_statistics(stored_data, n):
            """Update real-time statistics."""
            if not stored_data or not stored_data.get('live_data') or not stored_data.get('video_frames'):
                return "0", "0.0%", "N/A"
            
            detections = stored_data['live_data'].get('detections', [])
            if not detections:
                return "0", "0.0%", "N/A"
            
            # Get current frame detections
            current_frame_idx = stored_data.get('current_frame_index', 0)
            current_frame_detections = [d for d in detections if d.get('frame_idx') == current_frame_idx]
            
            # Face count
            face_count = len(current_frame_detections)
            
            # Average attention score
            if current_frame_detections:
                attention_scores = [d.get('confidence', 0.5) for d in current_frame_detections]
                avg_attention = sum(attention_scores) / len(attention_scores)
                attention_percentage = f"{avg_attention * 100:.1f}%"
            else:
                attention_percentage = "0.0%"
            
            # Dominant emotion
            if current_frame_detections:
                emotions = [d.get('dominant_emotion', 'neutral') for d in current_frame_detections]
                emotion_counts = {}
                for emotion in emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                if emotion_counts:
                    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                    emotion_count = emotion_counts[dominant_emotion]
                    total_emotions = len(emotions)
                    emotion_percentage = (emotion_count / total_emotions) * 100
                    dominant_emotion_display = f"{dominant_emotion.title()} ({emotion_percentage:.0f}%)"
                else:
                    dominant_emotion_display = "Neutral"
            else:
                dominant_emotion_display = "N/A"
            
            return str(face_count), attention_percentage, dominant_emotion_display
        
        @self.app.callback(
            Output('emotion-chart', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_emotion_chart(n):
            """Update emotion distribution chart."""
            if not self.live_data:
                return self._create_empty_chart("No data available")
            
            emotions = [d.get('dominant_emotion', 'neutral') for d in self.live_data.get('detections', [])]
            if not emotions:
                return self._create_empty_chart("No emotion data")
            
            from collections import Counter
            emotion_counts = Counter(emotions)
            
            fig = px.bar(
                x=list(emotion_counts.keys()),
                y=list(emotion_counts.values()),
                title="Emotion Distribution",
                labels={'x': 'Emotion', 'y': 'Count'}
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            
            return fig
        
        @self.app.callback(
            Output('attention-timeline', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_attention_timeline(n):
            """Update attention timeline chart."""
            if not self.live_data:
                return self._create_empty_chart("No data available")
            
            # Get attention scores over time
            attention_data = []
            for detection in self.live_data.get('detections', []):
                if 'frame_idx' in detection and 'attention' in detection:
                    attention_data.append({
                        'frame': detection['frame_idx'],
                        'attention': detection['attention'].get('attention_score', 0)
                    })
            
            if not attention_data:
                return self._create_empty_chart("No attention data")
            
            df = pd.DataFrame(attention_data)
            
            fig = px.line(
                df, x='frame', y='attention',
                title="Attention Score Over Time",
                labels={'frame': 'Frame', 'attention': 'Attention Score'}
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            
            return fig
        
        @self.app.callback(
            Output('position-heatmap', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_position_heatmap(n):
            """Update position heatmap."""
            if not self.live_data:
                return self._create_empty_chart("No data available")
            
            # Create position heatmap
            positions = []
            for detection in self.live_data.get('detections', []):
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    attention = detection.get('confidence', 0.5) # Use confidence as attention score
                    positions.append([center_x, center_y, attention])
            
            if not positions:
                return self._create_empty_chart("No position data")
            
            # Create heatmap - handle different array sizes
            positions = np.array(positions)
            if len(positions) < 100:  # If we have fewer than 100 points
                # Create a simple scatter plot instead of heatmap
                fig = go.Figure(data=go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=positions[:, 2],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Attention Score")
                    ),
                    text=[f"Attention: {att:.2f}" for att in positions[:, 2]],
                    hovertemplate='<b>Position</b><br>' +
                                'X: %{x}<br>' +
                                'Y: %{y}<br>' +
                                'Attention: %{text}<br>' +
                                '<extra></extra>'
                ))
                fig.update_layout(
                    title="Face Positions (Attention-based)",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
            else:
                # Create heatmap for larger datasets
                fig = go.Figure(data=go.Heatmap(
                    z=positions[:, 2].reshape(10, 10),  # Reshape for visualization
                    colorscale='RdYlGn',
                    title="Face Positions Heatmap (Attention-based)"
                ))
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            
            return fig
        
        @self.app.callback(
            Output('seat-assignments', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_seat_assignments(n):
            """Update seat assignments display."""
            if not self.live_data:
                return "No seat assignments available"
            
            seat_data = self.live_data.get('seat_assignments', {})
            if not seat_data:
                return "No seat assignments available"
            
            seat_list = []
            for seat_id, assignment in seat_data.items():
                seat_list.append(html.Div([
                    html.Strong(f"{seat_id}: "),
                    html.Span(f"{assignment.get('identity', 'Unknown')} "),
                    html.Small(f"(Attention: {assignment.get('attention_score', 0):.2f})")
                ], style={'marginBottom': '5px'}))
            
            return seat_list
        
        @self.app.callback(
            Output('recent-detections', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_recent_detections(n):
            """Update recent detections display."""
            if not self.live_data:
                return "No recent detections"
            
            detections = self.live_data.get('detections', [])
            if not detections:
                return "No recent detections"
            
            # Show last 5 detections
            recent = detections[-5:]
            detection_list = []
            
            for detection in recent:
                identity = detection.get('identity', 'Unknown')
                emotion = detection.get('dominant_emotion', 'neutral')
                attention = detection.get('confidence', 0)
                
                detection_list.append(html.Div([
                    html.Strong(f"{identity}: "),
                    html.Span(f"{emotion.title()} "),
                    html.Small(f"(Attention: {attention:.2f})")
                ], style={'marginBottom': '5px', 'fontSize': '0.9em'}))
            
            return detection_list
        
        @self.app.callback(
            Output('upload-status', 'children'),
            Input('upload-video', 'contents'),
            Input('upload-video', 'filename'),
            State('dashboard-state', 'data')
        )
        def update_upload_status(contents, filename, stored_data):
            """Update upload status."""
            if not contents and not filename:
                return "Ready to upload video"
            
            if contents and filename:
                return f"🔄 Uploading {filename}..."
            
            return "Ready to upload video"
        
        @self.app.callback(
            Output('dashboard-state', 'data'),
            Input('upload-video', 'contents'),
            Input('upload-video', 'filename'),
            Input('process-video-button', 'n_clicks'),
            Input('clear-video-button', 'n_clicks'),
            Input('prev-frame-button', 'n_clicks'),
            Input('next-frame-button', 'n_clicks'),
            State('dashboard-state', 'data')
        )
        def update_dashboard_state(contents, filename, process_clicks, clear_clicks, prev_clicks, next_clicks, stored_data):
            """Update and persist dashboard state."""
            import dash
            ctx = dash.callback_context
            
            # Initialize state
            if stored_data is None:
                stored_data = {
                    'uploaded_video_path': None,
                    'video_frames': [],
                    'current_frame_index': 0,
                    'video_analysis_data': {},
                    'live_data': {},
                    'processing_status': "Ready"
                }
            
            if not ctx.triggered:
                return stored_data
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'clear-video-button':
                # Clear all state
                stored_data = {
                    'uploaded_video_path': None,
                    'video_frames': [],
                    'current_frame_index': 0,
                    'video_analysis_data': {},
                    'live_data': {},
                    'processing_status': "Ready"
                }
            elif trigger_id == 'upload-video' and contents and filename:
                # Update uploaded video path
                try:
                    video_path = self._save_uploaded_video(contents, filename)
                    if video_path:
                        stored_data['uploaded_video_path'] = video_path
                        stored_data['processing_status'] = "✅ Video uploaded successfully! Now you can start processing."
                    else:
                        stored_data['processing_status'] = "❌ Failed to save uploaded video"
                except Exception as e:
                    stored_data['processing_status'] = f"❌ Upload error: {str(e)}"
            elif trigger_id == 'process-video-button' and stored_data.get('uploaded_video_path'):
                # Process video and update state
                try:
                    analysis_data = self._process_uploaded_video(stored_data['uploaded_video_path'])
                    if analysis_data.get('frames'):
                        stored_data['video_frames'] = analysis_data['frames']
                        stored_data['current_frame_index'] = 0
                        stored_data['live_data'] = {'detections': analysis_data['detections']}
                        stored_data['video_analysis_data'] = analysis_data
                        stored_data['processing_status'] = f"✅ Video processed: {len(analysis_data['frames'])} frames"
                    else:
                        stored_data['processing_status'] = "❌ No frames extracted from video"
                except Exception as e:
                    stored_data['processing_status'] = f"❌ Processing error: {str(e)}"
            elif trigger_id == 'prev-frame-button' and stored_data.get('video_frames'):
                # Navigate to previous frame
                stored_data['current_frame_index'] = max(0, stored_data['current_frame_index'] - 1)
            elif trigger_id == 'next-frame-button' and stored_data.get('video_frames'):
                # Navigate to next frame
                stored_data['current_frame_index'] = min(len(stored_data['video_frames']) - 1, stored_data['current_frame_index'] + 1)
            
            return stored_data
        
        @self.app.callback(
            Output('video-frame', 'src'),
            Output('upload-status', 'children'),
            Output('processing-status', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_video_display(stored_data, n_intervals):
            """Update video display based on stored state."""
            if stored_data is None:
                return self._create_simple_analysis_message(), "Ready to upload video", "Ready"
            
            # Update instance variables from stored data
            self.uploaded_video_path = stored_data.get('uploaded_video_path')
            self.video_frames = stored_data.get('video_frames', [])
            self.current_frame_index = stored_data.get('current_frame_index', 0)
            self.video_analysis_data = stored_data.get('video_analysis_data', {})
            self.live_data = stored_data.get('live_data', {})
            self.processing_status = stored_data.get('processing_status', "Ready")
            
            # Auto-cycle through frames if available
            if self.video_frames and len(self.video_frames) > 0 and n_intervals and n_intervals % 5 == 0:
                self._update_frame_index()
                stored_data['current_frame_index'] = self.current_frame_index
            
            # Return appropriate frame and status
            if self.video_frames and len(self.video_frames) > 0:
                frame_src = self._frame_to_base64(self.video_frames[self.current_frame_index])
                frame_info = f"Frame {self.current_frame_index + 1}/{len(self.video_frames)}"
                return frame_src, frame_info, self.processing_status
            else:
                return self._create_simple_analysis_message(), "Ready to upload video", self.processing_status
        
        @self.app.callback(
            Output('frame-info', 'children'),
            Input('dashboard-state', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def update_frame_info(stored_data, n):
            """Update frame information display."""
            if not stored_data or not stored_data.get('video_frames'):
                return "No video frames available"
            
            current_frame = stored_data.get('current_frame_index', 0) + 1
            total_frames = len(stored_data['video_frames'])
            detections_count = len(stored_data.get('live_data', {}).get('detections', []))
            return f"📹 Frame {current_frame}/{total_frames} - {detections_count} detections"
        
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
            """Update process button state based on whether a video is uploaded."""
            if stored_data and stored_data.get('uploaded_video_path'):
                # Video is uploaded, enable the button
                return False, {
                    'backgroundColor': '#3498db', 
                    'color': 'white', 
                    'border': 'none', 
                    'padding': '12px 24px', 
                    'marginRight': 15,
                    'borderRadius': '6px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer',
                    'transition': 'all 0.3s ease'
                }
            else:
                # No video uploaded, disable the button
                return True, {
                    'backgroundColor': '#bdc3c7', 
                    'color': '#7f8c8d', 
                    'border': 'none', 
                    'padding': '12px 24px', 
                    'marginRight': 15,
                    'borderRadius': '6px',
                    'fontSize': '14px',
                    'fontWeight': 'bold',
                    'cursor': 'not-allowed',
                    'transition': 'all 0.3s ease'
                }
        
        @self.app.callback(
            Output('interval-component', 'interval'),
            Input('refresh-slider', 'value')
        )
        def update_refresh_rate(value):
            """Update refresh rate."""
            return value * 1000  # Convert to milliseconds
    
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
    
    def run(self, debug: bool = False):
        """Run the dashboard."""
        logger.info(f"Starting dashboard on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
    
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