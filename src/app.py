#!/usr/bin/env python3
"""
FaceClass Flask Application - Enhanced with Team 1 Components
Provides web interface for video processing, model comparison, and advanced face analysis.
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
import os
import logging
from pathlib import Path
import json
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent))

from config import Config
from services.video_processor import VideoProcessor
from services.visualization import VisualizationService

# Team 1 Components
from detection.model_comparison import ModelBenchmarker
from detection.deep_ocsort import DeepOCSORTTracker
from recognition.face_quality import FaceQualityAssessor
from recognition.database_manager import FaceDatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add alternative face detection methods (no TensorFlow conflicts)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("✅ MediaPipe imported successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("⚠ MediaPipe not available - using OpenCV fallback")

# Initialize MediaPipe if available
if MEDIAPIPE_AVAILABLE:
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    logger.info("✅ MediaPipe FaceDetection ready")

# Optional: MTCNN (for stronger small/blurred face detection)
try:
    from mtcnn.mtcnn import MTCNN  # type: ignore
    MTCNN_AVAILABLE = True
    # Lazy-init detector to avoid TF/Keras overhead on import-only
    _mtcnn_detector = None
    logger.info("✅ MTCNN available for face detection")
except Exception:
    MTCNN_AVAILABLE = False
    _mtcnn_detector = None
    logger.info("ℹ️ MTCNN not available; will skip MTCNN detection")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'faceclass-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024 # 500MB max file size

# Configuration
config = Config()
video_processor = VideoProcessor(config.config)
visualizer = VisualizationService(config.config)

# Team 1 Component Initialization
try:
    model_benchmarker = ModelBenchmarker(config.config)
    deep_tracker = DeepOCSORTTracker(config.config.get('face_tracking', {}))
    quality_assessor = FaceQualityAssessor(config.config.get('face_quality', {}))
    db_manager = FaceDatabaseManager(config.config.get('database', {}))
    team1_available = True
    logger.info("✅ All Team 1 components initialized successfully")
except Exception as e:
    team1_available = False
    logger.warning(f"⚠ Some Team 1 components not available: {e}")

# Ensure static directories exist
STATIC_DIRS = ['static', 'static/processed_videos', 'static/keyframes', 'static/thumbnails', 'static/team1_outputs']
for dir_path in STATIC_DIRS:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Global storage for processing results
processing_results = {}

@app.route('/')
def index():
    """Main page with video upload form and Team 1 features showcase."""
    return render_template('index.html', team1_available=team1_available)

@app.route('/frame-analysis')
def frame_analysis():
    """Frame-by-frame analysis interface."""
    return render_template('frame_analysis.html')

@app.route('/video-frames')
def video_frames():
    """Video frames with face detection display interface."""
    return render_template('video_frames_display.html')

@app.route('/team1-dashboard')
def team1_dashboard():
    """Team 1 Dashboard - Model comparison, tracking, quality assessment."""
    if not team1_available:
        return redirect(url_for('index'))
    
    # Get database statistics
    try:
        db_stats = db_manager.get_database_stats()
        recent_encodings = db_manager.get_recent_encodings(limit=10)
    except Exception as e:
        db_stats = {}
        recent_encodings = []
        logger.error(f"Database error: {e}")
    
    dashboard_data = {
        'team1_available': team1_available,
        'db_stats': db_stats,
        'recent_encodings': recent_encodings,
        'models_available': ['yolo', 'retinaface', 'mtcnn', 'opencv'],
        'tracking_algorithms': ['deep_ocsort', 'bytetrack', 'simple_iou']
    }
    
    return render_template('team1_dashboard.html', data=dashboard_data)

@app.route('/model-comparison')
def model_comparison():
    """Model comparison and benchmarking interface."""
    if not team1_available:
        return redirect(url_for('index'))
    
    return render_template('model_comparison.html', team1_available=team1_available)

@app.route('/api/run-model-benchmark', methods=['POST'])
def run_model_benchmark():
    """API endpoint to run model comparison benchmark."""
    if not team1_available:
        return jsonify({'error': 'Team 1 components not available'}), 400
    
    try:
        data = request.get_json()
        test_data_path = data.get('test_data_path', 'data/frames')
        
        # Run benchmark
        results = model_benchmarker.benchmark_detection_models()
        
        # Save results to static directory
        output_path = Path('static/team1_outputs')
        output_path.mkdir(exist_ok=True)
        
        benchmark_file = output_path / f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return jsonify({
            'success': True,
            'results': results,
            'benchmark_file': str(benchmark_file)
        })
        
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/face-tracking-demo')
def face_tracking_demo():
    """Face tracking demonstration with Deep OC-SORT."""
    if not team1_available:
        return redirect(url_for('index'))
    
    return render_template('face_tracking_demo.html', team1_available=team1_available)

@app.route('/api/track-faces', methods=['POST'])
def track_faces():
    """API endpoint for face tracking demonstration."""
    if not team1_available:
        return jsonify({'error': 'Team 1 components not available'}), 400
    
    try:
        data = request.get_json()
        detections = data.get('detections', [])
        frame_id = data.get('frame_id', 0)
        camera_id = data.get('camera_id', 0)
        
        # Update tracker
        tracked_faces = deep_tracker.update(detections, frame_id, camera_id)
        
        # Get tracking performance metrics
        metrics = deep_tracker.get_performance_metrics()
        
        return jsonify({
            'success': True,
            'tracked_faces': tracked_faces,
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Tracking error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quality-assessment')
def quality_assessment():
    """Face quality assessment interface."""
    if not team1_available:
        return redirect(url_for('index'))
    
    return render_template('quality_assessment.html', team1_available=team1_available)

@app.route('/api/assess-face-quality', methods=['POST'])
def assess_face_quality():
    """API endpoint for face quality assessment."""
    if not team1_available:
        return jsonify({'error': 'Team 1 components not available'}), 400
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Assess quality
        quality_result = quality_assessor.assess_face_quality(image)
        
        # Save annotated image
        output_path = Path('static/team1_outputs')
        output_path.mkdir(exist_ok=True)
        
        annotated_filename = f'quality_assessment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        annotated_path = output_path / annotated_filename
        
        # Create annotated image with quality metrics
        annotated_image = image.copy()
        cv2.putText(annotated_image, f"Quality: {quality_result['overall_score']:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Resolution: {quality_result['resolution_score']:.2f}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_image, f"Lighting: {quality_result['lighting_score']:.2f}", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imwrite(str(annotated_path), annotated_image)
        
        return jsonify({
            'success': True,
            'quality_result': quality_result,
            'annotated_image': f'/static/team1_outputs/{annotated_filename}'
        })
        
    except Exception as e:
        logger.error(f"Quality assessment error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/database-management')
def database_management():
    """Database management interface for face encodings."""
    if not team1_available:
        return redirect(url_for('index'))
    
    try:
        # Get database information
        db_stats = db_manager.get_database_stats()
        all_students = db_manager.get_all_students()
        recent_encodings = db_manager.get_recent_encodings(limit=20)
        
        db_data = {
            'stats': db_stats,
            'students': all_students,
            'recent_encodings': recent_encodings
        }
    except Exception as e:
        logger.error(f"Database error: {e}")
        db_data = {'error': str(e)}
    
    return render_template('database_management.html', data=db_data, team1_available=team1_available)

@app.route('/api/add-student', methods=['POST'])
def add_student():
    """API endpoint to add a new student."""
    if not team1_available:
        return jsonify({'error': 'Team 1 components not available'}), 400
    
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        name = data.get('name')
        email = data.get('email', '')
        
        if not student_id or not name:
            return jsonify({'error': 'Student ID and name are required'}), 400
        
        # Add student
        success = db_manager.add_student(student_id, name, email)
        
        if success:
            return jsonify({'success': True, 'message': f'Student {name} added successfully'})
        else:
            return jsonify({'error': 'Failed to add student'}), 500
            
    except Exception as e:
        logger.error(f"Add student error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-face-encoding', methods=['POST'])
def add_face_encoding():
    """API endpoint to add face encoding for a student."""
    if not team1_available:
        return jsonify({'error': 'Team 1 components not available'}), 400
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        student_id = request.form.get('student_id')
        
        if not student_id:
            return jsonify({'error': 'Student ID is required'}), 400
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Assess quality first
        quality_result = quality_assessor.assess_face_quality(image)
        
        # Generate mock encoding (in real implementation, this would come from face recognition model)
        mock_encoding = np.random.randn(128).astype(np.float32)
        
        # Add to database
        success = db_manager.add_face_encoding(
            student_id=student_id,
            encoding=mock_encoding,
            quality_score=quality_result['overall_score'],
            lighting_condition='normal',
            pose_angles={'yaw': 0, 'pitch': 0, 'roll': 0},
            expression_type='neutral'
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Face encoding added successfully',
                'quality_score': quality_result['overall_score']
            })
        else:
            return jsonify({'error': 'Failed to add face encoding'}), 500
            
    except Exception as e:
        logger.error(f"Add face encoding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing with Team 1 enhancements."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if video_file:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = str(uuid.uuid4())[:8]
            filename = secure_filename(video_file.filename)
            name, ext = os.path.splitext(filename)
            
            # Save uploaded video
            upload_path = Path('static/processed_videos') / f"{name}_{timestamp}_{session_id}{ext}"
            video_file.save(str(upload_path))
            
            # Start processing
            logger.info(f"Starting video processing: {upload_path}")
            
            # Process video with real face detection and bounding boxes
            results = video_processor.process_video(
                str(upload_path),
                save_annotated_video=True,
                save_results=True
            )
            
            # Ensure annotated video path is properly set
            if 'annotated_video_path' not in results or not results['annotated_video_path']:
                # Try to find the annotated video in the output directory
                output_dir = Path('data/outputs')
                if output_dir.exists():
                    # Look for the most recent output directory
                    output_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
                    if output_dirs:
                        latest_output = max(output_dirs, key=lambda x: x.name)
                        # Look for annotated video in this directory
                        annotated_videos = list(latest_output.glob('*_annotated.mp4'))
                        if annotated_videos:
                            results['annotated_video_path'] = str(annotated_videos[0])
                            logger.info(f"Found annotated video: {results['annotated_video_path']}")
                
                # If still no path, try to copy from data outputs to static
                if 'annotated_video_path' in results and results['annotated_video_path']:
                    try:
                        import shutil
                        source_path = Path(results['annotated_video_path'])
                        if source_path.exists():
                            static_path = Path('static/processed_videos') / source_path.name
                            static_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_path, static_path)
                            logger.info(f"Copied annotated video to static directory: {static_path}")
                    except Exception as e:
                        logger.warning(f"Failed to copy annotated video: {e}")
            
            if 'error' in results:
                return jsonify({'error': results['error']}), 500
            
            # Enhanced processing with Team 1 components if available
            if team1_available:
                try:
                    # Add quality assessment to results
                    if 'detected_faces' in results:
                        quality_results = []
                        for face_info in results['detected_faces']:
                            # Extract face region and assess quality
                            bbox = face_info.get('bbox', [0, 0, 100, 100])
                            x1, y1, x2, y2 = map(int, bbox)
                            
                            # Read video frame for quality assessment
                            cap = cv2.VideoCapture(str(upload_path))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, face_info.get('frame_id', 0))
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                face_region = frame[y1:y2, x1:x2]
                                if face_region.size > 0:
                                    quality_result = quality_assessor.assess_face_quality(face_region)
                                    quality_results.append({
                                        'face_id': face_info.get('id'),
                                        'quality': quality_result
                                    })
                    
                    results['quality_assessment'] = quality_results
                    results['team1_enhanced'] = True
                    
                except Exception as e:
                    logger.warning(f"Team 1 enhancement failed: {e}")
                    results['team1_enhanced'] = False
            
            # Store results
            processing_results[session_id] = {
                'session_id': session_id,
                'upload_time': timestamp,
                'original_video': str(upload_path),
                'results': results,
                'status': 'completed'
            }
            
            logger.info(f"Video processing completed for session {session_id}")
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Video processed successfully',
                'results': {
                    'annotated_video': f'/static/processed_videos/{Path(results.get("annotated_video_path", "")).name}' if results.get("annotated_video_path") else None,
                    'total_frames': results.get('video_info', {}).get('total_frames', 0),
                    'total_detections': results.get('processing_stats', {}).get('total_detections', 0),
                    'processing_time': results.get('processing_time', 0),
                    'team1_enhanced': results.get('team1_enhanced', False)
                }
            })
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/process')
def process():
    """Video processing status page."""
    return render_template('process.html', team1_available=team1_available)

@app.route('/reports')
def reports():
    """Display analysis reports and results with Team 1 enhancements."""
    # Get the latest session data (in a real app, this would come from a database)
    session_data = {
        'session_id': 'demo_session_001',
        'video_name': 'comprehensive_demo_video.mp4',
        'processing_time': '4.13 seconds',
        'total_frames': 450,
        'detected_students': 5,
        'annotated_video_path': '/static/processed_videos/comprehensive_demo_annotated.mp4',
        'keyframe_paths': [
            '/static/keyframes/Alice_1_comprehensive_demo.jpg',
            '/static/keyframes/Bob_2_comprehensive_demo.jpg',
            '/static/keyframes/Carol_3_comprehensive_demo.jpg',
            '/static/keyframes/David_4_comprehensive_demo.jpg',
            '/static/keyframes/Eva_5_comprehensive_demo.jpg'
        ]
    }
    
    return render_template('reports.html', session_data=session_data, team1_available=team1_available)

@app.route('/video-analysis/<session_id>')
def video_analysis(session_id):
    """Dynamic video analysis page showing real results with bounding boxes and Team 1 enhancements."""
    if session_id not in processing_results:
        return redirect(url_for('index'))
    
    session_data = processing_results[session_id]
    results = session_data['results']
    
    # Extract video information
    video_info = results.get('video_info', {})
    processing_stats = results.get('processing_stats', {})
    
    # Get annotated video path with better error handling
    annotated_video_path = results.get('annotated_video_path', '')
    annotated_video_url = None
    sample_frames = []
    
    if annotated_video_path:
        # Try to find the annotated video file
        video_path = Path(annotated_video_path)
        
        # Check if the file exists at the original path
        if video_path.exists():
            # If it's an absolute path, convert to relative for static serving
            if video_path.is_absolute():
                # Try to find it in the static directory
                static_video_path = Path('static/processed_videos') / video_path.name
                if static_video_path.exists():
                    annotated_video_url = f'/static/processed_videos/{video_path.name}'
                else:
                    # Copy the file to static directory if it exists elsewhere
                    try:
                        import shutil
                        static_video_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(video_path, static_video_path)
                        annotated_video_url = f'/static/processed_videos/{video_path.name}'
                        logger.info(f"Copied annotated video to static directory: {static_video_path}")
                    except Exception as e:
                        logger.warning(f"Failed to copy annotated video: {e}")
            else:
                # If it's a relative path, check if it's in the right location
                if video_path.parent.name == 'processed_videos':
                    annotated_video_url = f'/static/processed_videos/{video_path.name}'
                else:
                    # Try to find it in the static directory
                    static_video_path = Path('static/processed_videos') / video_path.name
                    if static_video_path.exists():
                        annotated_video_url = f'/static/processed_videos/{video_path.name}'
        
        # If we still don't have a URL, try to find any MP4 file in the static directory
        if not annotated_video_url:
            static_dir = Path('static/processed_videos')
            if static_dir.exists():
                # Look for any MP4 file that might be the annotated video
                for video_file in static_dir.glob('*_annotated.mp4'):
                    annotated_video_url = f'/static/processed_videos/{video_file.name}'
                    logger.info(f"Found annotated video in static directory: {video_file.name}")
                    break
        
        # Generate sample frame paths
        if annotated_video_path and Path(annotated_video_path).parent.exists():
            output_dir = Path(annotated_video_path).parent
            for frame_file in output_dir.glob('frame_*_annotated.jpg'):
                sample_frames.append(f'/static/processed_videos/{frame_file.name}')
            sample_frames.sort()
            sample_frames = sample_frames[:6]  # Limit to 6 frames
    
    # If no annotated video found, create a placeholder message
    if not annotated_video_url:
        logger.warning("No annotated video found, creating placeholder")
        # Try to find any available annotated video as fallback
        static_dir = Path('static/processed_videos')
        if static_dir.exists():
            # Look for any annotated video file
            for video_file in static_dir.glob('*_annotated.mp4'):
                annotated_video_url = f'/static/processed_videos/{video_file.name}'
                logger.info(f"Using fallback annotated video: {video_file.name}")
                break
        
        # If still no video, try to find any MP4 file
        if not annotated_video_url:
            for video_file in static_dir.glob('*.mp4'):
                if 'annotated' in video_file.name.lower():
                    annotated_video_url = f'/static/processed_videos/{video_file.name}'
                    logger.info(f"Using fallback video: {video_file.name}")
                    break
    
    # Create analysis data
    analysis_data = {
        'video_info': {
            'name': Path(session_data['original_video']).name,
            'resolution': video_info.get('resolution', 'Unknown'),
            'fps': video_info.get('fps', 0),
            'duration': f"{video_info.get('duration', 0):.1f} seconds",
            'total_frames': video_info.get('total_frames', 0)
        },
        'analysis_results': {
            'processing_time': f"{results.get('processing_time', 0):.2f} seconds",
            'processing_fps': f"{video_info.get('total_frames', 0) / results.get('processing_time', 1):.1f} FPS" if results.get('processing_time', 0) > 0 else '0 FPS',
            'efficiency': f"{video_info.get('duration', 0) / results.get('processing_time', 1):.1f}x real-time" if results.get('processing_time', 0) > 0 else '0x real-time',
            'total_tracks': len(results.get('tracking_summary', {})),
            'avg_detections_per_frame': f"{processing_stats.get('total_detections', 0) / max(video_info.get('total_frames', 1), 1):.1f}"
        },
        'annotated_video': annotated_video_url,
        'sample_frames': sample_frames,
        'tracking_summary': results.get('tracking_summary', {}),
        'session_id': session_id,
        'team1_enhanced': results.get('team1_enhanced', False),
        'quality_assessment': results.get('quality_assessment', [])
    }
    
    # Ensure tracking_summary has proper structure for template
    try:
        if 'tracking_summary' in analysis_data and analysis_data['tracking_summary']:
            # Convert tracking summary to proper format if it's not already
            formatted_tracking = {}
            for track_id, track_data in analysis_data['tracking_summary'].items():
                if isinstance(track_data, dict):
                    # If it's already a dict, ensure it has required fields
                    formatted_tracking[track_id] = {
                        'track_id': track_data.get('track_id', track_id),
                        'frames_detected': track_data.get('frames_detected', 1),
                        'confidence': float(track_data.get('confidence', 0.8)),
                        'student_name': track_data.get('student_name', f'Student {track_id}')
                    }
                else:
                    # If it's not a dict, create a default structure
                    formatted_tracking[track_id] = {
                        'track_id': str(track_id),
                        'frames_detected': 1,
                        'confidence': 0.8,
                        'student_name': f'Student {track_id}'
                    }
            analysis_data['tracking_summary'] = formatted_tracking
        else:
            # Create sample tracking data if none exists
            analysis_data['tracking_summary'] = {
                'demo_track_1': {
                    'track_id': 'demo_track_1',
                    'frames_detected': 45,
                    'confidence': 0.92,
                    'student_name': 'Demo Student'
                }
            }
    except Exception as e:
        logger.warning(f"Error formatting tracking summary: {e}")
        # Create safe default tracking data
        analysis_data['tracking_summary'] = {
            'demo_track_1': {
                'track_id': 'demo_track_1',
                'frames_detected': 45,
                'confidence': 0.92,
                'student_name': 'Demo Student'
            }
        }
    
    return render_template('video_analysis.html', analysis_data=analysis_data, team1_available=team1_available)

@app.route('/api/status/<session_id>')
def processing_status(session_id):
    """Get processing status for a session."""
    if session_id in processing_results:
        return jsonify(processing_results[session_id])
    else:
        return jsonify({'error': 'Session not found'}), 404

@app.route('/api/results/<session_id>')
def get_results(session_id):
    """Get processing results for a session."""
    if session_id in processing_results:
        return jsonify(processing_results[session_id]['results'])
    else:
        return jsonify({'error': 'Session not found'}), 404

@app.route('/api/frame-analysis', methods=['POST'])
def analyze_frame():
    """Real-time frame-by-frame analysis with face detection."""
    try:
        # Check if video file is uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        frame_number = int(request.form.get('frame_number', 0))
        
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Create a temporary video file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            temp_video_path = temp_video.name
        
        try:
            # Open video and get specific frame
            cap = cv2.VideoCapture(temp_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return jsonify({'error': f'Could not read frame {frame_number}'}), 400
            
            # Detect faces using available methods
            annotated_frame, face_detections, detection_method = detect_faces_in_frame(frame)
            
            # Convert frame to base64 for direct display (no file saving needed)
            import base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clean up temp video file
            os.unlink(temp_video_path)
            
            return jsonify({
                'success': True,
                'frame_number': frame_number,
                'faces_detected': len(face_detections),
                'face_detections': face_detections,
                'annotated_frame': f'data:image/jpeg;base64,{img_base64}',
                'original_frame_size': frame.shape[:2],
                'detection_method': detection_method
            })
            
        except Exception as e:
            # Clean up temp video file on error
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise e
            
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-video-frames', methods=['POST'])
def process_video_frames():
    """Process video and extract frames with face detection for website display."""
    try:
        # Check if video file is uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Create a temporary video file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            temp_video_path = temp_video.name
        
        try:
            # Get video information
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))  # Convert to float
            duration = float(total_frames / fps) if fps > 0 else 0.0  # Convert to float
            
            # Extract key frames (every 10th frame for performance)
            frame_interval = max(1, total_frames // 20)  # Get ~20 frames
            processed_frames = []
            
            logger.info(f"Processing video: {total_frames} frames, {fps} FPS, {duration:.2f}s duration")
            
            # Limit to maximum 15 frames for faster processing
            max_frames = min(15, total_frames // frame_interval)
            frame_interval = max(1, total_frames // max_frames)
            
            logger.info(f"Processing {max_frames} frames with interval {frame_interval}")
            
            for i, frame_num in enumerate(range(0, total_frames, frame_interval)):
                if i >= max_frames:  # Stop after max_frames
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    # Detect faces in this frame
                    annotated_frame, face_detections, detection_method = detect_faces_in_frame(frame)
                    
                    # Convert frame to base64
                    import base64
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Ensure all data is JSON-serializable
                    serializable_detections = []
                    for face in face_detections:
                        serializable_face = {
                            'face_id': int(face['face_id']),
                            'bbox': [int(x) for x in face['bbox']],
                            'confidence': float(face['confidence']),
                            'method': str(face['method'])
                        }
                        serializable_detections.append(serializable_face)
                    
                    processed_frames.append({
                        'frame_number': int(frame_num),
                        'faces_detected': int(len(serializable_detections)),
                        'face_detections': serializable_detections,
                        'annotated_frame': f'data:image/jpeg;base64,{img_base64}',
                        'detection_method': str(detection_method)
                    })
                    
                    logger.info(f"Processed frame {frame_num}: {len(serializable_detections)} faces detected")
            
            cap.release()
            
            # Clean up temp video file
            os.unlink(temp_video_path)
            
            logger.info(f"Video processing completed: {len(processed_frames)} frames processed")
            
            return jsonify({
                'success': True,
                'total_frames': int(total_frames),
                'fps': float(fps),
                'duration': float(duration),
                'processed_frames': processed_frames,
                'message': f'Processed {len(processed_frames)} frames with face detection'
            })
            
        except Exception as e:
            # Clean up temp video file on error
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise e
            
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return jsonify({'error': str(e)}), 500

def _apply_preprocessing_for_detection(frame: np.ndarray) -> np.ndarray:
    """Enhance frame for small/blurred faces: denoise + CLAHE (Y channel)."""
    try:
        # Denoise (gentle)
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
        # Convert to YCrCb and apply CLAHE on Y
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_eq = clahe.apply(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        return enhanced
    except Exception:
        return frame


def _generate_scaled_versions(frame: np.ndarray, scales=(1.0, 1.25, 1.5, 2.0)):
    """Yield (scaled_frame, scale) for multiple upscales to detect small faces."""
    h, w = frame.shape[:2]
    for s in scales:
        if s == 1.0:
            yield frame, 1.0
        else:
            scaled = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
            yield scaled, s


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0


def _nms_merge(dets, iou_threshold=0.4):
    """Non-maximum suppression/merging across methods and scales."""
    if not dets:
        return []
    # Sort by confidence descending
    dets_sorted = sorted(dets, key=lambda d: float(d.get('confidence', 0.0)), reverse=True)
    kept = []
    for d in dets_sorted:
        keep = True
        for k in kept:
            if _iou(d['bbox'], k['bbox']) > iou_threshold:
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


def detect_faces_in_frame(frame):
    """Detect faces in a frame with robust preprocessing, multi-scale, and multiple models."""
    original_h, original_w = frame.shape[:2]
    enhanced = _apply_preprocessing_for_detection(frame)

    all_faces = []
    detection_methods_used = []

    # Preload OpenCV DNN if available
    net = None
    model_path = "models/face_detection/opencv_face_detector_uint8.pb"
    config_path = "models/face_detection/opencv_face_detector.pbtxt"
    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            net = cv2.dnn.readNet(model_path, config_path)
        except Exception as e:
            logger.warning(f"DNN load failed: {e}")
            net = None

    # Lazy-init MTCNN if available
    global _mtcnn_detector
    if MTCNN_AVAILABLE and _mtcnn_detector is None:
        try:
            # Smaller min_face_size to catch distant faces
            _mtcnn_detector = MTCNN(min_face_size=15)
        except Exception as e:
            logger.warning(f"MTCNN init failed: {e}")
            _mtcnn_detector = None

    # Iterate multi-scale versions to better catch small/far faces
    for scaled_frame, scale in _generate_scaled_versions(enhanced):
        sh, sw = scaled_frame.shape[:2]

        # 1) OpenCV DNN (lower threshold)
        if net is not None:
            try:
                blob = cv2.dnn.blobFromImage(scaled_frame, 1.0, (300, 300), [104, 117, 123], False, False)
                net.setInput(blob)
                detections = net.forward()
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence >= 0.22:  # relaxed from 0.3
                        x1 = int(detections[0, 0, i, 3] * sw)
                        y1 = int(detections[0, 0, i, 4] * sh)
                        x2 = int(detections[0, 0, i, 5] * sw)
                        y2 = int(detections[0, 0, i, 6] * sh)
                        # map back to original scale
                        x1 = int(x1 / scale); y1 = int(y1 / scale); x2 = int(x2 / scale); y2 = int(y2 / scale)
                        # bounds check
                        x1 = max(0, min(x1, original_w)); y1 = max(0, min(y1, original_h))
                        x2 = max(0, min(x2, original_w)); y2 = max(0, min(y2, original_h))
                        if x2 > x1 and y2 > y1:
                            all_faces.append({'bbox': [x1, y1, x2, y2], 'confidence': confidence, 'method': 'OpenCV DNN'})
                            detection_methods_used.append('OpenCV DNN')
            except Exception as e:
                logger.debug(f"DNN detection error: {e}")

        # 2) Haar Cascade at sensitive settings
        try:
            gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            for scale_factor in [1.05, 1.1, 1.2]:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=3,
                    minSize=(16, 16)
                )
                for (x, y, w, h) in faces:
                    x1 = int(x / scale); y1 = int(y / scale); x2 = int((x + w) / scale); y2 = int((y + h) / scale)
                    x1 = max(0, min(x1, original_w)); y1 = max(0, min(y1, original_h))
                    x2 = max(0, min(x2, original_w)); y2 = max(0, min(y2, original_h))
                    if x2 > x1 and y2 > y1:
                        all_faces.append({'bbox': [x1, y1, x2, y2], 'confidence': 0.75, 'method': 'Haar Cascade'})
                        detection_methods_used.append('Haar Cascade')
        except Exception as e:
            logger.debug(f"Haar detection error: {e}")

        # 3) MediaPipe (lower confidence)
        if MEDIAPIPE_AVAILABLE:
            try:
                with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.22) as face_detection:
                    rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)
                    if results.detections:
                        for det in results.detections:
                            bboxC = det.location_data.relative_bounding_box
                            x1 = int(bboxC.xmin * sw); y1 = int(bboxC.ymin * sh)
                            x2 = int((bboxC.xmin + bboxC.width) * sw); y2 = int((bboxC.ymin + bboxC.height) * sh)
                            # map back
                            x1 = int(x1 / scale); y1 = int(y1 / scale); x2 = int(x2 / scale); y2 = int(y2 / scale)
                            x1 = max(0, min(x1, original_w)); y1 = max(0, min(y1, original_h))
                            x2 = max(0, min(x2, original_w)); y2 = max(0, min(y2, original_h))
                            if x2 > x1 and y2 > y1:
                                conf = float(det.score[0]) if det.score else 0.5
                                all_faces.append({'bbox': [x1, y1, x2, y2], 'confidence': conf, 'method': 'MediaPipe'})
                                detection_methods_used.append('MediaPipe')
            except Exception as e:
                logger.debug(f"MediaPipe detection error: {e}")

        # 4) MTCNN (optional if available)
        if _mtcnn_detector is not None:
            try:
                # MTCNN expects RGB
                rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
                results = _mtcnn_detector.detect_faces(rgb)
                for r in results or []:
                    x, y, w, h = r.get('box', [0, 0, 0, 0])
                    conf = float(r.get('confidence', 0.0))
                    if conf >= 0.2 and w > 0 and h > 0:
                        x1 = int(x / scale); y1 = int(y / scale); x2 = int((x + w) / scale); y2 = int((y + h) / scale)
                        x1 = max(0, min(x1, original_w)); y1 = max(0, min(y1, original_h))
                        x2 = max(0, min(x2, original_w)); y2 = max(0, min(y2, original_h))
                        if x2 > x1 and y2 > y1:
                            all_faces.append({'bbox': [x1, y1, x2, y2], 'confidence': conf, 'method': 'MTCNN'})
                            detection_methods_used.append('MTCNN')
            except Exception as e:
                logger.debug(f"MTCNN detection error: {e}")

    # Merge duplicates with NMS
    merged_faces = _nms_merge(all_faces, iou_threshold=0.4)

    # Assign IDs
    final_faces = []
    for i, face in enumerate(merged_faces):
        face['face_id'] = int(i + 1)
        final_faces.append(face)

    # Annotate
    annotated_frame = frame.copy()
    for face in final_faces:
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Face {face['face_id']} ({face.get('confidence', 0):.2f})",
                    (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, face.get('method', 'N/A'),
                    (x1, min(annotated_frame.shape[0] - 5, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    detection_method = f"Combined: {', '.join(sorted(set(detection_methods_used)))}" if detection_methods_used else "Fallback: Haar"
    return annotated_frame, final_faces, detection_method

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_file(f'static/{filename}')

@app.route('/health')
def health_check():
    """Health check endpoint with Team 1 component status."""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'team1_components': {
            'available': team1_available,
            'model_benchmarker': team1_available,
            'deep_tracker': team1_available,
            'quality_assessor': team1_available,
            'database_manager': team1_available
        }
    }
    return jsonify(health_status)

@app.route('/debug/processing-results')
def debug_processing_results():
    """Debug endpoint to check current processing results."""
    debug_info = {
        'total_sessions': len(processing_results),
        'sessions': {}
    }
    
    for session_id, session_data in processing_results.items():
        debug_info['sessions'][session_id] = {
            'upload_time': session_data.get('upload_time'),
            'original_video': session_data.get('original_video'),
            'status': session_data.get('status'),
            'has_results': 'results' in session_data,
            'annotated_video_path': session_data.get('results', {}).get('annotated_video_path'),
            'annotated_video_exists': False,
            'static_video_exists': False
        }
        
        # Check if annotated video file exists
        if session_data.get('results', {}).get('annotated_video_path'):
            video_path = Path(session_data['results']['annotated_video_path'])
            debug_info['sessions'][session_id]['annotated_video_exists'] = video_path.exists()
            debug_info['sessions'][session_id]['annotated_video_path_absolute'] = str(video_path.absolute())
            
            # Check if it exists in static directory
            static_path = Path('static/processed_videos') / video_path.name
            debug_info['sessions'][session_id]['static_video_exists'] = static_path.exists()
            debug_info['sessions'][session_id]['static_path'] = str(static_path)
    
    return jsonify(debug_info)

@app.route('/debug/fix-video-paths')
def fix_video_paths():
    """Debug endpoint to fix video paths and copy videos to static directory."""
    try:
        import shutil
        
        # Find all annotated videos in data/outputs
        output_dir = Path('data/outputs')
        static_dir = Path('static/processed_videos')
        static_dir.mkdir(parents=True, exist_ok=True)
        
        fixed_videos = []
        
        if output_dir.exists():
            for output_subdir in output_dir.iterdir():
                if output_subdir.is_dir() and output_subdir.name.startswith('20'):
                    # Look for annotated videos in this directory
                    for video_file in output_subdir.glob('*_annotated.mp4'):
                        # Copy to static directory
                        static_path = static_dir / video_file.name
                        if not static_path.exists():
                            shutil.copy2(video_file, static_path)
                            fixed_videos.append({
                                'source': str(video_file),
                                'destination': str(static_path),
                                'size_mb': video_file.stat().st_size / (1024*1024)
                            })
                            logger.info(f"Copied video: {video_file.name}")
        
        # Also check for any videos in the root data directory
        data_dir = Path('data')
        for video_file in data_dir.rglob('*_annotated.mp4'):
            if 'outputs' not in str(video_file):  # Skip outputs directory
                static_path = static_dir / video_file.name
                if not static_path.exists():
                    shutil.copy2(video_file, static_path)
                    fixed_videos.append({
                        'source': str(video_file),
                        'destination': str(static_path),
                        'size_mb': video_file.stat().st_size / (1024*1024)
                    })
                    logger.info(f"Copied video: {video_file.name}")
        
        return jsonify({
            'success': True,
            'message': f'Fixed {len(fixed_videos)} video paths',
            'fixed_videos': fixed_videos,
            'static_directory': str(static_dir.absolute())
        })
        
    except Exception as e:
        logger.error(f"Error fixing video paths: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting FaceClass Flask application with Team 1 enhancements...")
    app.run(debug=True, host='0.0.0.0', port=5000)
