# FaceClass Face Detection, Tracking, and Recognition Pipeline

This document describes the improved face detection, tracking, and recognition pipeline for the FaceClass project.

## Overview

The new pipeline implements a robust, multi-stage approach to process video streams and identify students:

1. **Face Detection**: Uses YOLOv8, RetinaFace, MTCNN, or OpenCV for reliable face detection
2. **Face Tracking**: Implements Deep OC-SORT algorithm for robust multi-object tracking
3. **Face Recognition**: Uses ArcFace, FaceNet, DeepFace, or OpenCV for identity matching

## Architecture

```
Video Input → Face Detection → Face Tracking → Face Recognition → Results Storage
     ↓              ↓              ↓              ↓              ↓
  Validation   BBox + Conf    Track IDs     Student IDs    Database
```

## Services

### 1. Face Detection Service (`FaceDetectionService`)

**Location**: `src/services/face_detection.py`

**Features**:
- Multiple detection models (YOLOv8, RetinaFace, MTCNN, OpenCV)
- Automatic fallback to OpenCV if advanced models unavailable
- Configurable confidence and NMS thresholds
- Batch processing support

**Usage**:
```python
from services.face_detection import FaceDetectionService

detector = FaceDetectionService(config)
detections = detector.detect_faces(frame)
```

**Output Format**:
```python
[
    {
        "bbox": [x, y, w, h],
        "confidence": 0.95,
        "class_id": 1
    }
]
```

### 2. Face Tracking Service (`FaceTrackingService`)

**Location**: `src/services/face_tracking.py`

**Features**:
- Deep OC-SORT algorithm implementation
- Hungarian algorithm for optimal track association
- Fallback to greedy assignment if scipy unavailable
- Automatic track lifecycle management
- Motion prediction for occlusions

**Usage**:
```python
from services.face_tracking import FaceTrackingService

tracker = FaceTrackingService(config)
tracked_detections = tracker.update(detections, frame_id)
```

**Output Format**:
```python
[
    {
        "bbox": [x, y, w, h],
        "confidence": 0.95,
        "track_id": 1,
        "frame_id": 0
    }
]
```

### 3. Face Recognition Service (`FaceRecognitionService`)

**Location**: `src/services/face_recognition.py`

**Features**:
- Multiple recognition models (ArcFace, FaceNet, DeepFace, OpenCV)
- Automatic fallback to OpenCV if advanced models unavailable
- Persistent face database with pickle storage
- Configurable similarity thresholds
- Face embedding extraction and storage

**Usage**:
```python
from services.face_recognition import FaceRecognitionService

recognizer = FaceRecognitionService(config)
student_id, confidence = recognizer.identify_face(face_image)
```

**Output Format**:
```python
("STU001", 0.85)  # (student_id, confidence)
```

### 4. Video Processor (`VideoProcessor`)

**Location**: `src/services/video_processor.py`

**Features**:
- Orchestrates all services in a unified pipeline
- Frame-by-frame processing with progress tracking
- Annotated video output generation
- Comprehensive result storage and reporting
- Error handling and recovery

**Usage**:
```python
from services.video_processor import VideoProcessor

processor = VideoProcessor(config)
results = processor.process_video(
    video_path="input.mp4",
    output_dir="output/",
    save_annotated_video=True,
    save_results=True
)
```

## Configuration

The pipeline is configured through `config.yaml`:

```yaml
# Face Detection
face_detection:
  model: "yolo"  # yolo, retinaface, mtcnn, opencv
  confidence_threshold: 0.5
  nms_threshold: 0.4
  min_face_size: 20
  max_faces: 50

# Face Tracking
face_tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  confidence_threshold: 0.5

# Face Recognition
face_recognition:
  model: "arcface"  # arcface, facenet, deepface, opencv
  similarity_threshold: 0.6
  embedding_size: 512
```

## Database Schema

The pipeline stores results in a SQLite database with the following key tables:

### `processed_results`
Stores detection and tracking results for each frame:
- `session_id`: Processing session identifier
- `frame_id`: Frame number
- `track_id`: Unique track identifier
- `student_id`: Recognized student (or NULL if unknown)
- `bbox_x/y/width/height`: Bounding box coordinates
- `confidence`: Detection confidence
- `recognition_confidence`: Recognition confidence
- `timestamp`: Frame timestamp

### `students`
Stores known student information:
- `student_id`: Unique student identifier
- `name`: Student name
- `email`: Student email
- `class`: Student class

### `face_embeddings`
Stores face feature vectors:
- `student_id`: Student identifier
- `embedding_data`: Binary face embedding
- `embedding_type`: Model type used
- `embedding_size`: Embedding dimensions

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download Models** (optional):
```bash
# YOLOv8 models are downloaded automatically
# For other models, place them in models/face_detection/
```

3. **Setup Database**:
```bash
sqlite3 faceclass.db < src/database/schema.sql
```

## Usage Examples

### Basic Video Processing
```python
from services.video_processor import VideoProcessor
from config import Config

# Load configuration
config = Config()

# Initialize processor
processor = VideoProcessor(config)

# Process video
results = processor.process_video("input.mp4")
print(f"Processed {results['processing_stats']['frames_processed']} frames")
```

### Individual Service Usage
```python
from services.face_detection import FaceDetectionService
from services.face_tracking import FaceTrackingService
from services.face_recognition import FaceRecognitionService

# Initialize services
detector = FaceDetectionService(config)
tracker = FaceTrackingService(config)
recognizer = FaceRecognitionService(config)

# Process frame
detections = detector.detect_faces(frame)
tracked_detections = tracker.update(detections, frame_id)

for detection in tracked_detections:
    face_region = extract_face_region(frame, detection['bbox'])
    student_id, confidence = recognizer.identify_face(face_region)
    detection['student_id'] = student_id
    detection['recognition_confidence'] = confidence
```

### Adding New Students
```python
# Add student face to database
success = processor.add_student_face("STU004", face_image)
if success:
    print("Student face added successfully")
```

## Testing

Run the comprehensive test suite:

```bash
python test_face_pipeline.py
```

The test suite validates:
- Individual service functionality
- Complete pipeline integration
- Error handling and recovery
- Performance metrics

## Performance Optimization

### Model Loading
- Models are loaded once during service initialization
- Automatic fallback to lighter models if heavy models unavailable
- Configurable batch processing for multiple frames

### Frame Processing
- Configurable frame skipping for long videos
- Progress logging every 100 frames
- Memory-efficient track history management

### Database Operations
- Indexed queries for fast retrieval
- Batch inserts for bulk operations
- Connection pooling for concurrent access

## Troubleshooting

### Common Issues

1. **Model Loading Failures**:
   - Check if required packages are installed
   - Verify model file paths in configuration
   - Check GPU/CPU compatibility

2. **Memory Issues**:
   - Reduce `max_faces` in detection config
   - Lower `max_age` in tracking config
   - Process videos in smaller chunks

3. **Performance Issues**:
   - Use lighter models (OpenCV instead of YOLO)
   - Increase frame skipping interval
   - Reduce video resolution

### Logging

Enable detailed logging by setting log level in `config.yaml`:
```yaml
logging:
  level: "DEBUG"
```

## Future Enhancements

1. **GPU Acceleration**: CUDA support for YOLO and ArcFace
2. **Real-time Processing**: WebRTC integration for live streams
3. **Advanced Tracking**: Re-identification across camera views
4. **Cloud Integration**: AWS/Azure model hosting
5. **Mobile Support**: Edge device optimization

## Contributing

1. Follow the existing code style and documentation standards
2. Add comprehensive tests for new features
3. Update configuration files and documentation
4. Ensure backward compatibility

## License

This project is licensed under the MIT License - see LICENSE file for details.
