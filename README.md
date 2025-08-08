# FaceClass - Computer Vision Final Project

A comprehensive computer vision system for analyzing classroom videos to detect faces, recognize individuals, analyze emotions, and generate spatial heatmaps for educational insights.

## 🎯 Project Overview

FaceClass is designed to provide educators and researchers with detailed insights into classroom dynamics through automated video analysis. The system can:

- **Detect and track faces** in classroom videos using multiple detection models
- **Recognize known individuals** using face recognition algorithms
- **Analyze emotions and attention** states of students
- **Generate spatial heatmaps** showing activity distribution
- **Provide real-time dashboard** for live monitoring
- **Create comprehensive reports** for educational research

## 📁 Project Structure

```
FaceClass/
│
├── data/                         # Raw and processed data
│   ├── raw_videos/              # Original classroom recordings
│   ├── labeled_faces/           # Labeled face images for recognition
│   ├── heatmaps/                # Generated heatmaps for spatial layout
│   ├── outputs/                 # Processed results (e.g., per frame analysis)
│
├── models/                      # Pre-trained or custom models
│   ├── face_detection/          # e.g., YOLO, RetinaFace
│   ├── face_recognition/        # e.g., ArcFace, FaceNet
│   ├── emotion_recognition/     # e.g., AffectNet model
│   └── attention_detection/     # e.g., OpenFace, MediaPipe
│
├── notebooks/                   # Jupyter notebooks for experimentation
│   ├── face_detection_eval.ipynb
│   ├── emotion_analysis.ipynb
│   └── heatmap_analysis.ipynb
│
├── src/                         # All core source code
│   ├── __init__.py
│   ├── main.py                  # Entry point of the program
│   ├── config.py                # Configuration (paths, thresholds, etc.)
│   ├── utils/                   # Helper functions
│   │   └── video_utils.py
│   ├── detection/               # Face detection and tracking
│   │   └── face_tracker.py
│   ├── recognition/             # Face recognition pipeline
│   │   └── face_identifier.py
│   ├── emotion/                 # Emotion and attention analysis
│   │   └── emotion_detector.py
│   ├── layout_analysis/         # Heatmap and seat mapping
│   │   └── layout_mapper.py
│   └── dashboard/               # Visual dashboard code
│       └── dashboard_ui.py
│
├── reports/                     # Final reports and documentation
│   ├── final_report.pdf
│   ├── figures/                 # Charts, graphs, diagrams
│
├── requirements.txt             # Python package requirements
├── README.md                    # Project overview and how to run
└── .gitignore                   # Files to ignore in version control
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenCV
- PyTorch (for deep learning models)
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd FaceClass
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models (optional):**
   ```bash
   # Create model directories
   mkdir -p models/face_detection
   mkdir -p models/face_recognition
   mkdir -p models/emotion_recognition
   
   # Download models (instructions in model directories)
   ```

### Basic Usage

1. **Run the main analysis pipeline:**
   ```bash
   python src/main.py --video data/raw_videos/sample_classroom.mp4 --mode full
   ```

2. **Launch the dashboard only:**
   ```bash
   python src/main.py --mode dashboard
   ```

3. **Run specific analysis modes:**
   ```bash
   # Face detection only
   python src/main.py --video video.mp4 --mode detection
   
   # Face recognition
   python src/main.py --video video.mp4 --mode recognition
   
   # Emotion analysis
   python src/main.py --video video.mp4 --mode emotion
   ```

### Troubleshooting

**Common Issues:**

1. **"app.run_server has been replaced by app.run" error:**
   - ✅ **Fixed**: Updated dashboard to use `app.run()` instead of `app.run_server()`

2. **"No detections provided for heatmap generation" warning:**
   - ✅ **Fixed**: Added checks for empty detections before calling analysis modules
   - This warning appears when no faces are detected in the video or no video is provided

3. **Dashboard not showing data:**
   - ✅ **Fixed**: Dashboard now receives detection data even when no video is processed
   - The dashboard will show "No data available" when no detections are found

4. **Video file not found:**
   - Make sure your video file exists in the `data/raw_videos/` directory
   - Or provide the full path to your video file

5. **Import errors:**
   - Make sure you're in the correct directory (FaceClass root)
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment: `source venv/bin/activate`

## 🔧 Configuration

The system uses a YAML configuration file (`config.yaml`) for all settings. Key configuration options:

### Face Detection
```yaml
face_detection:
  model: "yolo"  # Options: yolo, retinaface, mtcnn, opencv
  confidence_threshold: 0.5
  nms_threshold: 0.4
  min_face_size: 20
```

### Face Recognition
```yaml
face_recognition:
  model: "arcface"  # Options: arcface, facenet, vggface
  similarity_threshold: 0.6
  embedding_size: 512
```

### Emotion Detection
```yaml
emotion_detection:
  model: "affectnet"  # Options: affectnet, fer2013
  emotions: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
  confidence_threshold: 0.3
```

### Dashboard
```yaml
dashboard:
  port: 8080
  host: "localhost"
  refresh_rate: 1.0
```

## 📊 Features

### 1. Face Detection & Tracking
- **Multiple Models**: Support for YOLO, RetinaFace, MTCNN, and OpenCV
- **Real-time Tracking**: IoU-based tracking across frames
- **Confidence Scoring**: Filter detections by confidence threshold

### 2. Face Recognition
- **Identity Matching**: Compare detected faces with known individuals
- **Database Management**: Add/remove faces from recognition database
- **Similarity Scoring**: Configurable similarity thresholds

### 3. Emotion Analysis
- **Multi-emotion Detection**: 7 basic emotions (happy, sad, angry, etc.)
- **Attention Analysis**: Gaze direction and head pose estimation
- **Engagement Scoring**: Combined attention and emotion metrics

### 4. Spatial Analysis
- **Heatmap Generation**: Presence, attention, and emotion-based heatmaps
- **Seat Assignment**: Automatic assignment to classroom seats
- **Clustering Analysis**: Spatial grouping of detected faces

### 5. Dashboard Interface
- **Real-time Monitoring**: Live updates of analysis results
- **Interactive Charts**: Emotion distribution, attention timelines
- **Spatial Visualization**: Classroom layout with face positions

## 📈 Analysis Outputs

The system generates several types of outputs:

### 1. Detection Results
- JSON files with frame-by-frame detection data
- Bounding boxes, confidence scores, track IDs
- Identity and emotion information

### 2. Heatmaps
- **Presence Heatmap**: Where faces are most frequently detected
- **Attention Heatmap**: Areas with highest attention scores
- **Emotion Heatmap**: Spatial distribution of emotions

### 3. Reports
- **Summary Reports**: Overall statistics and insights
- **Temporal Analysis**: How metrics change over time
- **Spatial Analysis**: Classroom layout and clustering

### 4. Visualizations
- **Classroom Layout**: Interactive seat assignment view
- **Charts**: Emotion distribution, attention trends
- **Heatmaps**: Spatial activity visualization

## 🔬 Research Applications

This system can be used for various educational research purposes:

### 1. Student Engagement Analysis
- Monitor attention levels during lectures
- Identify engagement patterns across different teaching methods
- Track individual student participation

### 2. Classroom Dynamics
- Analyze spatial distribution of student interactions
- Study group formation and clustering
- Assess classroom layout effectiveness

### 3. Teaching Effectiveness
- Evaluate impact of different teaching strategies
- Monitor student reactions to content
- Assess classroom atmosphere and mood

### 4. Accessibility Research
- Study attention patterns of students with different needs
- Analyze effectiveness of accommodations
- Monitor inclusive classroom practices

## 🛠️ Development

### Adding New Models

To add a new face detection model:

1. **Create model class** in `src/detection/`
2. **Update face_tracker.py** to include the new model
3. **Add configuration options** in `config.py`
4. **Update documentation** and examples

### Extending Analysis

To add new analysis features:

1. **Create analysis module** in appropriate `src/` subdirectory
2. **Update main.py** to include new analysis
3. **Add visualization components** to dashboard
4. **Update configuration** for new parameters

### Testing

```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_face_detection.py

# Run with coverage
pytest --cov=src tests/
```

## 📝 Jupyter Notebooks

The project includes several Jupyter notebooks for experimentation:

- **`face_detection_eval.ipynb`**: Compare different detection models
- **`emotion_analysis.ipynb`**: Analyze emotions and attention patterns
- **`heatmap_analysis.ipynb`**: Generate and analyze spatial heatmaps

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- PyTorch team for deep learning framework
- MediaPipe for face mesh and pose estimation
- Academic researchers in educational technology

## 📞 Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Contact the development team
- Check the documentation in the `docs/` directory

## 🔮 Future Work

Planned enhancements:

- **Real-time Processing**: Optimize for live video streams
- **Advanced Analytics**: Machine learning insights and predictions
- **Mobile Support**: iOS/Android applications
- **Cloud Integration**: Web-based analysis platform
- **Privacy Features**: Enhanced data protection and anonymization