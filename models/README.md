# Models Directory

This directory contains pre-trained models for various computer vision tasks.

## Directory Structure

```
models/
├── face_detection/          # Face detection models
│   ├── yolov8n-face.pt     # YOLO face detection model
│   ├── retinaface.pth      # RetinaFace model
│   └── mtcnn/              # MTCNN model files
├── face_recognition/        # Face recognition models
│   ├── arcface.pth         # ArcFace model
│   ├── facenet.pth         # FaceNet model
│   └── vggface.h5          # VGGFace model
├── emotion_recognition/     # Emotion detection models
│   ├── affectnet.pth       # AffectNet model
│   └── fer2013.h5          # FER2013 model
└── attention_detection/     # Attention detection models
    ├── mediapipe/          # MediaPipe models
    └── openface/           # OpenFace models
```

## Downloading Models

### Face Detection Models

1. **YOLO Face Detection**:
   ```bash
   wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
   mv yolov5n.pt models/face_detection/yolov8n-face.pt
   ```

2. **RetinaFace**:
   ```bash
   # Download from official repository
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   cp Pytorch_Retinaface/weights/Resnet50_Final.pth models/face_detection/retinaface.pth
   ```

3. **MTCNN**:
   ```bash
   pip install mtcnn
   # Models will be downloaded automatically on first use
   ```

### Face Recognition Models

1. **ArcFace**:
   ```bash
   pip install insightface
   # Models will be downloaded automatically
   ```

2. **FaceNet**:
   ```bash
   pip install facenet-pytorch
   # Models will be downloaded automatically
   ```

### Emotion Recognition Models

1. **AffectNet**:
   ```bash
   # Download from official repository
   # Place in models/emotion_recognition/affectnet.pth
   ```

2. **FER2013**:
   ```bash
   # Download from official repository
   # Place in models/emotion_recognition/fer2013.h5
   ```

## Model Configuration

Update the configuration file (`config.yaml`) to specify which models to use:

```yaml
face_detection:
  model: "yolo"  # or "retinaface", "mtcnn", "opencv"

face_recognition:
  model: "arcface"  # or "facenet", "vggface", "opencv"

emotion_detection:
  model: "affectnet"  # or "fer2013", "placeholder"
```

## Model Performance

| Model | Accuracy | Speed (FPS) | Memory (MB) |
|-------|----------|-------------|-------------|
| YOLO | 95% | 30 | 50 |
| RetinaFace | 98% | 15 | 100 |
| MTCNN | 92% | 8 | 80 |
| OpenCV | 85% | 60 | 10 |

## Custom Models

To add custom models:

1. Place model files in appropriate subdirectory
2. Update model loading code in respective modules
3. Add configuration options
4. Test with sample data

## Model Validation

Run model validation tests:

```bash
python -m pytest tests/test_models.py
```

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or use CPU
- **Model not found**: Check file paths and download models
- **Version conflicts**: Use compatible model versions 