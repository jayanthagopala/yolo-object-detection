# YOLO Object Detection

Experiments with YOLO object detection optimized for Apple Silicon (MacBook Air M4).

## Features

- Latest YOLOv8/YOLOv11 models from Ultralytics
- Apple Silicon optimization with MPS (Metal Performance Shaders)
- Real-time webcam detection
- Image and video processing
- Custom model training capabilities

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.8 or higher
- Webcam (optional, for real-time detection)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jayanthagopala/yolo-object-detection.git
   cd yolo-object-detection
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run basic detection:**
   ```bash
   python detect.py --source 0  # For webcam
   python detect.py --source path/to/your/image.jpg  # For image
   ```

## Usage

### Basic Detection
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # nano model for speed
# model = YOLO('yolov8s.pt')  # small model
# model = YOLO('yolov8m.pt')  # medium model
# model = YOLO('yolov8l.pt')  # large model
# model = YOLO('yolov8x.pt')  # extra large model

# Run inference
results = model('path/to/image.jpg')
results[0].show()  # Display results
```

### Real-time Webcam Detection
```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Apple Silicon Optimization

This project is optimized for Apple Silicon Macs:

- **MPS Backend**: Utilizes Metal Performance Shaders for GPU acceleration
- **Optimized Dependencies**: ARM64-native packages for maximum performance
- **Memory Efficient**: Optimized for unified memory architecture

The models will automatically use MPS (Metal Performance Shaders) if available, providing significant speed improvements over CPU-only inference.

## Model Performance on Apple Silicon

| Model | Size | mAP@0.5:0.95 | Speed (M4) |
|-------|------|--------------|------------|
| YOLOv8n | 6.2MB | 37.3 | ~30ms |
| YOLOv8s | 21.5MB | 44.9 | ~45ms |
| YOLOv8m | 49.7MB | 50.2 | ~70ms |
| YOLOv8l | 83.7MB | 52.9 | ~95ms |
| YOLOv8x | 136.7MB | 53.9 | ~150ms |

*Performance may vary based on image size and complexity*

## Project Structure

```
yolo-object-detection/
├── detect.py              # Main detection script
├── webcam_detect.py       # Real-time webcam detection
├── requirements.txt       # Project dependencies
├── models/               # Downloaded model weights
├── images/               # Sample images for testing
├── results/              # Detection results
└── README.md            # This file
```

## Contributing

Feel free to open issues and pull requests for improvements and bug fixes.

## License

This project is open source and available under the [MIT License](LICENSE).