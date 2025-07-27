# YOLO Object Detection with Face Emoji Overlay

A comprehensive object detection project optimized for Apple Silicon (M1/M2/M3/M4) with exciting face emoji overlay features! 

## Features

- **Real-time YOLO object detection** - High-performance object detection using YOLOv8
- **Face emoji overlays** - Fun emoji overlays on detected faces with 9 different emoji options
- **Apple Silicon optimization** - Leverages MPS (Metal Performance Shaders) for fast inference
- **Multiple detection modes** - Webcam, images, videos, and directories
- **Performance monitoring** - Real-time FPS and detection time tracking

## New Face Emoji Features 🎭

### Face Emoji Detection (`face_emoji_detect.py`)
Pure face detection with emoji overlays using OpenCV's Haar cascades:
- **9 different emojis**: 😀 😎 🤣 😍 🤔 😴 🤯 🥳 😈
- **Real-time face detection** with customizable emoji overlays
- **Interactive controls** - Switch emojis with number keys 1-9
- **Performance optimized** for smooth real-time operation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run face emoji detection
python3 face_emoji_detect.py

# With custom camera and emoji size
python3 face_emoji_detect.py --camera 0 --emoji-size 120
```

### YOLO + Face Emoji Detection (`yolo_face_emoji.py`)
Combines YOLO person detection with face emoji overlays:
- **Dual detection**: YOLO for people + Haar cascades for faces
- **Smart face detection** within person bounding boxes for better accuracy
- **Green person boxes** with emoji overlays on detected faces
- **Enhanced performance** by focusing face detection on person regions

```bash
# Run YOLO + face emoji detection
python3 yolo_face_emoji.py

# With custom YOLO model
python3 yolo_face_emoji.py --model yolov8s.pt --emoji-size 100
```

## Controls

### Face Emoji Detection Controls
- **`q`** - Quit application
- **`1-9`** - Switch between different emojis
- **`s`** - Save current frame
- **`r`** - Reset performance statistics
- **`p`** - Print detailed performance stats

### Available Emojis
1. 😀 Happy face
2. 😎 Cool with sunglasses
3. 🤣 Laughing
4. 😍 Heart eyes (love)
5. 🤔 Thinking
6. 😴 Sleeping
7. 🤯 Mind blown
8. 🥳 Party (with hat)
9. 😈 Devil

## Original YOLO Features

### Basic Object Detection (`detect.py`)
```bash
# Webcam detection
python3 detect.py --source 0

# Image detection
python3 detect.py --source image.jpg

# Video detection
python3 detect.py --source video.mp4

# Directory processing
python3 detect.py --source path/to/images/
```

### Real-time Webcam Detection (`webcam_detect.py`)
Optimized webcam detection with performance monitoring:
```bash
python3 webcam_detect.py --model yolov8n.pt --camera 0
```

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd yolo-object-detection
```

2. **Create and activate virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLO model** (automatic on first run)
The YOLOv8 nano model (`yolov8n.pt`) will be downloaded automatically.

## System Requirements

- **Python 3.8+**
- **macOS** (optimized for Apple Silicon, but works on Intel too)
- **Webcam** for real-time detection
- **4GB+ RAM** recommended

## Apple Silicon Optimization

This project is specifically optimized for Apple Silicon Macs:
- **MPS acceleration** - Uses Metal Performance Shaders for GPU acceleration
- **Optimized model loading** - Fast warmup and inference
- **Memory efficient** - Designed for M1/M2/M3/M4 memory architecture

## Performance

On Apple Silicon M2:
- **Face detection**: ~30+ FPS with emoji overlays
- **YOLO + Face**: ~25+ FPS with dual detection
- **YOLO only**: ~35+ FPS object detection

## File Structure

```
yolo-object-detection/
├── detect.py              # Main YOLO detection script
├── webcam_detect.py       # Real-time webcam detection
├── face_emoji_detect.py   # 🆕 Face detection with emoji overlays
├── yolo_face_emoji.py     # 🆕 YOLO + face emoji detection
├── requirements.txt       # Python dependencies
├── yolov8n.pt            # YOLO model (downloaded automatically)
├── test_image.jpg        # Test image
└── results/              # Output directory for saved results
```

## Examples

### Face Emoji Detection
Perfect for fun applications, video calls, and social media:
- Detects faces in real-time
- Overlays customizable emojis
- Smooth performance on Apple Silicon
- Interactive emoji switching

### YOLO + Face Combo
Best for applications requiring both person and face detection:
- First detects people using YOLO
- Then detects faces within person regions
- More accurate face detection
- Complete person + face information

## Troubleshooting

### Camera Access
If you get camera permission errors:
```bash
# Check camera access in System Preferences > Security & Privacy > Camera
```

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Performance Issues
- Ensure you're using the virtual environment: `source .venv/bin/activate`
- Check if MPS is available: The script will show "✅ Using MPS acceleration"
- Try smaller emoji sizes: `--emoji-size 60`
- Close other applications to free up resources

## License

MIT License - Feel free to use and modify for your projects!

## Contributing

Contributions welcome! Feel free to:
- Add new emoji designs
- Improve face detection accuracy
- Optimize performance further
- Add new detection features

---

**Have fun with your new face emoji detection! 🎭✨**