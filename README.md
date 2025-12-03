# Drone & Bird Detection System

## Project Overview
YOLOv8-based object detection and tracking system for identifying and tracking drones and birds in video footage. The model uses real-time object tracking with speed estimation and trajectory visualization.

## Dataset
- **Classes:** 2 (Bird, Drone)
- **Training Images:** 1,027
- **Validation Images:** 293
- **Test Images:** Available
- **Source:** Birds&Drons-1 dataset (Roboflow)

## Model Architecture
- **Base Model:** YOLOv8n (Nano)
- **Parameters:** 3,011,238
- **Input Size:** 640x640
- **Framework:** Ultralytics YOLOv8

## Training Configuration
- **Epochs:** 40 (initial) + fine-tuning
- **Batch Size:** 8 (optimized for RTX 3050 4GB)
- **Learning Rate:** 0.01 (initial), 0.0001 (fine-tuning)
- **Optimizer:** AdamW (auto)
- **Device:** CUDA (NVIDIA GeForce RTX 3050 Laptop GPU)
- **Augmentations:** Mosaic, Flip, HSV, Blur, Median Blur, CLAHE

## Features
### Detection & Tracking
- Real-time object detection with confidence scores
- Persistent multi-object tracking with unique IDs
- Bounding box visualization with class labels

### Speed & Motion Analysis
- Center point tracking (red dots)
- Movement trajectory visualization (purple trails)
- Speed calculation in pixels per second
- Track history retention (30 frames)

### Output
- Processed video with annotations (`output_detections.mp4`)
- Training metrics and plots in `runs/detect/`
- Best and last model weights saved

## File Structure
```
Drone_tarcking/
├── main.py                    # Inference & tracking script
├── train.py                   # Model training script
├── Birds&Drons-1/            # Dataset directory
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/detect/              # Training outputs
│   └── train/weights/
│       ├── best.pt           # Best model weights
│       └── last.pt           # Last checkpoint
└── output_detections.mp4     # Processed output video
```

## Usage

### Training
```bash
python train.py
```

### Inference
```bash
python main.py
```

### Configuration
Edit `main.py` to adjust:
- `VIDEO_SOURCE`: Input video path or webcam (0)
- `CONF_THRES`: Detection confidence threshold (0.4)
- `DISPLAY_WINDOW`: Enable/disable live preview

## Performance
- **Detection Speed:** ~60-70ms per frame
- **FPS:** 30 (input video)
- **Track Persistence:** Maintains consistent IDs across frames
- **GPU Utilization:** Optimized for 4GB VRAM

## Requirements
- Python 3.10+
- PyTorch 2.5.1+cu121
- Ultralytics 8.3.202
- OpenCV (opencv-python)
- NumPy

## Results
The model successfully detects and tracks both birds and drones in real-time with:
- High confidence detections
- Stable tracking across frames
- Accurate speed measurements
- Clear trajectory visualization

---
**Author:** Super-cod  
**Repository:** Drone_tarcking  
**Last Updated:** December 3, 2025
