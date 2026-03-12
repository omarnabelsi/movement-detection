# MotionScope v3.0 — CS2 Bot Detection System

Professional real-time CS2 bot detection and tracking using **YOLOv8** deep learning, **SORT** multi-object tracking, and an automated training pipeline.

## Architecture

```
mouce/
├── main.py                         # GUI entry point
├── config.yaml                     # Runtime configuration
├── requirements.txt                # Python dependencies
│
├── dataset_builder/                # Dataset creation pipeline
│   ├── roboflow_download.py        #   Download CS2 dataset from Roboflow
│   ├── frame_extractor.py          #   Extract frames from gameplay videos
│   └── dataset_analyzer.py         #   Class distribution & bias analysis
│
├── augmentation/                   # Data augmentation
│   └── transforms.py               #   Albumentations + OpenCV augmenters
│
├── detector/                       # Detection models
│   ├── yolo_detector.py            #   YOLOv8 inference wrapper
│   └── cs2_heuristics.py           #   CS2-specific post-processing
│
├── tracker/                        # Object tracking
│   └── sort_tracker.py             #   Kalman + Hungarian SORT tracker
│
├── trainer/                        # Model training
│   ├── train.py                    #   YOLOv8 training wrapper
│   └── hyperopt.py                 #   Optuna hyperparameter search
│
├── evaluator/                      # Model evaluation
│   └── metrics.py                  #   Precision, recall, F1, confusion matrix
│
├── active_learning/                # Active learning loop
│   └── loop.py                     #   Save uncertain frames for retraining
│
├── cs2_detector_pipeline/          # Main detection pipeline
│   └── pipeline.py                 #   End-to-end: YOLO → heuristics → SORT → annotate
│
├── detection/                      # Legacy motion detection (fallback)
│   ├── motion_detector.py          #   MOG2 pipeline
│   ├── cs2_bot_detector.py         #   Motion + color CS2 detector
│   └── tracker.py                  #   Centroid tracker
│
├── sources/                        # Video sources
│   ├── camera.py screen.py video.py
│
├── ui/                             # PyQt5 GUI
│   ├── main_window.py              #   Sidebar + dark theme
│   └── video_widget.py             #   Frame display widget
│
├── utils/                          # Utilities
│   ├── config.py fps_controller.py
│
├── data/                           # Datasets (gitignored)
│   ├── cs2_dataset/                #   Roboflow download location
│   └── active_learning/            #   Uncertain frames for review
│
├── models/                         # Trained weights (gitignored)
└── runs/                           # Training outputs (gitignored)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the CS2 Dataset from Roboflow

```bash
python -m dataset_builder.roboflow_download --api-key YOUR_ROBOFLOW_API_KEY
```

This downloads the [CS2 Player Detection](https://universe.roboflow.com/cs2-719ou/cs2-player-detection-ob7hc) dataset (477 images) in YOLOv8 format to `data/cs2_dataset/`.

Get a free API key at [app.roboflow.com](https://app.roboflow.com/).

### 3. (Optional) Augment the Dataset

```bash
python -m augmentation.transforms \
    --images data/cs2_dataset/train/images \
    --labels data/cs2_dataset/train/labels \
    --output data/cs2_dataset_augmented \
    --copies 3
```

### 4. Analyze Dataset for Bias

```bash
python -m dataset_builder.dataset_analyzer \
    --labels data/cs2_dataset/train/labels \
    --class-chart data/class_dist.png \
    --box-chart data/box_sizes.png
```

### 5. Train the YOLO Model

```bash
python -m trainer.train \
    --data data/cs2_dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch 16 \
    --device ""
```

Best weights are saved to `runs/detect/cs2_bot/weights/best.pt`.

### 6. (Optional) Hyperparameter Optimization

```bash
python -m trainer.hyperopt \
    --data data/cs2_dataset/data.yaml \
    --trials 20 \
    --epochs-per-trial 15
```

### 7. Evaluate the Model

```bash
python -m evaluator.metrics \
    --model runs/detect/cs2_bot/weights/best.pt \
    --data data/cs2_dataset/data.yaml
```

### 8. Run the GUI

```bash
python main.py
```

In the sidebar:
1. Select **Screen Capture** or **Video File** as source
2. Check **CS2 Bot Tracking**
3. Check **Use YOLO Model (AI)**
4. Click **Select Model** and choose your `best.pt`
5. Adjust the **Confidence Threshold** (default 35%)
6. Press **Start**

### 9. Extract Frames from Gameplay Videos

```bash
python -m dataset_builder.frame_extractor \
    --video path/to/gameplay.mp4 \
    --every-n 10 \
    --output data/extracted_frames
```

### 10. Active Learning

Uncertain frames are auto-saved to `data/active_learning/` during runtime. Export them for labeling:

```bash
python -m active_learning.loop --dir data/active_learning --export
```

Then label in Roboflow, add to the dataset, and retrain.

## Detection Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| **YOLO AI** | YOLOv8 + CS2 heuristics + SORT tracking | Requires GPU for real-time | Best — trained on CS2 bots |
| **Motion + Color** | MOG2 + shape filter + red/orange color | Fast (CPU only) | Good for visible enemy outlines |
| **Motion Only** | Generic MOG2 motion detection | Fastest | Many false positives |

## Key Components

### YOLOv8 Detector (`detector/yolo_detector.py`)
- GPU auto-detection (CUDA/CPU)
- Configurable confidence and IoU thresholds
- Single-frame and batch inference
- Lazy model loading with warm-up

### CS2 Heuristics (`detector/cs2_heuristics.py`)
- Player size filtering (min/max height)
- Aspect ratio gating (human silhouette proportions)
- HUD region exclusion (top/bottom strips)
- Region-of-interest masking

### SORT Tracker (`tracker/sort_tracker.py`)
- Kalman filter motion prediction
- Hungarian algorithm assignment
- Stable persistent IDs across frames
- Configurable max age and min hits

### Active Learning (`active_learning/loop.py`)
- Saves frames with low-confidence detections
- Saves frames with zero detections (potential false negatives)
- Cooldown to prevent flooding
- JSON metadata for each saved frame

## GPU Support

The system auto-detects CUDA. If a GPU is available, YOLO inference runs on GPU.

To force CPU: set `device: cpu` in `config.yaml` or use `--device cpu` in CLI tools.
