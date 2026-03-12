# Hand Gesture Computer Control

AI-powered hand gesture system that uses MediaPipe, PyTorch, and PyAutoGUI to
control your computer with hand gestures captured from a webcam.

## Project Structure

```
gesture_control/
├── main.py                 # Entry point (record / train / run)
├── hand_tracker.py         # MediaPipe Hands — 21 landmark extraction
├── dataset_recorder.py     # Automated gesture dataset recorder
├── train_model.py          # PyTorch classifier (42→128→64→6)
├── gesture_classifier.py   # Real-time inference wrapper
├── mouse_controller.py     # PyAutoGUI gesture→action mapping
├── smoothing.py            # Majority-vote prediction smoothing
├── utils.py                # Constants, FPS counter, helpers
├── requirements.txt        # Python dependencies
├── dataset/                # Recorded gesture samples (CSV)
└── models/                 # Trained model weights (.pth)
```

## Setup

```bash
cd gesture_control
pip install -r requirements.txt
```

> **GPU note:** For RTX 3070 Ti acceleration install the CUDA build of PyTorch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

## Usage

### Step 1 — Record Gesture Data

```bash
python main.py --record
```

- Show your hand to the webcam.
- Press **0–5** to record samples for each gesture:

| Key | Gesture      |
|-----|--------------|
| 0   | move         |
| 1   | click        |
| 2   | scroll       |
| 3   | drag         |
| 4   | volume_up    |
| 5   | volume_down  |

- Aim for **~100+ samples per gesture** for good accuracy.
- Press **q** or **ESC** to save and exit.
- Samples are appended to `dataset/gestures.csv`.

### Step 2 — Train the Classifier

```bash
python main.py --train --epochs 60 --lr 0.001 --batch 64
```

The best model is saved to `models/gesture_model.pth`.

### Step 3 — Run Gesture Control

```bash
python main.py
```

- Your webcam opens with a live overlay showing landmarks, gesture name,
  confidence, and FPS.
- Recognised gestures automatically control the mouse / keyboard.
- Press **ESC** to exit safely.

## Gesture → Action Mapping

| Gesture      | Action                                |
|--------------|---------------------------------------|
| move         | Move cursor (index finger tip)        |
| click        | Left mouse click                      |
| scroll       | Scroll up                             |
| drag         | Toggle drag (mouseDown / mouseUp)     |
| volume_up    | Volume up media key                   |
| volume_down  | Volume down media key                 |

## Architecture

- **Hand tracking:** MediaPipe Hands (21 landmarks × 2 = 42 normalised values)
- **Classifier:** PyTorch feedforward net — 42 → 128 (ReLU) → 64 (ReLU) → 6
- **Loss:** CrossEntropyLoss · **Optimiser:** Adam
- **Smoothing:** Majority vote over last 5 frames
- **Action cooldown:** 0.4 s between discrete actions to prevent repeats
