# Sign Language to Text and Speech Conversion

This project implements a real-time sign language recognition system using computer vision and deep learning. It can detect American Sign Language (ASL) gestures through a webcam and convert them to text and speech.

## Features

- Real-time hand gesture detection using MediaPipe
- CNN-based sign language classification for A-Z letters
- Text-to-speech conversion
- GUI interface with Tkinter
- Data collection tools for training

## Files Overview

- `final_pred.py` - Main GUI application with full features
- `prediction_wo_gui.py` - Console-based prediction without GUI
- `data_collection_binary.py` - Tool for collecting binary image data
- `data_collection_final.py` - Tool for collecting skeleton hand data
- `cnn8grps_rad1_model.h5` - Pre-trained CNN model
- `white.jpg` - White background image for skeleton drawing
- `AtoZ_3.1/` - Directory structure for training data

## Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python==4.8.1.78
- cvzone==1.6.1
- tensorflow==2.13.0
- keras==2.13.1
- numpy==1.24.3
- pyttsx3==2.90
- pillow==10.0.1
- pyenchant==3.2.2

## Usage

### Running the GUI Application
```bash
python final_pred.py
```

### Running Console Prediction
```bash
python prediction_wo_gui.py
```

### Data Collection
```bash
# For binary images
python data_collection_binary.py

# For skeleton data
python data_collection_final.py
```

## Controls

- **ESC** - Exit application
- **'a'** - Start/stop data collection (in data collection scripts)
- **'n'** - Next letter (in data collection scripts)

## How It Works

1. **Hand Detection**: Uses CVZone HandTrackingModule to detect hand landmarks
2. **Feature Extraction**: Converts hand landmarks to skeleton representation
3. **Classification**: CNN model predicts the sign language letter
4. **Post-processing**: Applies rules to improve accuracy and handle gestures
5. **Output**: Displays predicted text and converts to speech

## Model Architecture

The CNN model (`cnn8grps_rad1_model.h5`) is trained on 8 groups of similar gestures:
- Group 0: A, E, M, N, S, T
- Group 1: B, D, F, I, K, R, U, V, W
- Group 2: C, O
- Group 3: G, H
- Group 4: L
- Group 5: P, Q, Z
- Group 6: X
- Group 7: Y, J

## Notes

- Ensure good lighting for optimal hand detection
- Keep hand within camera frame
- The model works best with clear, distinct gestures
- Press ESC to exit any running script
