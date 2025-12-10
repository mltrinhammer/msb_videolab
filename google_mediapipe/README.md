# MediaPipe Face Blendshape Extraction

This folder contains scripts for extracting facial blendshape coefficients from videos using Google's MediaPipe Face Landmarker model.

## What are Blendshapes?

Blendshapes are coefficients (values 0.0-1.0) representing different facial expressions and movements. MediaPipe outputs 52 blendshape scores per frame, including:
- Eye movements (blink, squint, wide)
- Eyebrow positions (inner up, outer up, down)
- Mouth shapes (smile, frown, pucker, funnel)
- Jaw movements (open, left, right, forward)
- Cheek puff
- And many more...

These can be used for:
- Facial expression analysis
- Emotion recognition
- Avatar/character animation
- Virtual effects and filters

## Setup

### Requirements
Install the required packages:

```bash
pip install mediapipe opencv-python numpy pandas
```

Or use a requirements file if provided in this folder.

### Download the Model
Download the MediaPipe face landmarker model:

**Model URL:**
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Save the model file (e.g., to `./models/face_landmarker.task` or any location you prefer).

## Usage

### Basic Command
```bash
python run_mediapipe.py --in /path/to/videos --out /path/to/output --model /path/to/face_landmarker.task
```

### Arguments
- `--in`: Input directory containing video files (searched recursively)
- `--out`: Output directory for CSV files with blendshape data
- `--model`: Path to the `face_landmarker.task` model file
- `--min-detection`: (Optional) Minimum confidence for face detection (0.0-1.0, default: 0.5)
- `--min-tracking`: (Optional) Minimum confidence for face tracking (0.0-1.0, default: 0.5)

### Example
```bash
python run_mediapipe.py \
  --in ./videos \
  --out ./blendshapes \
  --model ./models/face_landmarker.task \
  --min-detection 0.6 \
  --min-tracking 0.5
```

## Output Format

The script generates one CSV file per video with the following columns:
- `frame`: Frame number (0-indexed)
- `timestamp_ms`: Timestamp in milliseconds
- `timestamp_sec`: Timestamp in seconds
- 52 blendshape columns (e.g., `browDownLeft`, `eyeBlinkLeft`, `mouthSmileLeft`, etc.)

Each row represents one frame. If no face is detected in a frame, the blendshape values will be `NaN`.

### Example Output
```
frame,timestamp_ms,timestamp_sec,browDownLeft,browDownRight,eyeBlinkLeft,eyeBlinkRight,...
0,0,0.0,0.123,0.145,0.002,0.001,...
1,33,0.033,0.125,0.147,0.003,0.002,...
2,67,0.067,0.128,0.150,0.005,0.004,...
```

## Blendshape Names (52 total)

The model outputs the following 52 blendshapes:

**Eyebrows:**
- browDownLeft, browDownRight
- browInnerUp
- browOuterUpLeft, browOuterUpRight

**Eyes:**
- eyeBlinkLeft, eyeBlinkRight
- eyeLookDownLeft, eyeLookDownRight
- eyeLookInLeft, eyeLookInRight
- eyeLookOutLeft, eyeLookOutRight
- eyeLookUpLeft, eyeLookUpRight
- eyeSquintLeft, eyeSquintRight
- eyeWideLeft, eyeWideRight

**Cheeks:**
- cheekPuff
- cheekSquintLeft, cheekSquintRight

**Nose:**
- noseSneerLeft, noseSneerRight

**Jaw:**
- jawOpen
- jawForward
- jawLeft, jawRight

**Mouth:**
- mouthClose
- mouthFunnel
- mouthPucker
- mouthLeft, mouthRight
- mouthSmileLeft, mouthSmileRight
- mouthFrownLeft, mouthFrownRight
- mouthDimpleLeft, mouthDimpleRight
- mouthStretchLeft, mouthStretchRight
- mouthRollLower, mouthRollUpper
- mouthShrugLower, mouthShrugUpper
- mouthPressLeft, mouthPressRight
- mouthLowerDownLeft, mouthLowerDownRight
- mouthUpperUpLeft, mouthUpperUpRight

**Tongue:**
- tongueOut

## Notes
- The script processes videos recursively, preserving the directory structure in the output.
- A processing log (`mediapipe_processing_log.txt`) is created in the output directory.
- Face detection rate is reported for each video (percentage of frames with detected faces).
- The script uses VIDEO mode for temporal tracking, which improves performance and accuracy.

## Troubleshooting
- **No face detected:** Try lowering `--min-detection` (e.g., 0.3) if faces are not being detected.
- **Model not found:** Ensure you've downloaded the model file and provided the correct path.
- **Memory issues:** Process videos in smaller batches or reduce video resolution before processing.
- **Slow processing:** This is normal for CPU processing. Use a GPU-enabled environment for faster inference.

## References
- [MediaPipe Face Landmarker Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe Face Landmarker Python Guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Blendshape Model Card (PDF)](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)
