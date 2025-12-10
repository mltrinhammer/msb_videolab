"""
MediaPipe Face Blendshape Extraction for Videos

This script processes video files and extracts 52 facial blendshape coefficients
using Google's MediaPipe Face Landmarker model. Blendshapes represent facial
expressions and can be used for animation, emotion analysis, and avatar creation.

The model outputs:
- 52 blendshape scores per face per frame (coefficients 0.0-1.0)
- Face landmarks (478 3D points)
- Facial transformation matrices (optional)

Usage:
    python run_mediapipe.py --in /path/to/videos --out /path/to/output --model /path/to/face_landmarker.task

Requirements:
    - mediapipe
    - opencv-python
    - numpy
    - pandas

Model download:
    Download the face_landmarker.task model from:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def download_model_if_needed(model_path: Path) -> None:
    """Download the MediaPipe face landmarker model if it doesn't exist."""
    if model_path.exists():
        return
    
    print(f"Model not found at {model_path}")
    print("Please download the model from:")
    print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
    print(f"and save it to: {model_path}")
    sys.exit(1)


def extract_blendshapes_from_video(
    video_path: Path,
    output_path: Path,
    model_path: Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """
    Extract blendshape coefficients from a video file.
    
    Args:
        video_path: Path to input video file
        output_path: Path to output CSV file
        model_path: Path to face_landmarker.task model file
        min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
        min_tracking_confidence: Minimum confidence for face tracking (0.0-1.0)
    """
    print(f"Processing: {video_path.name}")
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"  Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # Create face landmarker options
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    
    # Initialize results storage
    all_results = []
    blendshape_names = None
    
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Calculate timestamp in milliseconds
                timestamp_ms = int(frame_idx * 1000 / fps)
                
                # Detect face landmarks and blendshapes
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Extract blendshape data
                if result.face_blendshapes and len(result.face_blendshapes) > 0:
                    # Get blendshapes for the first detected face
                    blendshapes = result.face_blendshapes[0]
                    
                    # Store blendshape names on first detection
                    if blendshape_names is None:
                        blendshape_names = [bs.category_name for bs in blendshapes]
                    
                    # Extract scores
                    scores = [bs.score for bs in blendshapes]
                    
                    # Store frame data
                    row = {
                        'frame': frame_idx,
                        'timestamp_ms': timestamp_ms,
                        'timestamp_sec': timestamp_ms / 1000.0,
                    }
                    
                    # Add blendshape scores
                    for name, score in zip(blendshape_names, scores):
                        row[name] = score
                    
                    all_results.append(row)
                else:
                    # No face detected - store NaN values
                    row = {
                        'frame': frame_idx,
                        'timestamp_ms': timestamp_ms,
                        'timestamp_sec': timestamp_ms / 1000.0,
                    }
                    
                    if blendshape_names is not None:
                        for name in blendshape_names:
                            row[name] = np.nan
                    
                    all_results.append(row)
                
                frame_idx += 1
                
                # Progress indicator
                if frame_idx % 100 == 0:
                    progress = (frame_idx / frame_count) * 100
                    print(f"  Progress: {frame_idx}/{frame_count} ({progress:.1f}%)")
        
        # Save results to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            detected_frames = df.dropna(subset=blendshape_names if blendshape_names else []).shape[0]
            detection_rate = (detected_frames / len(df)) * 100 if len(df) > 0 else 0
            
            print(f"  [ok] Saved {len(df)} frames to {output_path.name}")
            print(f"  Face detection rate: {detection_rate:.1f}% ({detected_frames}/{len(df)} frames)")
        else:
            print(f"  [warning] No frames processed")
    
    except Exception as e:
        print(f"  [error] Processing failed: {e}")
    
    finally:
        cap.release()


def process_directory(
    in_dir: Path,
    out_dir: Path,
    model_path: Path,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """
    Process all video files in a directory recursively.
    
    Args:
        in_dir: Input directory containing video files
        out_dir: Output directory for CSV files
        model_path: Path to face_landmarker.task model
        min_detection_confidence: Minimum confidence for face detection
        min_tracking_confidence: Minimum confidence for face tracking
    """
    # Common video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Find all video files recursively
    video_files = []
    for ext in video_extensions:
        video_files.extend(in_dir.rglob(f"*{ext}"))
        video_files.extend(in_dir.rglob(f"*{ext.upper()}"))
    
    video_files = sorted(set(video_files))
    
    if not video_files:
        print(f"No video files found in {in_dir}")
        return
    
    print(f"Found {len(video_files)} video file(s)")
    print(f"Output directory: {out_dir}")
    print()
    
    # Create log file
    log_path = out_dir / "mediapipe_processing_log.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with log_path.open("w", encoding="utf-8") as log_file:
        def log(msg: str) -> None:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
        
        log(f"Processing {len(video_files)} video file(s)")
        log(f"Model: {model_path}")
        log(f"Min detection confidence: {min_detection_confidence}")
        log(f"Min tracking confidence: {min_tracking_confidence}")
        log("")
        
        for idx, video_path in enumerate(video_files, 1):
            log(f"[{idx}/{len(video_files)}] {video_path.name}")
            
            # Preserve directory structure
            try:
                rel_path = video_path.relative_to(in_dir)
            except ValueError:
                rel_path = Path(video_path.name)
            
            # Output CSV path (replace extension with .csv)
            output_path = out_dir / rel_path.with_suffix('.csv')
            
            try:
                extract_blendshapes_from_video(
                    video_path=video_path,
                    output_path=output_path,
                    model_path=model_path,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            except Exception as e:
                log(f"  [error] Unexpected error: {e}")
            
            log("")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Extract MediaPipe face blendshapes from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mediapipe.py --in /path/to/videos --out /path/to/output --model face_landmarker.task
  python run_mediapipe.py --in ./videos --out ./blendshapes --model ./models/face_landmarker.task --min-detection 0.6

Model download:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
        """
    )
    
    ap.add_argument(
        "--in",
        dest="in_dir",
        required=True,
        help="Input directory containing video files (searched recursively)"
    )
    ap.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        help="Output directory for CSV files with blendshape data"
    )
    ap.add_argument(
        "--model",
        dest="model_path",
        required=True,
        help="Path to face_landmarker.task model file"
    )
    ap.add_argument(
        "--min-detection",
        dest="min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for face detection (0.0-1.0, default: 0.5)"
    )
    ap.add_argument(
        "--min-tracking",
        dest="min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for face tracking (0.0-1.0, default: 0.5)"
    )
    
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    
    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    
    # Validate inputs
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Error: Input directory does not exist: {in_dir}")
        return 2
    
    # Check model file
    download_model_if_needed(model_path)
    
    # Validate confidence values
    if not (0.0 <= args.min_detection_confidence <= 1.0):
        print(f"Error: min-detection must be between 0.0 and 1.0")
        return 2
    
    if not (0.0 <= args.min_tracking_confidence <= 1.0):
        print(f"Error: min-tracking must be between 0.0 and 1.0")
        return 2
    
    # Process videos
    process_directory(
        in_dir=in_dir,
        out_dir=out_dir,
        model_path=model_path,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
