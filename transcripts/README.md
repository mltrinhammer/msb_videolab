# Transcript Generation with transcribe.py

This folder contains scripts for generating transcripts from audio/video files using `transcribe.py`.

**Important:** For best performance, run these scripts on the workstation, not on your local laptop!

## Setup

Before running any scripts, install the required Python packages using the provided `sync_requirements.txt` file.

### Windows (PowerShell)
```powershell
python -m venv msb_transcribe
.\msb_transcribe\Scripts\Activate.ps1
pip install -r sync_requirements.txt
```

### macOS (bash/zsh)
```bash
python3 -m venv msb_transcribe
source msb_transcribe/bin/activate
pip install -r sync_requirements.txt
```

## Usage

The main script is `transcribe.py`. It takes audio or video files and generates transcripts using a speech-to-text model (GPU recommended).

### Example command
```bash
python transcribe.py --input /path/to/audio_or_video_file --output /path/to/output_transcript.txt
```

- `--input`: Path to the audio or video file to transcribe.
- `--output`: Path where the transcript will be saved.

## Notes
- Always run on a workstation with a GPU for fast and accurate transcription.
- Make sure you have installed all dependencies from `sync_requirements.txt` before running the script.


