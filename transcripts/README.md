# Transcript Generation with transcribe.py

This folder contains scripts for generating transcripts from audio/video files using `transcribe.py`.

**Important:** For best performance, run these scripts on the workstation, not on your local laptop!

## Setup
The diarization pipeline requires hugging face token authentication. Make sure to create a profile on hugging face, and get a token. Input this token in the config file.
Before running any scripts, install the required Python packages using the provided `requirements.txt` file. Ensure python 3.11 is installed on your machine and ffmpeg too!

### Windows (PowerShell)
```powershell
# Run the setup script to create the environment and install dependencies
.\setup_environment.ps1
```

### macOS (bash/zsh)
```bash
# Run the setup script to create the environment and install dependencies
chmod +x setup_environment.sh
./setup_environment.sh
```

## Usage

The main script is `transcribe.py`. It takes audio or video files and generates transcripts using a speech-to-text model (GPU recommended).

### Example command
```bash
python transcribe.py --input /path/to/audio_or_video_file --output /path/to/output_transcript
```

- `--input`: Path to the audio or video file to transcribe.
- `--output`: Path where the transcript will be saved.

## Notes
- Always run on a workstation with a GPU for fast and accurate transcription.



