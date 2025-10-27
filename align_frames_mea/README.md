# Aligning MEA outputs with different FPS
This folder contains scripts to align MEA time series to a common target frame rate (FPS).

Both Python and R implementations are provided and produce identical results.

IMPORTANT: MEA source files must follow this exact naming convention:

  IDENTIFIER_YEAR-MONTH-DAY_INTERVIEWTYPE_MEA.txt

Example valid filename:

  A0EA_2024-06-04_Bindung_MEA.txt

(i.e., IDENTIFIER_YEAR_MONTH_DAY_INTERVIEWTYPE_MEA.txt — the algorithm expects the interview type in the second-to-last position and the file extension `.txt`. This is because we need to look-up in the frame-info.xlsx file to find the hard-coded FPS rate. We cannot infer that from the source files as they dont have a timestamp column)

## Behavior and assumptions
- The scripts only align the first two columns!
- Frame numbers are taken from the row index (implicit frame 0, 1, 2, ...). The scripts convert frame indices to time in seconds using the source FPS value from `frame_info.xlsx`.
- Source FPS per participant & interview type is read from the `frame_info.xlsx` file with columns `Pseudonym`, `FPS BRFI`, `FPS STiP`, `FPS WF`.
- Resampling: rolling-mean smoothing (when source_fps > target_fps) followed by linear interpolation onto the new time grid.
- Output files preserve subfolder structure and are written as whitespace-delimited TXT files (no header).
- A log `align_fps_log.txt` is created in the output directory summarizing status per file.

---

## Python: align_fps_mea.py
This script aligns MEA TXT time series to a target FPS using rolling-mean smoothing and linear interpolation.

### Command-line arguments
- `--in`: Input directory containing MEA TXT files.
- `--frame-info`: Path to the `frame_info.xlsx` file with FPS metadata.
- `--out`: Output directory where resampled TXT files will be saved (a subfolder called `aligned_{target}fps` will be created).
- `--target-fps`: Target FPS for resampling (default: 24).

### Example
```bash
python align_frames_mea/align_fps_mea.py --in /path/to/MEA/ --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24
```

A log file `align_fps_log.txt` will be created in the output directory, summarizing the processing status for each file.

---

## R: align_fps_mea.R
This R script provides the same functionality as the Python version, aligning MEA TXT time series to a target FPS using rolling-mean smoothing and linear interpolation.

### Command-line arguments
- `--in`: Input directory containing MEA TXT files.
- `--frame-info`: Path to the `frame_info.xlsx` file with FPS metadata.
- `--out`: Output directory where resampled TXT files will be saved (a subfolder called `aligned_{target}fps` will be created).
- `--target-fps`: Target FPS for resampling (default: 24).

### Example
```bash
Rscript align_frames_mea/align_fps_mea.R --in /path/to/MEA/ --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24
```

A log file `align_fps_log.txt` will be created in the output directory, summarizing the processing status for each file.

---

## Notes & troubleshooting
- If a file is skipped, check that its filename matches the naming convention and that the `frame_info.xlsx` contains the corresponding Pseudonym and the correct FPS column for the interview type.
- The R script requires `readxl`, `data.table`, `zoo`, and `optparse`.

