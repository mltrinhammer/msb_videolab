# Aligning OpenFace outputs with different FPS
The code in this repo solves the problem of aligning a set of OpenFace outputs to the same fps-rate, if the source files have different fps.

Both Python and R implementations are provided and produce identical results.

## Python: align_fps.py
This script aligns OpenFace CSV time series to a target FPS using rolling-mean smoothing and linear interpolation.

### Command-line arguments

- `--in`: Input directory containing OpenFace CSV files.
- `--frame-info`: Path to the `frame_info.xlsx` file with FPS metadata. This xlsx file should contain the fps ratings of the recordings. 
- `--out`: Output directory where resampled CSVs will be saved (a subfolder called `aligned_fps` will be created that holds the aligned files).
- `--target-fps`: Target FPS for (default: 24).

### Example
```bash
python align_fps.py --in /path/to/OpenFace_Output_MSB/ --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24
```

A log file `align_fps_log.txt` will be created in the output directory, summarizing the processing status for each file.

---

## R: align_fps.R
This R script provides the same functionality as the Python version, aligning OpenFace CSV time series to a target FPS using rolling-mean smoothing and linear interpolation.

### Command-line arguments

- `--in`: Input directory containing OpenFace CSV files.
- `--frame-info`: Path to the `frame_info.xlsx` file with FPS metadata. This xlsx file should contain the fps ratings of the recordings.
- `--out`: Output directory where resampled CSVs will be saved (a subfolder called `aligned_fps` will be created that holds the aligned files).
- `--target-fps`: Target FPS for resampling (default: 24).

### Example
`Rscript align_fps.R --in /path/to/OpenFace_Output_MSB/ --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24`

A log file `align_fps_log.txt` will be created in the output directory, summarizing the processing status for each file.

