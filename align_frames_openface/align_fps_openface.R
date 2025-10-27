#!/usr/bin/env Rscript

# Align OpenFace CSV time series down to a target FPS (default 24) using rolling-mean smoothing
# and linear interpolation based on the 'timestamp' column.
#
# Inputs:
#   - Directory of CSV files named like: IDENTIFIER_YYYY-MM-DD_Interviewtype_targetperson.csv
#   - Excel file frame_info.xlsx with columns:
#       Pseudonym, FPS BRFI, FPS STiP, FPS WF
# The interpolation and downsampling is done according to:
# - Rolling mean smoothing (only where source_fps > target_fps)
# Original timestamps: T₀, T₁, …, T_{N−1}
# Original values for some AU: X₀ … X_{N−1}
# Window size: w_raw = ceil(source_fps / target_fps) --> enforced odd for centering
# This operation outputs new AU values (Y) calculated as the mean of its neighbors;
# the values then need to be transformed onto the new timestep grid
# This is done to smooth out jitters in the high frequency of the AU estimates
#
# -- Time resampling (linear interpolation)
# Step = Δ = 1 / target_fps
# Generate new relative times (new_timestep vector): τ_k = k · Δ, for k = 0, 1, …, K
# Z_k (new AU/feature vectors) = Y_i + (Y{i+1} − Y_i) * (T'k − T_i) / (T{i+1} − T_i)
#
# We thus compute a weighted average of the two smoothed neighbor values,
# weighted according to how far away the target value was between its neighbors.
#
# Usage:
# Rscript align_fps.R --in /path/to/OpenFace_Output_MSB/ --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24

suppressPackageStartupMessages({
  library(readxl)
  library(data.table)
  library(zoo)
  library(optparse)
})

# ------------------------------
# Filename parsing
# ------------------------------

parse_identifier_and_type <- function(filename) {
  # Parse identifier and interview type from a filename.
  # IDENTIFIER_YYYY-MM-DD_Interviewtype_In.csv
  # IDENTIFIER_YYYY-MM-DD-Interviewtype-In.csv
  # If the second-to-last token is 'geschnitten', the interview type is the token before it.
  
  name <- basename(filename)
  if (grepl("\\.csv$", name, ignore.case = TRUE)) {
    name <- sub("\\.csv$", "", name, ignore.case = TRUE)
  }
  
  # Replace '-' with '_' for consistent splitting
  name <- gsub("-", "_", name)
  tokens <- strsplit(name, "_")[[1]]
  tokens <- tokens[tokens != ""]
  
  if (length(tokens) < 3) {
    return(NULL)
  }
  
  identifier <- tokens[1]
  penultimate <- tolower(tokens[length(tokens) - 1])
  
  if (penultimate == "geschnitten") {
    if (length(tokens) < 4) {
      return(NULL)
    }
    interview_type <- tokens[length(tokens) - 2]
  } else {
    interview_type <- tokens[length(tokens) - 1]
  }
  
  return(list(
    identifier = trimws(identifier),
    interview_type = trimws(interview_type)
  ))
}

# ------------------------------
# Frame info (Excel) helpers
# ------------------------------

TYPE_TO_COLUMN <- list(
  bindung = "FPS BRFI",
  personal = "FPS STiP",
  wunder = "FPS WF"
)

normalize_type_for_column <- function(interview_type) {
  # Map various interview type strings to the frame_info.xlsx column name.
  # Returns the column name to use in the Excel ('FPS BRFI'/'FPS STiP'/'FPS WF').
  
  tl <- tolower(trimws(interview_type))
  
  if (grepl("bind", tl)) {
    return(TYPE_TO_COLUMN$bindung)
  }
  if (grepl("^pers", tl) || grepl("^st", tl) || grepl("personal", tl)) {
    return(TYPE_TO_COLUMN$personal)
  }
  if (grepl("^wun", tl) || grepl("wf", tl)) {
    return(TYPE_TO_COLUMN$wunder)
  }
  
  return(NULL)
}

load_frame_info <- function(xlsx_path) {
  df <- read_excel(xlsx_path)
  colnames(df) <- trimws(colnames(df))
  df$Pseudonym <- trimws(as.character(df$Pseudonym))
  return(df)
}

get_source_fps <- function(frame_info, identifier, interview_type) {
  col <- normalize_type_for_column(interview_type)
  if (is.null(col)) {
    return(NULL)
  }
  
  row <- frame_info[trimws(as.character(frame_info$Pseudonym)) == trimws(identifier), ]
  
  if (nrow(row) == 0) {
    return(NULL)
  }
  
  fps <- as.numeric(row[[col]][1])
  return(fps)
}

# ------------------------------
# Resampling logic
# ------------------------------

compute_time_axis_from_timestamp <- function(df) {
  # Return times (seconds), the exact timestamp column name used, and (if present) the frame column name.
  # Requires a 'timestamp' column (case-insensitive).
  
  # Normalize header names to strip BOM/whitespace
  norm_cols <- gsub("\ufeff", "", colnames(df))
  norm_cols <- trimws(norm_cols)
  col_map <- setNames(colnames(df), tolower(norm_cols))
  
  ts_col <- col_map["timestamp"]
  if (is.na(ts_col)) {
    stop("Required 'timestamp' column not found (case-insensitive)")
  }
  
  fr_col <- col_map["frame"]
  if (is.na(fr_col)) {
    fr_col <- NULL
  }
  
  times <- as.numeric(df[[ts_col]])
  
  return(list(
    times = times,
    ts_col = ts_col,
    fr_col = fr_col
  ))
}

resample_to_target <- function(df, src_fps, target_fps = 24.0) {
  # Resample dataframe to target fps using rolling-mean smoothing plus interpolation on 'timestamp'.
  # - Recompute 'timestamp' and 'frame' (if present originally, frame is regenerated)
  # - Preserve column order from the original
  # - Non-numeric columns are aligned by nearest original sample
  
  # Clean headers for whitespace
  colnames(df) <- gsub("\ufeff", "", colnames(df))
  colnames(df) <- trimws(colnames(df))
  
  time_info <- compute_time_axis_from_timestamp(df)
  times_old <- time_info$times
  ts_col <- time_info$ts_col
  fr_col <- time_info$fr_col
  
  t0 <- times_old[1]
  times_rel <- times_old - t0
  
  # Sort by timestamp
  df_sorted <- df[order(df[[ts_col]]), ]
  rownames(df_sorted) <- NULL
  
  t_end <- times_rel[length(times_rel)]
  
  # Compute number of new frames and create new timestep grid
  n_new <- floor(t_end * target_fps) + 1
  new_rel <- (0:(n_new - 1)) / target_fps
  new_rel <- round(new_rel, 6)
  
  original_cols <- colnames(df)
  
  # Separate numeric and non-numeric columns (exclude frame/timestamp from numeric set)
  numeric_cols <- colnames(df_sorted)[sapply(df_sorted, is.numeric)]
  numeric_cols <- setdiff(numeric_cols, c(ts_col, fr_col))
  
  non_numeric_cols <- setdiff(original_cols, numeric_cols)
  non_numeric_cols <- setdiff(non_numeric_cols, c(ts_col, fr_col))
  
  # Determine smoothing window (odd, >=1) based on ratio between source and target FPS
  ratio <- src_fps / target_fps
  if (ratio <= 1) {
    window <- 1
  } else {
    window <- ceiling(ratio)
    if (window %% 2 == 0) {
      window <- window + 1
    }
  }
  
  # Interpolate numeric columns on the time grid
  interp_data <- list()
  
  for (col in numeric_cols) {
    series <- as.numeric(df_sorted[[col]])
    
    # Linear interpolation for missing values
    series <- na.approx(series, na.rm = FALSE, rule = 2)
    
    # Apply rolling mean if window > 1
    if (window > 1) {
      series <- rollmean(series, k = window, fill = NA, align = "center")
      # Fill remaining NAs at edges
      series <- na.approx(series, na.rm = FALSE, rule = 2)
    }
    
    # Check if all values are NA
    if (all(is.na(series))) {
      interp_data[[col]] <- rep(NA, n_new)
      next
    }
    
    # Interpolate any remaining NAs
    if (any(is.na(series))) {
      idx <- seq_along(series)
      mask <- !is.na(series)
      series <- approx(idx[mask], series[mask], xout = idx, rule = 2)$y
    }
    
    # Linear interpolation to new time grid
    interp_data[[col]] <- approx(times_rel, series, xout = new_rel, rule = 2)$y
  }
  
  # Align non-numeric columns by nearest original time
  nonnum_df <- data.frame()
  if (length(non_numeric_cols) > 0) {
    non_src <- df_sorted[, non_numeric_cols, drop = FALSE]
    non_src$`__time__` <- times_rel
    new_time_df <- data.frame(`__time__` = new_rel, check.names = FALSE)
    
    # Nearest neighbor join
    setDT(non_src)
    setDT(new_time_df)
    nonnum_df <- non_src[new_time_df, on = "`__time__`", roll = "nearest"]
    nonnum_df[, `__time__` := NULL]
    nonnum_df <- as.data.frame(nonnum_df)
  }
  
  # Build output with original order
  out_cols <- list()
  
  for (col in original_cols) {
    if (col == ts_col) {
      # Write clean, zero-based seconds for timestamp
      out_cols[[col]] <- new_rel
    } else if (!is.null(fr_col) && col == fr_col) {
      # Generate new sequential frame indices starting at 0
      out_cols[[col]] <- 0:(n_new - 1)
    } else if (col %in% names(interp_data)) {
      out_cols[[col]] <- interp_data[[col]]
    } else if (col %in% colnames(nonnum_df)) {
      out_cols[[col]] <- nonnum_df[[col]]
    } else {
      # Column type unknown; carry forward the first value
      out_cols[[col]] <- rep(df_sorted[[col]][1], n_new)
    }
  }
  
  out_df <- as.data.frame(out_cols, check.names = FALSE)
  return(out_df)
}

# ------------------------------
# CLI glue
# ------------------------------

process_directory <- function(in_dir, out_dir, frame_info_xlsx, target_fps = 24.0) {
  # Process all CSVs under in_dir recursively and write outputs preserving subfolders under out_dir.
  # The output structure mirrors the input tree beneath a new parent folder (aligned_XXfps).
  
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  log_path <- file.path(out_dir, "align_fps_log.txt")
  log_con <- file(log_path, open = "w")
  
  log_msg <- function(msg) {
    cat(msg, "\n")
    cat(msg, "\n", file = log_con)
    flush(log_con)
  }
  
  frame_info <- load_frame_info(frame_info_xlsx)
  csv_files <- list.files(in_dir, pattern = "\\.csv$", recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
  csv_files <- sort(csv_files)
  
  if (length(csv_files) == 0) {
    log_msg(sprintf("No CSV files found under %s (recursive)", in_dir))
    close(log_con)
    return()
  }
  
  log_msg(sprintf("Processing %d file(s) to %.1f FPS (recursive)", length(csv_files), target_fps))
  
  for (csv_path in csv_files) {
    parsed <- parse_identifier_and_type(csv_path)
    
    if (is.null(parsed)) {
      log_msg(sprintf("[skip] Cannot parse identifier/type from: %s", basename(csv_path)))
      next
    }
    
    src_fps <- get_source_fps(frame_info, parsed$identifier, parsed$interview_type)
    
    if (is.null(src_fps) || is.na(src_fps)) {
      log_msg(sprintf("[skip] Missing FPS for identifier='%s', type='%s' in frame_info.xlsx",
                      parsed$identifier, parsed$interview_type))
      next
    }
    
    df <- tryCatch({
      fread(csv_path, data.table = FALSE)
    }, error = function(e) {
      log_msg(sprintf("[skip] Failed to read %s: %s", basename(csv_path), e$message))
      return(NULL)
    })
    
    if (is.null(df)) {
      next
    }
    
    # Always run resampling pipeline
    out_df <- tryCatch({
      resample_to_target(df, src_fps = src_fps, target_fps = target_fps)
    }, error = function(e) {
      log_msg(sprintf("[error] Resampling %s: %s", basename(csv_path), e$message))
      return(NULL)
    })
    
    if (is.null(out_df)) {
      next
    }
    
    # Preserve subfolder structure beneath out_dir
    rel_path <- sub(paste0("^", gsub("\\\\", "/", normalizePath(in_dir, winslash = "/")), "/?"), "", 
                    gsub("\\\\", "/", normalizePath(csv_path, winslash = "/")))
    out_path <- file.path(out_dir, rel_path)
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    
    tryCatch({
      fwrite(out_df, out_path)
      log_msg(sprintf("[ok] %s: %.1f -> %.1f FPS (rows %d -> %d)",
                      rel_path, src_fps, target_fps, nrow(df), nrow(out_df)))
    }, error = function(e) {
      log_msg(sprintf("[error] Writing %s: %s", out_path, e$message))
    })
  }
  
  close(log_con)
}

# ------------------------------
# Main entry point
# ------------------------------

main <- function() {
  option_list <- list(
    make_option(c("--in"), dest = "in_dir", type = "character", default = NULL,
                help = "Input directory containing CSV files", metavar = "PATH"),
    make_option(c("--frame-info"), dest = "frame_info", type = "character", default = NULL,
                help = "Path to frame_info.xlsx", metavar = "PATH"),
    make_option(c("--out"), dest = "out_dir", type = "character", default = NULL,
                help = "Output directory for resampled CSVs", metavar = "PATH"),
    make_option(c("--target-fps"), dest = "target_fps", type = "double", default = 24.0,
                help = "Target FPS (default: 24)", metavar = "NUMBER")
  )
  
  opt_parser <- OptionParser(
    option_list = option_list,
    description = "Downsample OpenFace CSVs to a target FPS using frame_info.xlsx metadata"
  )
  opt <- parse_args(opt_parser)
  
  if (is.null(opt$in_dir) || is.null(opt$frame_info) || is.null(opt$out_dir)) {
    print_help(opt_parser)
    stop("Missing required arguments: --in, --frame-info, and --out are required")
  }
  
  in_dir <- normalizePath(opt$in_dir, winslash = "/", mustWork = TRUE)
  frame_info <- normalizePath(opt$frame_info, winslash = "/", mustWork = TRUE)
  
  out_dir_name <- sprintf("aligned_%.0ffps", opt$target_fps)
  out_dir <- file.path(normalizePath(opt$out_dir, winslash = "/", mustWork = FALSE), out_dir_name)
  
  if (!dir.exists(in_dir)) {
    stop(sprintf("Input directory does not exist: %s", in_dir))
  }
  if (!file.exists(frame_info)) {
    stop(sprintf("Frame info Excel not found: %s", frame_info))
  }
  
  process_directory(in_dir, out_dir, frame_info, target_fps = opt$target_fps)
  
  return(0)
}

# Run main if script is executed directly
if (!interactive()) {
  main()
}
