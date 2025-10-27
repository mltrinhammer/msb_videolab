#!/usr/bin/env Rscript

# Align MEA TXT time series down to a target FPS (default 24) using rolling-mean smoothing
# and linear interpolation.
#
# Inputs:
#   - Directory of TXT files named like: IDENTIFIER_YYYY-MM-DD_Interviewtype_MEA.txt
#   - Excel file frame_info.xlsx with columns:
#       Pseudonym, FPS BRFI, FPS STiP, FPS WF
# The interpolation and downsampling is done according to:
# - Rolling mean smoothing (only where source_fps > target_fps)
# Original frame indices: 0, 1, …, N−1
# Original values for features: X₀ … X_{N−1}
# Window size: w_raw = ceil(source_fps / target_fps) --> enforced odd for centering
# This operation outputs new feature values (Y) calculated as the mean of its neighbors;
# the values then need to be transformed onto the new timestep grid
# This is done to smooth out jitters in the high frequency of the feature estimates
#
# -- Time resampling (linear interpolation)
# Step = Δ = 1 / target_fps
# Generate new relative times (new_timestep vector): τ_k = k · Δ, for k = 0, 1, …, K
# Z_k (new feature vectors) = Y_i + (Y{i+1} − Y_i) * (T'k − T_i) / (T{i+1} − T_i)
#
# We thus compute a weighted average of the two smoothed neighbor values,
# weighted according to how far away the target value was between its neighbors.
#
# Usage:
# Rscript align_fps_mea.R --in /path/to/mea/txt/files --frame-info /path/to/frame_info.xlsx --out /path/to/output --target-fps 24

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
  # IDENTIFIER_YYYY-MM-DD_Interviewtype_In.txt
  # IDENTIFIER_YYYY-MM-DD-Interviewtype-In.txt
  # If the second-to-last token is 'geschnitten', the interview type is the token before it.
  
  name <- basename(filename)
  if (grepl("\\.txt$", name, ignore.case = TRUE)) {
    name <- sub("\\.txt$", "", name, ignore.case = TRUE)
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
# Resampling logic for MEA TXT files
# ------------------------------

resample_to_target_mea <- function(df, src_fps, target_fps = 24.0) {
  # Resample MEA dataframe to target fps using rolling-mean smoothing plus interpolation.
  # - Only the first two columns are resampled
  # - Row index represents the frame number
  # - First and second columns are the numeric features to resample
  # - Returns a dataframe with the same two-column structure
  
  # Get the first two columns
  if (ncol(df) < 2) {
    stop("MEA file must have at least 2 columns")
  }
  
  col1 <- colnames(df)[1]
  col2 <- colnames(df)[2]
  
  # Frame numbers are the row indices (0-based in the logic, but R is 1-indexed)
  frames_old <- 0:(nrow(df) - 1)
  
  # Convert frame numbers to time (in seconds) using source FPS
  times_old <- frames_old / src_fps
  t0 <- times_old[1]
  times_rel <- times_old - t0  # should be 0-based
  t_end <- times_rel[length(times_rel)]
  
  # Compute number of new frames and create new timestep grid
  n_new <- floor(t_end * target_fps) + 1
  new_rel <- (0:(n_new - 1)) / target_fps
  new_rel <- round(new_rel, 6)
  
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
  
  # Process both columns
  interp_data <- list()
  
  for (col in c(col1, col2)) {
    series <- as.numeric(df[[col]])
    
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
      idx <- seq_along(series) - 1  # 0-based
      mask <- !is.na(series)
      series <- approx(idx[mask], series[mask], xout = idx, rule = 2)$y
    }
    
    # Linear interpolation to new time grid
    interp_data[[col]] <- approx(times_rel, series, xout = new_rel, rule = 2)$y
  }
  
  # Build output dataframe with original column names (or create generic if no names)
  out_df <- data.frame(
    interp_data[[col1]],
    interp_data[[col2]],
    check.names = FALSE
  )
  
  # Set column names to match input (if they existed)
  if (!is.null(col1) && !is.null(col2)) {
    colnames(out_df) <- c(col1, col2)
  }
  
  return(out_df)
}

# ------------------------------
# CLI glue
# ------------------------------

process_directory <- function(in_dir, out_dir, frame_info_xlsx, target_fps = 24.0) {
  # Process all TXT files under in_dir recursively and write outputs preserving subfolders under out_dir.
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
  txt_files <- list.files(in_dir, pattern = "\\.txt$", recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
  txt_files <- sort(txt_files)
  
  if (length(txt_files) == 0) {
    log_msg(sprintf("No TXT files found under %s (recursive)", in_dir))
    close(log_con)
    return()
  }
  
  log_msg(sprintf("Processing %d file(s) to %.1f FPS (recursive)", length(txt_files), target_fps))
  
  for (txt_path in txt_files) {
    parsed <- parse_identifier_and_type(txt_path)
    
    if (is.null(parsed)) {
      log_msg(sprintf("[skip] Cannot parse identifier/type from: %s", basename(txt_path)))
      next
    }
    
    src_fps <- get_source_fps(frame_info, parsed$identifier, parsed$interview_type)
    
    if (is.null(src_fps) || is.na(src_fps)) {
      log_msg(sprintf("[skip] Missing FPS for identifier='%s', type='%s' in frame_info.xlsx",
                      parsed$identifier, parsed$interview_type))
      next
    }
    
    df <- tryCatch({
      # Read TXT file - assuming whitespace delimited, no header
      fread(txt_path, data.table = FALSE, header = FALSE)
    }, error = function(e) {
      log_msg(sprintf("[skip] Failed to read %s: %s", basename(txt_path), e$message))
      return(NULL)
    })
    
    if (is.null(df)) {
      next
    }
    
    # Always run resampling pipeline
    out_df <- tryCatch({
      resample_to_target_mea(df, src_fps = src_fps, target_fps = target_fps)
    }, error = function(e) {
      log_msg(sprintf("[error] Resampling %s: %s", basename(txt_path), e$message))
      return(NULL)
    })
    
    if (is.null(out_df)) {
      next
    }
    
    # Preserve subfolder structure beneath out_dir
    rel_path <- sub(paste0("^", gsub("\\\\", "/", normalizePath(in_dir, winslash = "/")), "/?"), "", 
                    gsub("\\\\", "/", normalizePath(txt_path, winslash = "/")))
    out_path <- file.path(out_dir, rel_path)
    dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
    
    tryCatch({
      # Write as space-delimited TXT file without header
      fwrite(out_df, out_path, sep = " ", col.names = FALSE)
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
                help = "Input directory containing TXT files", metavar = "PATH"),
    make_option(c("--frame-info"), dest = "frame_info", type = "character", default = NULL,
                help = "Path to frame_info.xlsx", metavar = "PATH"),
    make_option(c("--out"), dest = "out_dir", type = "character", default = NULL,
                help = "Output directory for resampled TXT files", metavar = "PATH"),
    make_option(c("--target-fps"), dest = "target_fps", type = "double", default = 24.0,
                help = "Target FPS (default: 24)", metavar = "NUMBER")
  )
  
  opt_parser <- OptionParser(
    option_list = option_list,
    description = "Downsample MEA TXT files to a target FPS using frame_info.xlsx metadata"
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
