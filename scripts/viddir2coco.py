# viddir2coco.py
"""
Processes a directory of videos to extract frames, run YOLO object detection,
optionally allows for GUI-based review of detections, and outputs results in COCO format.

Frames are extracted to a subdirectory named 'images' within the video directory.
COCO JSON and other outputs are saved to a subdirectory named 'labels' within the video directory.

The simplest way to run this script is:
1. Locate YOLO weights and video collection
2. Run with default parameters:
python scripts/viddir2coco.py 
    --video-dir "..\video_dir" \
    --yolo-model-path "..\weights\best.pt" \
    --enable-detection-review-gui \
    --device cpu
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import shutil
import traceback # For GUI error handling
from typing import Tuple, Dict, Set, List, Optional, Any

from detection_gui import (list_video_files, extract_and_select_frames, 
        YoloDetector, CorrectionGUI, COCOFormatter, prepare_detection_coco_data
)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos, run YOLO detection, optionally review, and output COCO JSON. "
                    "Outputs (frames, COCO JSON) are stored in 'images' and 'labels' subfolders of --video-dir respectively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Path to the directory containing input videos. Outputs will be stored in subfolders here.")
    parser.add_argument("--yolo-model-path", type=str, required=True,
                        help="Path to the Ultralytics YOLO model file (e.g., best.pt).")

    parser.add_argument("--frames-to-sample-per-video", type=int, default=100,
                        help="Number of frames to initially sample uniformly from each video for pre-screening.")
    parser.add_argument("--frames-to-select-per-video", type=int, default=50,
                        help="Maximum number of 'good' frames (meeting detection criteria) to select and save per video after pre-screening.")
    parser.add_argument("--output-detection-coco-name", type=str, default="video_detections_coco.json",
                        help="Filename for the output COCO JSON (detections only).")

    parser.add_argument("--confidence-threshold", type=float, default=0.25,
                        help="Confidence threshold for YOLO predictions.")
    parser.add_argument("--iou-threshold", type=float, default=0.45,
                        help="IoU threshold for Non-Maximum Suppression (NMS).")
    parser.add_argument("--max-detections-per-frame", type=int, default=1,
                        help="Maximum number of detections to keep per frame (after NMS, sorted by confidence). Also used as the upper bound for selecting 'good' frames during video processing.")

    parser.add_argument("--enable-detection-review-gui", action='store_true',
                        help="Enable the Tkinter GUI for reviewing and correcting detections.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for YOLO model inference (e.g., 'cpu', 'cuda:0').")

    return parser.parse_args()

def main(args):
    """Main execution function."""
    # Derive output paths based on video_dir
    video_dir_path = Path(args.video_dir)
    if not video_dir_path.is_dir():
        print(f"ERROR: Video directory not found: {args.video_dir}", file=sys.stderr)
        sys.exit(1)

    extracted_frames_dir = video_dir_path / "images"
    output_dir = video_dir_path / "labels" # For COCO json and other potential script outputs

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_frames_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Video to Detection COCO script...")
    print(f"Parameters: {vars(args)}")
    print(f"Derived extracted frames directory: {extracted_frames_dir}")
    print(f"Derived output directory for labels/COCO: {output_dir}")

    # --- Phase 2: YOLO Detector Implementation ---
    try:
        yolo_detector = YoloDetector(
            model_path=args.yolo_model_path,
            device=args.device,
            conf_thresh=args.confidence_threshold,
            iou_thresh=args.iou_threshold,
            max_det=args.max_detections_per_frame
        )
    except Exception as e:
        print(f"Failed to initialize YoloDetector: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Phase 3: Video Processing and Frame Extraction ---
    print("\n--- Starting Video Processing and Frame Extraction ---")
    video_files = list_video_files(args.video_dir)
    if not video_files:
        print(f"No video files found in {args.video_dir}. Exiting.", file=sys.stderr)
        sys.exit(1)

    all_saved_frame_paths = []
    for video_file_path in video_files: # tqdm for this loop can be added if many videos
        # Pass args.max_detections_per_frame as the criteria for frame selection
        saved_paths_this_video = extract_and_select_frames(
            video_path=video_file_path,
            detector=yolo_detector, # Use the initialized YoloDetector for pre-screening
            num_to_sample=args.frames_to_sample_per_video,
            num_to_select=args.frames_to_select_per_video,
            output_frames_root_dir=extracted_frames_dir,
            max_detections_for_selection=args.max_detections_per_frame 
        )
        all_saved_frame_paths.extend(saved_paths_this_video)

    if not all_saved_frame_paths:
        print("No frames were selected from any videos after pre-screening. Exiting.", file=sys.stderr)
        sys.exit(0) # Not an error, but no data to process
    
    print(f"\nTotal {len(all_saved_frame_paths)} frames selected from all videos for main detection pass.")

    # --- Run Main Detection on All Extracted Frames ---
    print("\n--- Running Main Face Detection on All Selected Extracted Frames ---")
    # detect_directory expects a string path to the directory containing the extracted frames.
    # The frames are saved directly into extracted_frames_dir (not in subdirs per video here)
    initial_detections_on_frames = yolo_detector.detect_directory(str(extracted_frames_dir))
    
    if not initial_detections_on_frames:
        print("No detections found on any of the extracted frames. Exiting.", file=sys.stderr)
        sys.exit(0)
    print(f"Main detection complete. Found detections for {len(initial_detections_on_frames)} images (frames).")

    # Initialize corrected_detections and deleted_paths before potential GUI modification
    # Ensure keys are absolute paths (detect_directory should already provide this)
    corrected_detections = {str(Path(k).resolve()): v for k,v in initial_detections_on_frames.items()}
    deleted_paths_from_review = set()

    # --- Phase 4: GUI Integration ---
    if args.enable_detection_review_gui:
        print("\n--- Launching Detection Correction GUI for Video Frames ---")
        if not corrected_detections: # Should have exited if initial_detections_on_frames was empty
             print("Warning: No detections to review. Skipping GUI.", file=sys.stderr)
        else:
             gui = CorrectionGUI(corrected_detections) # Pass the detections from YoloDetector
             corrected_detections_gui_output, deleted_paths_from_review_gui = gui.run()
             
             # Ensure keys from GUI output are also absolute and paths in set are absolute strings
             corrected_detections = {str(Path(k).resolve()): v for k,v in corrected_detections_gui_output.items()}
             deleted_paths_from_review = {str(Path(p).resolve()) for p in deleted_paths_from_review_gui}
             print(f"Detection review complete. Marked {len(deleted_paths_from_review)} frames for deletion.")
    else:
        print("\n--- Skipping Detection Review GUI Stage ---")
        # corrected_detections is already initial_detections_on_frames (with resolved paths)
        # deleted_paths_from_review is already an empty set

    # --- Phase 5: COCO Output Generation ---
    print("\n--- Preparing Final Detection COCO Data ---")
    # prepare_detection_coco_data expects absolute paths in corrected_detections and deleted_paths_from_review
    final_coco_data = prepare_detection_coco_data(corrected_detections, deleted_paths_from_review)

    print("--- Saving Final Detection COCO Results ---")
    coco_formatter = COCOFormatter()
    # Save COCO annotations in a subdirectory of the main output_dir
    # The output_dir is already effectively <video_dir>/labels
    # output_coco_annotations_dir = output_dir / "annotations_video_detections" # This would make it <video_dir>/labels/annotations_video_detections
    # For simplicity, let's save directly into <video_dir>/labels
    # output_coco_annotations_dir.mkdir(parents=True, exist_ok=True) # output_dir is already created
    output_coco_path = output_dir / args.output_detection_coco_name
    coco_formatter.save_coco_json(final_coco_data, str(output_coco_path))
    
    print(f"\n--- Video to Detection COCO Script Complete! ---")
    print(f"Final detection COCO saved to: {output_coco_path}")
    print(f"Extracted and (potentially reviewed) frames are in: {extracted_frames_dir}")
    if args.enable_detection_review_gui:
        print(f"{len(deleted_paths_from_review)} frames were marked for deletion during GUI review (and excluded from COCO).")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)
