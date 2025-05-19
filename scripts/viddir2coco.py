# viddir2coco.py
"""
Processes a directory of videos to extract frames, run YOLO object detection,
optionally allows for GUI-based review of detections, and outputs results in COCO format.

Example Usage:
python viddir2coco.py \\
    --video-dir /path/to/videos \\
    --extracted-frames-dir /path/to/output/extracted_frames \\
    --output-dir /path/to/output \\
    --yolo-model-path /path/to/your_model.pt \\
    --frames-to-sample-per-video 100 \\
    --frames-to-select-per-video 20 \\
    --output-detection-coco-name video_face_detections.json \\
    --confidence-threshold 0.3 \\
    --max-detections-per-frame 2 \\
    --enable-detection-review-gui \\
    --device cuda:0
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

# --- Constants for Drawing (for CorrectionGUI) ---
# Helper function to convert BGR tuple to Hex string for Tkinter
def bgr_to_hex(bgr_tuple):
    return f"#{bgr_tuple[2]:02x}{bgr_tuple[1]:02x}{bgr_tuple[0]:02x}"

DET_BBOX_COLOR_MAP = {
    "face": bgr_to_hex((0, 0, 255)),    # Red
    "default": bgr_to_hex((128, 128, 128)) # Gray
}
DET_SELECTED_COLOR = bgr_to_hex((255, 100, 255)) # Pinkish
DET_HANDLE_COLOR = bgr_to_hex((255, 255, 0))   # Cyan
DET_TEXT_COLOR = "#CCCCCC" # Light gray hex
DET_TEMP_BOX_COLOR = bgr_to_hex((0, 255, 0))   # Green

class YoloDetector:
    """Handles running an Ultralytics YOLO model and managing results."""
    def __init__(self, model_path: str, device: str, 
                 conf_thresh: float = 0.25, 
                 iou_thresh: float = 0.45, 
                 max_det: int = 1):
        """
        Initialize YOLO detection model.

        Args:
            model_path (str): Path to the YOLO model weights (.pt file).
            device (str): Device to run model on (e.g., 'cpu', 'cuda:0').
            conf_thresh (float): Confidence threshold for detections.
            iou_thresh (float): IoU threshold for NMS.
            max_det (int): Maximum number of detections to return per image.
        """
        try:
            self.model = YOLO(model_path)
            self.model.to(device) # Ensure model is on the specified device
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model from {model_path} on device {device}: {e}", file=sys.stderr)
            raise # Re-raise exception to halt execution if model loading fails
        
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det

        print(f"YOLO model {model_path} loaded successfully on {device}.")
        print(f"  Confidence threshold: {self.conf_thresh}")
        print(f"  IoU threshold: {self.iou_thresh}")
        print(f"  Max detections per frame: {self.max_det}")


    def detect_single_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single image frame.

        Args:
            frame (np.ndarray): The image frame (BGR format).

        Returns:
            List[Dict{'box': [x1,y1,x2,y2], 'label': 'face', 'confidence': float}]
        """
        if frame is None:
            return []

        processed_bboxes_for_img = []
        try:
            # Perform inference
            results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det, # Handled by predict
                # classes=self.target_classes, # Uncomment if specific classes are targeted
                device=self.device,
                verbose=False  # Suppress Ultralytics' own console output for predictions
            )

            if results and results[0]: # results is a list, results[0] is for the first image
                res = results[0]
                boxes = res.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] in pixel coords
                confidences = res.boxes.conf.cpu().numpy()
                # classes = res.boxes.cls.cpu().numpy() # If needed

                for i in range(len(boxes)):
                    box = boxes[i].tolist()
                    confidence = float(confidences[i])
                    # cls_id = int(classes[i])
                    # label = self.model.names[cls_id] if hasattr(self.model, 'names') else 'face' # Use model names if available
                    label = 'face'

                    processed_bboxes_for_img.append({
                        'box': box,
                        'label': label,
                        'confidence': confidence
                    })

        except Exception as e:
            print(f"ERROR during YOLO prediction on a frame: {e}", file=sys.stderr)
            # Optionally, print traceback: traceback.print_exc()
            return [] # Return empty list on error

        return processed_bboxes_for_img

    def detect_directory(self, img_dir_str: str) -> Dict[str, List[Dict]]:
        """
        Run detection on all images in a directory. Images are searched non-recursively.

        Args:
            img_dir_str (str): Path to the image directory.

        Returns:
            dict: {absolute_image_path: List[Dict{'box': [x1,y1,x2,y2], 'label': 'face', 'confidence': float}]}
        """
        img_dir = Path(img_dir_str)
        if not img_dir.is_dir():
            print(f"ERROR: Image directory not found: {img_dir_str}", file=sys.stderr)
            return {}

        image_paths = []
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        for ext in extensions:
            image_paths.extend(list(img_dir.glob(ext)))
        
        if not image_paths:
            print(f"No images found in {img_dir} with extensions: {extensions}")
            return {}

        results_dict = {}
        # Ensure paths are absolute strings for consistency and sort them
        absolute_image_paths = sorted([str(p.resolve()) for p in image_paths])

        for img_path_str in tqdm(absolute_image_paths, desc=f"Running YOLO detection in {img_dir.name}"):
            frame = cv2.imread(img_path_str)
            if frame is None:
                print(f"Warning: Failed to read image {img_path_str}, skipping.", file=sys.stderr)
                continue
            
            detections = self.detect_single_frame(frame)
            results_dict[img_path_str] = detections
            
        return results_dict


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos, run YOLO detection, optionally review, and output COCO JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video-dir", type=str, required=True,
                        help="Path to the directory containing input videos.")
    parser.add_argument("--extracted-frames-dir", type=str, required=True,
                        help="Path where selected frames from videos will be saved.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the output directory for COCO JSON and other outputs.")
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
    output_dir = Path(args.output_dir)
    extracted_frames_dir = Path(args.extracted_frames_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_frames_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Video to Detection COCO script...")
    print(f"Parameters: {vars(args)}")

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
    output_coco_annotations_dir = output_dir / "annotations_video_detections"
    output_coco_annotations_dir.mkdir(parents=True, exist_ok=True)
    output_coco_path = output_coco_annotations_dir / args.output_detection_coco_name
    coco_formatter.save_coco_json(final_coco_data, str(output_coco_path))
    
    print(f"\n--- Video to Detection COCO Script Complete! ---")
    print(f"Final detection COCO saved to: {output_coco_path}")
    print(f"Extracted and (potentially reviewed) frames are in: {extracted_frames_dir}")
    if args.enable_detection_review_gui:
        print(f"{len(deleted_paths_from_review)} frames were marked for deletion during GUI review (and excluded from COCO).")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)

# --- Video Processing Helper Functions (Adapted from pseudolabel_gui.py) ---
def list_video_files(video_dir_str: str) -> List[Path]:
    """Scans video_dir for common video file extensions."""
    video_dir = Path(video_dir_str)
    if not video_dir.is_dir():
        print(f"ERROR: Video directory not found: {video_dir_str}", file=sys.stderr)
        return []
        
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'] # Common video extensions
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f'*{ext}')))
        video_files.extend(list(video_dir.glob(f'*{ext.upper()}'))) # Add uppercase for robustness

    unique_video_files = sorted(list(set([f for f in video_files if f.is_file()])))
    
    if not unique_video_files:
        print(f"No video files found in {video_dir} with extensions: {video_extensions}")
    else:
        print(f"Found {len(unique_video_files)} video files in {video_dir}.")
    return unique_video_files

def extract_and_select_frames(
    video_path: Path,
    detector: YoloDetector, # Changed to YoloDetector
    num_to_sample: int,
    num_to_select: int,
    output_frames_root_dir: Path,
    max_detections_for_selection: int # Added: for criteria 1 <= len(detections) <= max_detections_for_selection
) -> List[str]:
    """
    Extracts frames from a video, runs YOLO detection, selects frames meeting criteria,
    and saves them to output_frames_root_dir with video stem prefixed to filename.
    Selection criteria: 1 <= number of detected faces <= max_detections_for_selection.
    """
    print(f"Processing video: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}", file=sys.stderr)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames or metadata error.", file=sys.stderr)
        cap.release()
        return []

    actual_num_to_sample = min(num_to_sample, total_frames)
    if actual_num_to_sample < num_to_sample:
        print(f"  Warning: Requested {num_to_sample} frames to sample, but video only has {total_frames}. Sampling {actual_num_to_sample} frames.")

    if actual_num_to_sample == 0:
        cap.release()
        return[]
        
    frame_indices_to_sample = np.linspace(0, total_frames - 1, actual_num_to_sample, dtype=int)
    
    good_candidate_frames_data = [] # Stores {'frame': frame_image, 'original_index': frame_idx}
    
    print(f"  Sampling {len(frame_indices_to_sample)} frames for pre-screening (Targeting up to {num_to_select} good frames)...", flush=True)
    for frame_idx in tqdm(frame_indices_to_sample, desc=f"  Pre-screening {video_path.stem}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        detections = detector.detect_single_frame(frame) # Use YoloDetector instance
        
        # Condition: number of detections is between 1 and max_detections_for_selection (inclusive)
        if 1 <= len(detections) <= max_detections_for_selection:
            good_candidate_frames_data.append({'frame': frame.copy(), 'original_index': frame_idx})
            if len(good_candidate_frames_data) >= num_to_select:
                 print(f"  Reached {num_to_select} good frames early for {video_path.stem} at frame index {frame_idx}. Stopping pre-screen.", flush=True)
                 break 
    cap.release()

    selected_saved_frame_paths = []
    if not good_candidate_frames_data:
        print(f"  No frames meeting selection criteria (1 to {max_detections_for_selection} faces) found after pre-screening for {video_path.name}.", flush=True)
        return []

    frames_to_save = good_candidate_frames_data[:num_to_select]
    output_frames_root_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Saving {len(frames_to_save)} selected frames for {video_path.name} to {output_frames_root_dir}", flush=True)
    video_stem = video_path.stem
    for frame_data in frames_to_save: # No tqdm here, usually small number
        original_idx = frame_data['original_index']
        frame_to_save = frame_data['frame']
        save_path = output_frames_root_dir / f"{video_stem}_frame_{original_idx:06d}.png" # Ensure video_stem is part of filename
        try:
            cv2.imwrite(str(save_path), frame_to_save)
            selected_saved_frame_paths.append(str(save_path.resolve()))
        except Exception as e:
            print(f"Error saving frame {save_path}: {e}", file=sys.stderr)
            
    print(f"  Selected and saved {len(selected_saved_frame_paths)} frames from {video_path.name}.", flush=True)
    return selected_saved_frame_paths

# --- CorrectionGUI (Adapted from pseudolabel_gui.py) ---
class CorrectionGUI:
    """Tkinter GUI for bbox correction"""
    def __init__(self, image_bbox_dict: Dict[str, List[Dict]]):
        """
        Initialize GUI for bbox correction

        Args:
            image_bbox_dict (dict): {image_path: List[Dict{'box': [x1,y1,x2,y2], ...}]}
        """
        self.image_paths = list(image_bbox_dict.keys())
        self.image_paths.sort() # Ensure consistent order
        self.bboxes_data = image_bbox_dict # Store the full dict list
        self.current_idx = 0
        self.modified = False
        self.deleted_images = set()  # Track deleted images
        self.creating_new_box = False  # Track if we're in new box creation mode

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Bounding Box Correction GUI")

        # Setup GUI components
        self._setup_gui()

        # Mouse interaction state
        self.dragging = False
        self.drag_start = None
        self.selected_box_internal_idx = None # Index within the list for the current image
        self.drag_type = None  # 'move', 'resize', 'create', or None
        self.new_box_start = None
        self.handle_size = 8  # Size of corner handles
        self.selected_corner = None # To track which handle is being dragged
        self.temp_box_id = None # For drawing temporary new box

        if not self.image_paths:
            print("Warning: No images found for CorrectionGUI.", file=sys.stderr)
            self.root.after(100, self.root.quit) # Schedule quit if no images
            return
        self._load_current_image()

    def _setup_gui(self):
        """Setup GUI layout and controls"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        instructions = (
            "'n': New box mode (LMB drag to create)\n"
            "LMB drag: Move/resize selected box\n"
            "RMB on box: Delete box\n"
            "Del key: Delete current image\n"
            "Left/Right arrows: Prev/Next image\n"
            "Esc: Complete editing & close"
        )
        ttk.Label(main_frame, text=instructions, justify=tk.LEFT).pack(pady=5, anchor=tk.W)

        self.canvas = tk.Canvas(main_frame, bg='#333333') # Darker gray canvas
        self.canvas.pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)

        ttk.Button(ctrl_frame, text="Previous (<-)", command=self._prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Next (->)", command=self._next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Complete (Esc)", command=self._complete).pack(side=tk.RIGHT, padx=5)

        self.progress_var = tk.StringVar()
        ttk.Label(ctrl_frame, textvariable=self.progress_var).pack(side=tk.LEFT, padx=20)

        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Button-3>', self._on_right_click)

        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())
        self.root.bind('<Delete>', lambda e: self._delete_current_image())
        self.root.bind('<Escape>', lambda e: self._complete())
        self.root.bind('n', self._toggle_new_box_mode)

    def _load_current_image(self):
        if not self.image_paths or not (0 <= self.current_idx < len(self.image_paths)):
            if not self.image_paths:
                print("No images remaining to display.", file=sys.stderr)
                self._complete()
            return

        img_path = self.image_paths[self.current_idx]
        try:
            self.current_image_cv = cv2.imread(img_path)
            if self.current_image_cv is None: raise IOError(f"Failed to load image: {img_path}")
            self.current_image_rgb = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}", file=sys.stderr)
            self.canvas.delete("all")
            error_img_pil = Image.new('RGB', (800, 600), color='darkred')
            self.photo = ImageTk.PhotoImage(error_img_pil)
            self.canvas.config(width=800, height=600)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.create_text(400, 300, text=f"Error loading:\n{os.path.basename(img_path)}", fill="white", font=("Arial", 16), justify=tk.CENTER)
            self.progress_var.set(f"Error loading Image {self.current_idx + 1}")
            return

        h, w = self.current_image_rgb.shape[:2]
        max_w, max_h = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        if max_w <= 1 or max_h <=1: max_w,max_h = 800,600 # Fallback if canvas not drawn
        
        self.scale = min(max_w/w, max_h/h) if w > 0 and h > 0 else 1.0
        self.display_w, self.display_h = int(w * self.scale), int(h * self.scale)

        img_pil = Image.fromarray(self.current_image_rgb).resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all") # Clear previous image and boxes
        self.canvas.config(width=self.display_w, height=self.display_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags='image')
        self._draw_boxes()
        self._update_progress()

    def _update_progress(self):
        total_initial_imgs = len(self.image_paths) + len(self.deleted_images)
        current_display_count = len(self.image_paths)
        if current_display_count > 0:
            self.progress_var.set(f"Image {self.current_idx + 1} of {current_display_count} (Initial total: {total_initial_imgs})")
        else:
            self.progress_var.set("No images to display.")

    def _draw_boxes(self):
        self.canvas.delete('box')
        self.canvas.delete('handle')
        if not self.image_paths or not (0 <= self.current_idx < len(self.image_paths)): return

        img_path = self.image_paths[self.current_idx]
        current_image_boxes = self.bboxes_data.get(img_path, [])

        for i, box_data in enumerate(current_image_boxes):
            box = box_data['box'] # [x1,y1,x2,y2] original coords
            label = box_data.get('label', 'face')
            
            x1s, y1s, x2s, y2s = [c * self.scale for c in box]

            color_hex = DET_SELECTED_COLOR if i == self.selected_box_internal_idx else DET_BBOX_COLOR_MAP.get(label, DET_BBOX_COLOR_MAP['default'])
            thickness = 3 if i == self.selected_box_internal_idx else 2

            self.canvas.create_rectangle(x1s, y1s, x2s, y2s, outline=color_hex, width=thickness, tags=('box', f'box{i}'))
            
            label_text = f"{label}" 
            if 'confidence' in box_data:
                 label_text += f": {box_data['confidence']:.2f}"
            text_x, text_y = x1s, y1s - 10 if y1s > 10 else y1s + 5
            self.canvas.create_text(text_x, text_y, text=label_text, fill=color_hex, anchor=tk.NW if y1s > 10 else tk.SW, tags=('box', f'box{i}'), font=("Arial", 8))

            if i == self.selected_box_internal_idx:
                handles_coords = [(x1s, y1s), (x2s, y1s), (x1s, y2s), (x2s, y2s),
                                  (x1s, (y1s+y2s)/2), (x2s, (y1s+y2s)/2), # Mid sides L/R
                                  ((x1s+x2s)/2, y1s), ((x1s+x2s)/2, y2s)] # Mid sides T/B
                handle_names = ['topleft', 'topright', 'bottomleft', 'bottomright',
                                'midleft', 'midright', 'midtop', 'midbottom']
                for (hx, hy), name in zip(handles_coords, handle_names):
                    self.canvas.create_rectangle(hx - self.handle_size/2, hy - self.handle_size/2, 
                                                 hx + self.handle_size/2, hy + self.handle_size/2, 
                                                 fill=DET_HANDLE_COLOR, outline='black', tags=('handle', f'box{i}', name))

    def _get_handle_at_pos(self, click_x: float, click_y: float) -> Optional[str]:
        if self.selected_box_internal_idx is None: return None
        img_path = self.image_paths[self.current_idx]
        box_data = self.bboxes_data[img_path][self.selected_box_internal_idx]
        x1s, y1s, x2s, y2s = [c * self.scale for c in box_data['box']]
        
        handles_map = {
            'topleft': (x1s, y1s), 'topright': (x2s, y1s), 'bottomleft': (x1s, y2s), 'bottomright': (x2s, y2s),
            'midleft': (x1s, (y1s+y2s)/2), 'midright': (x2s, (y1s+y2s)/2),
            'midtop': ((x1s+x2s)/2, y1s), 'midbottom': ((x1s+x2s)/2, y2s)
        }
        for name, (hx, hy) in handles_map.items():
            if abs(click_x - hx) <= self.handle_size/2 and abs(click_y - hy) <= self.handle_size/2:
                return name
        return None

    def _find_box_at_pos(self, click_x: float, click_y: float) -> Optional[int]:
        if not self.image_paths or not (0 <= self.current_idx < len(self.image_paths)): return None
        img_path = self.image_paths[self.current_idx]
        for i in range(len(self.bboxes_data[img_path]) - 1, -1, -1): # Topmost first
            x1s, y1s, x2s, y2s = [c * self.scale for c in self.bboxes_data[img_path][i]['box']]
            if x1s <= click_x <= x2s and y1s <= click_y <= y2s:
                return i
        return None

    def _on_mouse_down(self, event):
        self.drag_start = (event.x, event.y)
        if self.creating_new_box:
            self.drag_type = 'create'
            self.new_box_start = (event.x, event.y)
            self.selected_box_internal_idx = None
            self._draw_boxes() # Redraw to deselect any existing box
            self.dragging = True
            return

        clicked_handle = self._get_handle_at_pos(event.x, event.y)
        if clicked_handle:
            self.drag_type = 'resize'
            self.selected_corner = clicked_handle # This now stores string like 'topleft', 'midright'
            img_path = self.image_paths[self.current_idx]
            self.drag_start_box = list(self.bboxes_data[img_path][self.selected_box_internal_idx]['box']) # Original coords
            self.dragging = True
        else:
            clicked_box_idx = self._find_box_at_pos(event.x, event.y)
            if clicked_box_idx is not None:
                self.selected_box_internal_idx = clicked_box_idx
                self.drag_type = 'move'
                img_path = self.image_paths[self.current_idx]
                box_coords_scaled = [c * self.scale for c in self.bboxes_data[img_path][self.selected_box_internal_idx]['box']]
                self.drag_offset = (event.x - box_coords_scaled[0], event.y - box_coords_scaled[1])
                self.drag_start_box = list(self.bboxes_data[img_path][self.selected_box_internal_idx]['box'])
                self.dragging = True
                self._draw_boxes() # To show selection
            else:
                self.selected_box_internal_idx = None
                self.drag_type = None
                self.dragging = False
                self._draw_boxes() # To deselect

    def _on_mouse_drag(self, event):
        if not self.dragging: return

        if self.drag_type == 'create':
            if self.temp_box_id: self.canvas.delete(self.temp_box_id)
            if self.new_box_start:
                self.temp_box_id = self.canvas.create_rectangle(self.new_box_start[0], self.new_box_start[1], event.x, event.y, outline=DET_TEMP_BOX_COLOR, width=2, tags='temp_box')
            return

        if self.selected_box_internal_idx is None or self.drag_type is None or not self.drag_start_box: return
        
        img_path = self.image_paths[self.current_idx]
        img_h_orig, img_w_orig = self.current_image_cv.shape[:2]
        current_box_orig = self.bboxes_data[img_path][self.selected_box_internal_idx]['box'] # direct ref

        dx_display, dy_display = event.x - self.drag_start[0], event.y - self.drag_start[1]
        dx_orig, dy_orig = dx_display / self.scale, dy_display / self.scale

        if self.drag_type == 'move':
            new_x1 = self.drag_start_box[0] + dx_orig
            new_y1 = self.drag_start_box[1] + dy_orig
            width_orig, height_orig = self.drag_start_box[2] - self.drag_start_box[0], self.drag_start_box[3] - self.drag_start_box[1]
            new_x2, new_y2 = new_x1 + width_orig, new_y1 + height_orig

            # Clamp to image bounds
            new_x1 = max(0.0, min(new_x1, img_w_orig - 1.0))
            new_y1 = max(0.0, min(new_y1, img_h_orig - 1.0))
            new_x2 = max(new_x1 + 1, min(new_x2, img_w_orig - 1.0))
            new_y2 = max(new_y1 + 1, min(new_y2, img_h_orig - 1.0))
            
            current_box_orig[0], current_box_orig[1], current_box_orig[2], current_box_orig[3] = new_x1, new_y1, new_x2, new_y2

        elif self.drag_type == 'resize':
            mouse_x_orig, mouse_y_orig = (event.x / self.scale), (event.y / self.scale)
            # Clamp mouse to image boundary for intuitive resizing near edges
            mouse_x_orig = max(0.0, min(mouse_x_orig, img_w_orig - 1.0))
            mouse_y_orig = max(0.0, min(mouse_y_orig, img_h_orig - 1.0))

            x1, y1, x2, y2 = self.drag_start_box # Use the state at drag start for consistent calcs

            if self.selected_corner == 'topleft': x1,y1 = mouse_x_orig, mouse_y_orig
            elif self.selected_corner == 'topright': x2,y1 = mouse_x_orig, mouse_y_orig
            elif self.selected_corner == 'bottomleft': x1,y2 = mouse_x_orig, mouse_y_orig
            elif self.selected_corner == 'bottomright': x2,y2 = mouse_x_orig, mouse_y_orig
            elif self.selected_corner == 'midleft': x1=mouse_x_orig
            elif self.selected_corner == 'midright': x2=mouse_x_orig
            elif self.selected_corner == 'midtop': y1=mouse_y_orig
            elif self.selected_corner == 'midbottom': y2=mouse_y_orig

            # Ensure x1 < x2 and y1 < y2, swap if necessary and update corner focus
            if x2 < x1: x1, x2 = x2, x1 # python swap
            if y2 < y1: y1, y2 = y2, y1
            
            current_box_orig[0], current_box_orig[1], current_box_orig[2], current_box_orig[3] = x1, y1, x2, y2

        self.modified = True
        self._draw_boxes()

    def _on_mouse_up(self, event):
        if self.dragging and self.drag_type == 'create' and self.new_box_start:
            if self.temp_box_id: 
                try: self.canvas.delete(self.temp_box_id)
                except tk.TclError: pass # Item might be gone
                self.temp_box_id = None

            x1_disp, y1_disp = min(self.new_box_start[0], event.x), min(self.new_box_start[1], event.y)
            x2_disp, y2_disp = max(self.new_box_start[0], event.x), max(self.new_box_start[1], event.y)

            x1_orig, y1_orig = x1_disp / self.scale, y1_disp / self.scale
            x2_orig, y2_orig = x2_disp / self.scale, y2_disp / self.scale
            
            img_h_orig, img_w_orig = self.current_image_cv.shape[:2]
            x1_orig = max(0.0, min(x1_orig, img_w_orig - 1.0))
            y1_orig = max(0.0, min(y1_orig, img_h_orig - 1.0))
            x2_orig = max(0.0, min(x2_orig, img_w_orig - 1.0))
            y2_orig = max(0.0, min(y2_orig, img_h_orig - 1.0))

            if abs(x2_orig - x1_orig) > 5 and abs(y2_orig - y1_orig) > 5: # Min box size
                img_path = self.image_paths[self.current_idx]
                new_box_data = {'box': [x1_orig, y1_orig, x2_orig, y2_orig], 'label': 'face', 'confidence': 1.0}
                self.bboxes_data[img_path].append(new_box_data)
                self.modified = True
                self.selected_box_internal_idx = len(self.bboxes_data[img_path]) - 1
                self._draw_boxes()
            else:
                print("New box too small, discarded.")
            self._toggle_new_box_mode(force_off=True) # Exit create mode

        self.dragging = False
        self.drag_type = None
        self.drag_start = None
        self.new_box_start = None
        self.drag_start_box = None 

    def _on_right_click(self, event):
        clicked_box_idx = self._find_box_at_pos(event.x, event.y)
        if clicked_box_idx is not None:
            img_path = self.image_paths[self.current_idx]
            self.bboxes_data[img_path].pop(clicked_box_idx)
            self.modified = True
            if clicked_box_idx == self.selected_box_internal_idx: self.selected_box_internal_idx = None
            elif self.selected_box_internal_idx is not None and clicked_box_idx < self.selected_box_internal_idx: self.selected_box_internal_idx -=1
            self._draw_boxes()

    def _prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.selected_box_internal_idx = None
            self._load_current_image()

    def _next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.selected_box_internal_idx = None
            self._load_current_image()

    def _complete(self):
        print("Detection Correction GUI complete.")
        self.root.quit()
        # self.root.destroy() # Might cause issues if quit already called

    def _delete_current_image(self):
        if not self.image_paths or not (0 <= self.current_idx < len(self.image_paths)): return
        current_path = self.image_paths[self.current_idx]
        print(f"Marking image for deletion: {current_path}")
        self.deleted_images.add(current_path)
        self.modified = True
        self.image_paths.pop(self.current_idx)
        # self.bboxes_data.pop(current_path, None) # Keep data, deletion set handles it

        if not self.image_paths: self._complete()
        elif self.current_idx >= len(self.image_paths): self.current_idx = len(self.image_paths) - 1
        # No change to current_idx if it's still valid
        self.selected_box_internal_idx = None
        self._load_current_image()

    def _toggle_new_box_mode(self, event=None, force_off=False):
        if force_off:
            self.creating_new_box = False
        else:
            self.creating_new_box = not self.creating_new_box
        
        if self.creating_new_box:
            self.canvas.config(cursor="cross")
            self.selected_box_internal_idx = None
            if self.temp_box_id: 
                try: self.canvas.delete(self.temp_box_id) 
                except tk.TclError: pass
                self.temp_box_id = None
            self.new_box_start = None
            print("New box creation mode: ON")
        else:
            self.canvas.config(cursor="")
            print("New box creation mode: OFF")
            if self.temp_box_id: 
                try: self.canvas.delete(self.temp_box_id)
                except tk.TclError: pass
                self.temp_box_id = None
            if self.drag_type == 'create': # Reset if creation was aborted by toggling mode
                self.dragging = False
                self.drag_type = None
        self._draw_boxes() # Redraw to update selection/cursor

    def run(self) -> Tuple[Dict[str, List[Dict]], Set[str]]:
        print("Starting Detection Correction GUI...")
        self.root.mainloop()
        try:
            if self.root.winfo_exists(): # Check if window still exists
                 self.root.destroy()
        except tk.TclError as e:
            print(f"Tkinter error during GUI destroy (might be normal on exit): {e}", file=sys.stderr)
            pass # Window might already be destroyed
        print("Detection Correction GUI finished.")
        return self.bboxes_data, self.deleted_images

# --- COCO Output Generation (Adapted from pseudolabel_gui.py) ---
class COCOFormatter:
    """Handles saving the final COCO JSON data."""
    def save_coco_json(self, coco_data: Dict, output_path_str: str):
        """
        Save the provided COCO dictionary to a JSON file.

        Args:
            coco_data (dict): The complete COCO dictionary to save.
            output_path_str (str): Path to save JSON file.
        """
        output_path = Path(output_path_str)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=4)
            print(f"COCO JSON successfully saved to: {output_path}")
        except Exception as e:
            print(f"ERROR: Failed to write COCO JSON file to {output_path}: {e}", file=sys.stderr)
            traceback.print_exc()

def prepare_detection_coco_data(
    image_bbox_dict: Dict[str, List[Dict]],
    deleted_image_paths: Set[str]
) -> Dict:
    """
    Prepares COCO JSON data structure for detections only.
    Assumes 'face' is the only category (ID 1).
    Expects image_bbox_dict keys to be absolute paths.
    """
    coco_data = {
        "info": {"description": "Video Frame Detections from YOLO via viddir2coco.py"},
        "licenses": [],
        "categories": [{
            "id": 1, 
            "name": "face", 
            "supercategory": "face"
        }],
        "images": [],
        "annotations": []
    }
    img_id_counter = 0
    ann_id_counter = 0

    # Sort image paths for consistent image IDs and processing order
    # Ensure keys are strings, as they might be Path objects if not careful upstream
    sorted_image_paths = sorted([str(p) for p in image_bbox_dict.keys()])

    for abs_frame_path_str in tqdm(sorted_image_paths, desc="Preparing Detection COCO Data"):
        if abs_frame_path_str in deleted_image_paths:
            # print(f"Skipping deleted image for COCO: {abs_frame_path_str}") # Optional: for debugging
            continue

        try:
            # It's crucial that abs_frame_path_str is a valid path to an image file
            # Reading the image just for dimensions can be slow if many files.
            # Consider if dimensions can be stored earlier if performance is an issue.
            # For now, KISS and read it.
            frame_image = cv2.imread(abs_frame_path_str)
            if frame_image is None:
                print(f"Warning: Could not read image {abs_frame_path_str} for COCO dimensions. Skipping.", file=sys.stderr)
                continue
            height, width = frame_image.shape[:2]
        except Exception as e:
            print(f"Error reading image {abs_frame_path_str} for dimensions: {e}. Skipping.", file=sys.stderr)
            continue
        
        coco_data["images"].append({
            "id": img_id_counter,
            "file_name": abs_frame_path_str, # Storing absolute path
            "height": height,
            "width": width
        })

        detections_for_image = image_bbox_dict.get(abs_frame_path_str, [])
        for det_idx, det in enumerate(detections_for_image):
            box = det.get('box') # Expected format: [x1, y1, x2, y2]
            if not box or len(box) != 4:
                print(f"Warning: Malformed box data {box} for image {abs_frame_path_str}, detection index {det_idx}. Skipping annotation.", file=sys.stderr)
                continue
            
            x1, y1, x2, y2 = map(float, box) # Ensure float for COCO
            coco_x = x1
            coco_y = y1
            coco_w = x2 - x1
            coco_h = y2 - y1

            if not (coco_x >= 0 and coco_y >= 0 and coco_w > 0 and coco_h > 0 and coco_x + coco_w <= width + 0.001 and coco_y + coco_h <= height + 0.001): # Add tolerance for float precision
                # print(f"Warning: Invalid box coords/dims {[coco_x, coco_y, coco_w, coco_h]} for image {abs_frame_path_str} (dims {width}x{height}). Box: {box}. Clamping/Skipping...", file=sys.stderr)
                # Option 1: Clamp (ensure values are within image bounds)
                coco_x = max(0.0, coco_x)
                coco_y = max(0.0, coco_y)
                coco_w = min(coco_w, width - coco_x)
                coco_h = min(coco_h, height - coco_y)
                if not (coco_w > 0 and coco_h > 0): # If clamping made it invalid, skip
                    # print(f"  Skipping annotation for {abs_frame_path_str} after clamping due to invalid dimensions: w={coco_w}, h={coco_h}")
                    continue
                # Option 2: Skip (as done by the original if condition basically)
                # continue

            coco_data["annotations"].append({
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "category_id": 1, # 'face' category_id is 1
                "bbox": [coco_x, coco_y, coco_w, coco_h],
                "area": coco_w * coco_h,
                "iscrowd": 0,
                "score": float(det.get('confidence', 1.0))
            })
            ann_id_counter += 1
        img_id_counter += 1
        
    print(f"Prepared COCO data with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")
    return coco_data

