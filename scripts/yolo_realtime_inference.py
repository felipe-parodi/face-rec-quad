# yolo_realtime_inference.py

import cv2
from ultralytics import YOLO
from pathlib import Path
import time # Added for inference time calculation

# --- Global Configuration ---
CONFIDENCE_THRESHOLD = 0.5
MAX_DETECTIONS = 2 # New global variable for max detections
IOU_THRESHOLD = 0.4 # Lower values mean stricter NMS

def run_inference_on_video(model_path: str, video_path: str):
    """
    Performs real-time inference on a video file using a trained YOLO model.

    Args:
        model_path (str): Path to the trained YOLO model (e.g., best.pt).
        video_path (str): Path to the video file or camera index (e.g., 0 for default camera).
    """
    # 1. Load the YOLO model
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # 2. Open video capture
    try:
        # Check if video_path is an integer (camera index) or string (file path)
        if video_path.isdigit():
            video_source = int(video_path)
            print(f"Using camera index: {video_source}")
        else:
            video_source = video_path
            if not Path(video_source).exists():
                print(f"Error: Video file not found at {video_source}")
                return
            print(f"Processing video file: {video_source}")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
    except Exception as e:
        print(f"Error initializing video capture: {e}")
        return

    # Get video properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # 3. Loop through video frames
    window_name = "YOLO Real-Time Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allow resizing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Perform inference
        # Setting stream=True can be more efficient for continuous streams if memory/latency is an issue,
        # but for typical display, processing frame by frame is fine.
        # verbose=False to reduce console output from YOLO.
        try:
            start_time = time.perf_counter()
            results = model.predict(source=frame, save=False, verbose=False, 
                                    conf=CONFIDENCE_THRESHOLD, 
                                    iou=IOU_THRESHOLD, # Added IOU threshold for NMS
                                    max_det=MAX_DETECTIONS)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

        except Exception as e:
            print(f"Error during model prediction: {e}")
            continue # Skip this frame

        # results is a list (usually with one element for a single image/frame)
        # Each element is a Results object.
        if results and results[0]:
            # Use the .plot() method to get the frame with detections drawn
            processed_frame = results[0].plot() 
        else:
            # If no detections or error, show the original frame
            processed_frame = frame 

        # Display inference time on the frame
        cv2.putText(processed_frame,
                    f"Inference: {inference_time_ms:.2f} ms",
                    (10, 30), # Position (bottom-left corner from where text starts)
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, # Font scale
                    (0, 255, 0), # Color (B, G, R) - green
                    2) # Thickness

        # Display the processed frame
        cv2.imshow(window_name, processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
    
    # 4. Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished and resources released.")

if __name__ == '__main__':
    # --- Configuration ---
    MODEL_PATH = r"..\yolo_face_detection_workspace\training_runs\weights\best.pt"
    VIDEO_PATH = r"..\test_3min.mp4" 
    # To use a webcam, set VIDEO_PATH = "0" (or other camera index)

    # Global CONFIDENCE_THRESHOLD, MAX_DETECTIONS, and IOU_THRESHOLD are defined at the top of the script

    # Ensure paths are valid (optional, but good practice)
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
    elif VIDEO_PATH.isdigit() or Path(VIDEO_PATH).exists():
         run_inference_on_video(MODEL_PATH, VIDEO_PATH)
    else:
        print(f"Error: Video file not found at {VIDEO_PATH}")
