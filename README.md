# Monkey Face Detection and Recognition
Real-time face detection and recognition for macaque monkeys.

Scripts:
- `train_yolo_face_det.py`: Train a YOLO model for face detection.
- `yolo_realtime_inference.py`: Real-time face detection.

Data: See [gdrive](https://drive.google.com/drive/folders/1-hRzIeAdXGI_QNBKDt966Kx2Uz7G8xJN?usp=sharing) for sample images.

Next steps:
- Collect larger dataset for training.
- Label monkey IDs.
- Finetune detection (YOLO) and recognition (ArcFace).

## Setup

These instructions guide you through setting up a Conda environment for running the scripts, specifically using CPU for computations.

1.  **Create a new Conda environment:**
    Replace `face-rec-env` with your preferred environment name.
    ```bash
    conda create -n face-rec-env python=3.10
    conda activate face-rec-env
    ```

2.  **Install PyTorch (with or without GPU):**
    It's generally recommended to install PyTorch first, as Ultralytics depends on it. Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the latest command if needed, ensuring you select the CPU version for your OS.
    A common command for CPU-only on Windows/Linux is:
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    # Or using pip (ensure you are in the correct environment):
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

3.  **Install Ultralytics and other dependencies:**
    ```bash
    pip install ultralytics opencv-python Pillow tqdm
    ```

4.  **Verify installation (optional):**
    Open a Python interpreter in your activated Conda environment and try:
    ```python
    import torch
    import cv2
    from ultralytics import YOLO

    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    # Test loading a generic YOLO model (requires internet for first download)
    try:
        model = YOLO('yolov8n.pt') 
        print("Ultralytics YOLO loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
    ```

This setup should allow you to run the scripts that depend on Ultralytics YOLO models on your CPU.
