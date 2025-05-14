import json
import os
import shutil
from pathlib import Path
import random # For example plotting

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
import cv2

# --- Configuration ---
# User-defined paths
ORIGINAL_COCO_JSON_PATH = Path(r"A:\NonEnclosureProjects\inprep\PrimateFace\data\seb_faceid\det_labels.json")
IMAGE_SOURCE_DIR = Path(r"A:\NonEnclosureProjects\inprep\PrimateFace\data\seb_faceid\train_imgs") # Where original images are if JSON paths fail

# Script-managed paths
PROJECT_WORKSPACE_DIR = ORIGINAL_COCO_JSON_PATH.parent / "yolo_face_detection_workspace"

STAGING_DIR = PROJECT_WORKSPACE_DIR / "coco_staging"
STAGED_IMAGES_DIR = STAGING_DIR / "images"
STAGED_JSON_PATH = STAGING_DIR / "annotations.json" # Standard name for staged json

YOLO_CONVERSION_OUTPUT_DIR = PROJECT_WORKSPACE_DIR / "yolo_conversion_output" # Temp dir for convert_coco's direct output

FINAL_YOLO_DATASET_DIR = PROJECT_WORKSPACE_DIR / "final_yolo_dataset"
FINAL_YAML_PATH = FINAL_YOLO_DATASET_DIR / "dataset.yaml"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO is 1.0 - TRAIN_RATIO - VAL_RATIO

YOLO_MODEL_NAME = 'yolo11x.pt'
# --- Helper Functions ---

def prepare_staged_coco_data(original_json_path: Path, 
                             image_source_dir: Path, 
                             staged_images_output_dir: Path, 
                             staged_json_output_path: Path):
    """
    1. Reads the original COCO JSON.
    2. For each image:
        a. Checks if the absolute path in 'file_name' exists.
        b. If not, checks for os.path.basename(file_name) in image_source_dir.
        c. If a valid image source is found, copies the image to staged_images_output_dir using its basename.
        d. Updates the image entry in a new JSON structure to use only the basename for 'file_name'.
    3. Filters annotations to include only those for successfully staged images.
    4. Saves the new JSON (with basenames and filtered annotations/categories) to staged_json_output_path.
    Returns: List of category objects from COCO JSON (e.g., [{'id': 0, 'name': 'face', ...}]).
    """
    # Implementation details:
    # - Create output directories.
    # - Load original JSON.
    # - Iterate, validate paths, copy files, build new image list for JSON.
    # - Filter annotations.
    # - Save new JSON.
    # - Extract and return categories.
    print(f"Preparing staged COCO data from {original_json_path} to {staged_json_output_path}...")
    staged_images_output_dir.mkdir(parents=True, exist_ok=True)
    staged_json_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(original_json_path, 'r') as f:
        coco_data = json.load(f)

    new_images = []
    valid_image_ids = set()
    image_id_map = {} # To map old image IDs to new image IDs if necessary (though not strictly needed if we filter by file_name)

    for img_idx, image_info in enumerate(coco_data.get('images', [])):
        original_file_path_str = image_info['file_name']
        original_file_path = Path(original_file_path_str)
        base_name = original_file_path.name

        source_image_path = None
        if original_file_path.exists() and original_file_path.is_file():
            source_image_path = original_file_path
        else:
            potential_path_in_source_dir = image_source_dir / base_name
            if potential_path_in_source_dir.exists() and potential_path_in_source_dir.is_file():
                source_image_path = potential_path_in_source_dir
            else:
                print(f"Warning: Image not found at '{original_file_path_str}' or '{potential_path_in_source_dir}'. Skipping.")
                continue
        
        destination_image_path = staged_images_output_dir / base_name
        try:
            shutil.copy2(source_image_path, destination_image_path)
            new_image_info = image_info.copy()
            new_image_info['file_name'] = base_name # Use only basename
            new_images.append(new_image_info)
            valid_image_ids.add(image_info['id']) # Keep track of original IDs of copied images
        except Exception as e:
            print(f"Error copying image {source_image_path} to {destination_image_path}: {e}")
            continue

    if not new_images:
        print("Error: No images were successfully staged. Aborting.")
        return []

    new_annotations = []
    if 'annotations' in coco_data:
        for ann_info in coco_data['annotations']:
            if ann_info['image_id'] in valid_image_ids:
                new_annotations.append(ann_info)

    staged_coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_data.get('categories', []) # Keep original categories
    }

    with open(staged_json_output_path, 'w') as f:
        json.dump(staged_coco_data, f, indent=4)
    
    print(f"Successfully staged {len(new_images)} images and {len(new_annotations)} annotations.")
    return coco_data.get('categories', [])

def plot_coco_example(json_path: Path, images_dir: Path, num_examples: int = 1):
    """Plots random example images with their bounding boxes from a COCO JSON file."""
    print(f"Plotting COCO example from {json_path} with images from {images_dir}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    if not coco_data.get('images'):
        print("No images found in COCO JSON.")
        return

    # Create a mapping from image_id to image_info and annotations
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    image_id_to_annotations = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    # Select random image IDs to plot
    available_image_ids = list(image_id_to_info.keys())
    if not available_image_ids:
        print("No image IDs available to plot.")
        return
        
    num_to_plot = min(num_examples, len(available_image_ids))
    if num_to_plot == 0:
        print("No examples to plot.")
        return
        
    selected_image_ids = random.sample(available_image_ids, num_to_plot)

    # Determine subplot layout
    cols = int(num_to_plot**0.5)
    rows = (num_to_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if num_to_plot == 1:
        axes = [axes] # Make it iterable
    else:
        axes = axes.flatten()

    for i, img_id in enumerate(selected_image_ids):
        img_info = image_id_to_info[img_id]
        img_path = images_dir / img_info['file_name'] # Assumes file_name is basename

        if not img_path.exists():
            print(f"Image file not found: {img_path}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                axes[i].axis('off')
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            img_annotations = image_id_to_annotations.get(img_id, [])

            for ann in img_annotations:
                bbox = ann['bbox'] # [x, y, width, height]
                x, y, w, h = bbox
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                # Optionally, add category name if categories are present and mapped
                # cat_id = ann['category_id']
                # cat_name = coco_data['categories'][cat_id-1]['name'] # Example, needs proper mapping
                # draw.text((x, y), cat_name, fill="red")

            if i < len(axes):
                axes[i].imshow(image)
                axes[i].set_title(f"Image ID: {img_id}\n{img_info['file_name']}")
                axes[i].axis('off')
        except Exception as e:
            print(f"Error processing or plotting image {img_path}: {e}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, 'Error loading image', ha='center', va='center')
                axes[i].axis('off')

    # Hide any unused subplots
    for j in range(num_to_plot, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def split_and_organize_yolo_data(source_images_dir: Path, 
                                 source_labels_dir: Path, 
                                 destination_dataset_dir: Path,
                                 train_ratio: float, val_ratio: float):
    """
    1. Lists all .txt label files in source_labels_dir.
    2. Splits the list of label basenames (stems) into train, validation, and test sets.
    3. Creates the directory structure in destination_dataset_dir.
    4. For each label basename, finds the corresponding image in source_images_dir and copies both to the respective split directories.
    """
    print(f"Splitting and organizing YOLO data. Source images: {source_images_dir}, Source labels: {source_labels_dir}, Destination: {destination_dataset_dir}...")

    # Create destination directories
    for split in ['train', 'val', 'test']:
        (destination_dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (destination_dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 1. List available .txt label files and get their stems (basenames without .txt)
    if not source_labels_dir.exists():
        print(f"Error: Source labels directory does not exist: {source_labels_dir}. Aborting split.")
        return
        
    label_files = [f for f in source_labels_dir.iterdir() if f.is_file() and f.suffix.lower() == '.txt']
    if not label_files:
        print(f"No .txt label files found in {source_labels_dir}. Aborting split.")
        return
    
    label_basenames = [f.stem for f in label_files]
    print(f"Found {len(label_basenames)} label files in {source_labels_dir} to be split.")

    # Create a quick lookup for actual image file paths in source_images_dir based on stem
    # This helps find the image with its correct extension (jpg, png, etc.)
    image_stem_to_path_map = {img_f.stem: img_f for img_f in source_images_dir.iterdir() 
                              if img_f.is_file() and img_f.suffix.lower() in ['.png', '.jpg', '.jpeg']}

    # Filter label_basenames to include only those that have a corresponding image in source_images_dir
    available_basenames = [bn for bn in label_basenames if bn in image_stem_to_path_map]
    if not available_basenames:
        print(f"No label files in {source_labels_dir} correspond to any images in {source_images_dir}. Aborting split.")
        return
    if len(available_basenames) < len(label_basenames):
        print(f"Warning: {len(label_basenames) - len(available_basenames)} label files did not have a matching image in {source_images_dir} and will be excluded.")

    # Calculate split sizes based on *available* data
    num_available = len(available_basenames)
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 <= test_ratio < 1 and (train_ratio + val_ratio + test_ratio) <= 1.00001 ):
        print(f"Invalid train/val/test ratios: {train_ratio}/{val_ratio}/{test_ratio}. Sum must be <= 1. Aborting split.")
        return

    # Split: first into train and temp (val + test)
    train_basenames, temp_basenames = train_test_split(
        available_basenames, 
        train_size=train_ratio, 
        shuffle=True, 
        random_state=42 # for reproducibility
    )

    val_basenames, test_basenames = [], []
    if temp_basenames: 
        if test_ratio > 1e-9 and (val_ratio + test_ratio) > 1e-9: 
            val_proportion_in_temp = val_ratio / (val_ratio + test_ratio)
            if val_proportion_in_temp >= 1.0: # e.g. test_ratio is 0 or very small
                val_basenames = temp_basenames
            elif val_proportion_in_temp <= 1e-9: # e.g. val_ratio is 0 or very small
                test_basenames = temp_basenames
            else:
                val_basenames, test_basenames = train_test_split(
                    temp_basenames, 
                    train_size=val_proportion_in_temp, 
                    shuffle=True, 
                    random_state=42
                )
        else: # If test_ratio is effectively zero, all temp goes to val
            val_basenames = temp_basenames
    
    print(f"Dataset split based on available labels: {len(train_basenames)} train, {len(val_basenames)} val, {len(test_basenames)} test items.")

    def copy_files(split_name, basenames_list_for_split):
        img_dest_dir = destination_dataset_dir / 'images' / split_name
        lbl_dest_dir = destination_dataset_dir / 'labels' / split_name
        copied_count = 0
        for bn in basenames_list_for_split:
            label_file_path = source_labels_dir / f"{bn}.txt"
            image_file_path = image_stem_to_path_map.get(bn) # Get from map

            if image_file_path and image_file_path.exists() and label_file_path.exists():
                try:
                    shutil.copy2(image_file_path, img_dest_dir / image_file_path.name)
                    shutil.copy2(label_file_path, lbl_dest_dir / label_file_path.name)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying files for basename {bn} to {split_name}: {e}")
            else:
                if not (image_file_path and image_file_path.exists()):
                    print(f"Warning: Image file for basename '{bn}' (expected at '{image_file_path}') not found. Skipping for '{split_name}'.")
                if not label_file_path.exists():
                    # This should ideally not happen if basenames_list_for_split comes from discovered label files
                    print(f"Warning: Label file {label_file_path} does not exist. Skipping for '{split_name}'.")
        print(f"Copied {copied_count} image/label pairs to {split_name}.")

    copy_files('train', train_basenames)
    copy_files('val', val_basenames)
    copy_files('test', test_basenames)
    
    print("Data splitting and organization complete.")

def create_yolo_dataset_yaml(yaml_path: Path, dataset_dir: Path, class_names_list: list):
    """
    Creates the dataset.yaml file for YOLO training.
    Paths in YAML are relative to the YAML file's directory.
    """
    print(f"Creating YOLO dataset YAML at {yaml_path} for dataset at {dataset_dir}...")
    
    # Ensure the parent directory for the YAML file exists
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Paths in the YAML should be relative to dataset_dir (which is yaml_path.parent)
    # Ultralytics standard is to have paths relative to the yaml file itself.
    # So if yaml is at FINAL_YOLO_DATASET_DIR/dataset.yaml
    # train path will be ./images/train

    content = f"""
path: {yaml_path.parent.resolve()}  # Root directory of the dataset (absolute path)
# Train, val, test images directories (relative to 'path')
train: ./images/train
val: ./images/val
test: ./images/test

# Classes
names:
"""
    # Add class names
    for i, name in enumerate(class_names_list):
        content += f"  {i}: {name}\n"

    try:
        with open(yaml_path, 'w') as f:
            f.write(content)
        print(f"Successfully created dataset YAML: {yaml_path}")
        print("--- YAML Content ---")
        print(content)
        print("--------------------")
    except Exception as e:
        print(f"Error creating YAML file {yaml_path}: {e}")

def plot_inference_results_grid(results, class_names: list, num_images_to_plot: int = 9, save_path: Path = None):
    """Plots a grid of images with detected bounding boxes from YOLO results and optionally saves it."""
    print(f"Plotting inference results grid for up to {num_images_to_plot} images...")

    if not results:
        print("No results to plot.")
        return

    num_results = len(results)
    plot_count = min(num_images_to_plot, num_results)

    if plot_count == 0:
        print("No results to display in grid.")
        return

    cols = int(plot_count**0.5)
    rows = (plot_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if plot_count == 1:
        axes = [axes] # Make it iterable
    else:
        axes = axes.flatten()

    for i in range(plot_count):
        result = results[i]
        try:
            # The .plot() method of a Results object returns a BGR numpy array with detections plotted
            img_with_boxes = result.plot() # This returns a numpy array (BGR by default)
            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib
            
            if i < len(axes):
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"Result {i+1}") # You can customize title, e.g., with original path if available
                axes[i].axis('off')
        except Exception as e:
            print(f"Error plotting result {i}: {e}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, 'Error plotting', ha='center', va='center')
                axes[i].axis('off')

    # Hide any unused subplots
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            print(f"Inference grid saved to {save_path}")
        except Exception as e:
            print(f"Error saving inference grid to {save_path}: {e}")
    plt.show()

# --- Main Script Logic ---
def main():
    # 0. Create workspace directories
    PROJECT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    STAGED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_CONVERSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load COCO JSON, preprocess paths, copy images to a staging area.
    print("Step 1: Preprocessing COCO JSON and staging images...")
    categories = prepare_staged_coco_data(ORIGINAL_COCO_JSON_PATH, IMAGE_SOURCE_DIR, 
                                          STAGED_IMAGES_DIR, STAGED_JSON_PATH)
    if not categories:
        print("Error: Could not extract categories or no data was processed. Exiting.")
        return
    # Assuming single class or categories are correctly ordered by convert_coco later
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])] 
    if not class_names:
        print("Warning: No class names extracted from COCO categories. Using default 'face'.")
        class_names = ['face'] # Fallback if categories were empty or malformed
    print(f"Detected class names: {class_names}")

    # 2. Plot an example image and annotations from the staged COCO data.
    print("\nStep 2: Plotting an example from staged COCO data...")
    plot_coco_example(STAGED_JSON_PATH, STAGED_IMAGES_DIR, num_examples=2) # Plot 2 examples

    # 3. Convert staged COCO data to YOLO format.
    print("\nStep 3: Converting staged COCO to YOLO format...")
    # convert_coco expects labels_dir to be the direct path to the JSON file.
    # It saves labels to save_dir/labels/<json_stem> and YAML to save_dir.parent/save_dir.name.yaml.
    # We will use our own YAML creation and splitting logic.

    # Ensure YOLO_CONVERSION_OUTPUT_DIR is clean to prevent Ultralytics from creating '...output2', etc.
    # We do this to encourage it to use the base name if possible, but we will find the actual dir later.
    if YOLO_CONVERSION_OUTPUT_DIR.exists():
        print(f"Cleaning up intended YOLO conversion output directory: {YOLO_CONVERSION_OUTPUT_DIR}")
        shutil.rmtree(YOLO_CONVERSION_OUTPUT_DIR)
    # We don't strictly need to recreate it here if we are finding the latest below,
    # but it doesn't hurt if convert_coco prefers the base name when available.
    YOLO_CONVERSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Call convert_coco. We will determine the actual output path afterwards.
    # labels_dir should be the directory containing the COCO JSON file(s).
    # STAGING_DIR is yolo_face_detection_workspace/coco_staging/
    # STAGED_JSON_PATH is yolo_face_detection_workspace/coco_staging/annotations.json
    convert_coco(labels_dir=str(STAGING_DIR),  # Pass the directory containing the JSON
                 save_dir=str(YOLO_CONVERSION_OUTPUT_DIR), # Still provide the intended base save_dir
                 use_segments=False,
                 use_keypoints=False,
                 cls91to80=False)
    
    # Find the actual output directory created by convert_coco
    # It often creates suffixed directories like 'yolo_conversion_output2', etc.
    # We look in the parent of our intended YOLO_CONVERSION_OUTPUT_DIR.
    search_dir_for_conversion_outputs = YOLO_CONVERSION_OUTPUT_DIR.parent
    base_output_name = YOLO_CONVERSION_OUTPUT_DIR.name # e.g., "yolo_conversion_output"

    possible_output_dirs = []
    for d in search_dir_for_conversion_outputs.iterdir():
        if d.is_dir() and d.name.startswith(base_output_name):
            possible_output_dirs.append(d)

    if not possible_output_dirs:
        print(f"Error: No output directory starting with '{base_output_name}' found in {search_dir_for_conversion_outputs}. Cannot locate convert_coco results.")
        return

    # Sort by modification time, most recent first
    possible_output_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    actual_conversion_output_basedir = possible_output_dirs[0]
    print(f"Dynamically identified convert_coco output directory: {actual_conversion_output_basedir}")
    
    # The labels are inside a 'labels' subfolder, and then a subfolder named after the original JSON stem.
    converted_labels_source_dir = actual_conversion_output_basedir / "labels" / STAGED_JSON_PATH.stem

    print(f"Expecting labels in: {converted_labels_source_dir}")

    if not converted_labels_source_dir.exists() or not any(converted_labels_source_dir.iterdir()):
        # Fallback: What if the structure is simpler, e.g. actual_conversion_output_basedir / "labels" ?
        simple_labels_path = actual_conversion_output_basedir / "labels"
        if simple_labels_path.exists() and any(f.suffix == '.txt' for f in simple_labels_path.iterdir()):
            print(f"Warning: Labels found directly in {simple_labels_path}, not in a '{STAGED_JSON_PATH.stem}' subfolder within it. Using this path.")
            converted_labels_source_dir = simple_labels_path
        else:
            print(f"Error: YOLO labels not found or empty in dynamically identified primary directory: {converted_labels_source_dir} "
                  f"or its fallback: {simple_labels_path}. "
                  f"Please check the structure within '{actual_conversion_output_basedir}'.")
            return
    
    print(f"YOLO labels source directory for split: {converted_labels_source_dir}")

    # 3.5. Split images and YOLO labels into train/val/test sets.
    print("\nStep 3.5: Splitting data into train/val/test sets...")
    split_and_organize_yolo_data(STAGED_IMAGES_DIR, converted_labels_source_dir,
                                 FINAL_YOLO_DATASET_DIR, TRAIN_RATIO, VAL_RATIO)

    # Create the dataset.yaml for the final structured dataset.
    create_yolo_dataset_yaml(FINAL_YAML_PATH, FINAL_YOLO_DATASET_DIR, class_names)

    # 4. Load YOLO model.
    print(f"\nStep 4: Loading YOLO model ({YOLO_MODEL_NAME})...")
    try:
        model = YOLO(YOLO_MODEL_NAME)
    except Exception as e:
        print(f"Error loading model {YOLO_MODEL_NAME}: {e}")
        print("Ensure the model name is correct and Ultralytics can download/access it.")
        return

    # 5. Train YOLO model.
    print("\nStep 5: Training YOLO model...")
    training_results_dir = PROJECT_WORKSPACE_DIR / "training_runs"
    try:
        model.train(data=str(FINAL_YAML_PATH), 
                    epochs=50, # Adjust as needed
                    batch=16,  # Adjust based on GPU memory, e.g., 8, 16, 32
                    imgsz=640, 
                    project=str(training_results_dir.parent), # Project is parent of name
                    name=training_results_dir.name, # Name is the experiment folder
                    exist_ok=True # Allow overwriting if rerunning
                    )
        # Path to best.pt is typically <project>/<name>/weights/best.pt
        best_model_path = training_results_dir / "weights" / "best.pt"
        if not best_model_path.exists(): # Ultralytics sometimes creates an exp folder like `train` or `train2` if name clashes
            # Try to find the latest run if explicit path failed
            run_folders = sorted([d for d in training_results_dir.parent.iterdir() if d.is_dir() and d.name.startswith(training_results_dir.name)], key=os.path.getmtime, reverse=True)
            if run_folders:
                potential_best_path = run_folders[0] / "weights" / "best.pt"
                if potential_best_path.exists():
                    best_model_path = potential_best_path
                else:
                     print(f"Could not find best.pt in latest run folder: {run_folders[0]}") # Check if train created a new folder e.g. training_runs2 etc.
                     # Fallback: search inside training_results_dir for any `best.pt`
                     found_paths = list(training_results_dir.rglob("best.pt"))
                     if found_paths:
                        best_model_path = found_paths[0]
                        print(f"Found best.pt at {best_model_path}")
                     else:
                        print(f"Warning: best.pt not found in expected training output directory: {training_results_dir / 'weights'}")
                        best_model_path = None # Set to None if not found
            else:
                print(f"Warning: No training run folder found like {training_results_dir.name} in {training_results_dir.parent}")
                best_model_path = None
        print(f"Best model path: {best_model_path}")

    except Exception as e:
        print(f"Error during model training: {e}")
        best_model_path = None # Ensure it's None if training fails

    # 6. Evaluate the model on the test set.
    print("\nStep 6: Evaluating model...")
    if best_model_path and best_model_path.exists():
        try:
            eval_model = YOLO(best_model_path)
            evals_base_dir = PROJECT_WORKSPACE_DIR / "evals"
            
            # Evaluate on test set
            test_set_images_exist = (FINAL_YOLO_DATASET_DIR / "images" / "test").exists() and \
                                    any((FINAL_YOLO_DATASET_DIR / "images" / "test").iterdir())
            
            if test_set_images_exist:
                test_eval_output_dir = evals_base_dir / "test_set_results"
                test_eval_output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Saving detailed test set evaluation results to: {test_eval_output_dir}")
                metrics_test = eval_model.val(data=str(FINAL_YAML_PATH), 
                                            split='test', 
                                            project=str(test_eval_output_dir.parent), 
                                            name=str(test_eval_output_dir.name),
                                            exist_ok=True)
                print("Evaluation metrics (test set):", metrics_test.results_dict)
                summary_metrics_path_test = test_eval_output_dir / "summary_metrics.json"
                with open(summary_metrics_path_test, 'w') as f:
                    json.dump(metrics_test.results_dict, f, indent=4)
                print(f"Test set summary metrics saved to {summary_metrics_path_test}")
            else:
                print("Test set not found or empty, skipping evaluation on test split.")

            # Evaluate on validation set if test was skipped or as additional info
            val_set_images_exist = (FINAL_YOLO_DATASET_DIR / "images" / "val").exists() and \
                                 any((FINAL_YOLO_DATASET_DIR / "images" / "val").iterdir())

            if not test_set_images_exist and val_set_images_exist:
                print("Evaluating on validation set as test set was not available.")
                val_eval_output_dir = evals_base_dir / "validation_set_results"
                val_eval_output_dir.mkdir(parents=True, exist_ok=True)
                print(f"Saving detailed validation set evaluation results to: {val_eval_output_dir}")
                metrics_val = eval_model.val(data=str(FINAL_YAML_PATH), 
                                           split='val', 
                                           project=str(val_eval_output_dir.parent), 
                                           name=str(val_eval_output_dir.name),
                                           exist_ok=True)
                print("Evaluation metrics (validation set):", metrics_val.results_dict)
                summary_metrics_path_val = val_eval_output_dir / "summary_metrics.json"
                with open(summary_metrics_path_val, 'w') as f:
                    json.dump(metrics_val.results_dict, f, indent=4)
                print(f"Validation set summary metrics saved to {summary_metrics_path_val}")
            elif not val_set_images_exist and not test_set_images_exist:
                 print("Neither test nor validation set found or empty for evaluation.")

        except Exception as e:
            print(f"Error during model evaluation: {e}")
    else:
        print(f"Could not find best model at '{best_model_path}' (or training failed). Skipping evaluation.")

    # 7. Inference on test set --> plot as grid.
    print("\nStep 7: Running inference on test set and plotting results...")
    if best_model_path and best_model_path.exists():
        try:
            inference_model = YOLO(best_model_path)
            test_images_input_dir = FINAL_YOLO_DATASET_DIR / "images" / "test"
            if test_images_input_dir.exists() and any(test_images_input_dir.iterdir()):
                test_image_files = [str(f) for f in test_images_input_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                if test_image_files:
                    results = inference_model.predict(source=test_image_files, save=False, conf=0.25)
                    inference_grid_save_path = PROJECT_WORKSPACE_DIR / "inference_results_grid.png"
                    plot_inference_results_grid(results, class_names, num_images_to_plot=9, save_path=inference_grid_save_path)
                else:
                    print("No images found in the test images directory for inference.")
            else:
                print(f"Test images directory not found or empty: {test_images_input_dir}. Skipping inference plotting.")
        except Exception as e:
            print(f"Error during inference or plotting: {e}")
    else:
        print(f"Could not find best model at '{best_model_path}' (or training failed). Skipping inference.")

    print("\nScript finished.")

if __name__ == '__main__':
    main()