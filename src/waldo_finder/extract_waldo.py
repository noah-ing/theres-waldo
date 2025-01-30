"""Extract Waldo crops from annotated images for pre-training."""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import shutil
from typing import Tuple

def create_training_dirs() -> Tuple[Path, Path, Path]:
    """Create directory structure for two-stage training."""
    # Create base directories
    base_dir = Path("training_data")
    base_dir.mkdir(exist_ok=True)
    
    # Stage 1: Waldo appearance learning
    stage1_dir = base_dir / "stage1"
    stage1_dir.mkdir(exist_ok=True)
    
    # Subdirectories for positive/negative examples
    waldo_dir = stage1_dir / "waldo"
    waldo_dir.mkdir(exist_ok=True)
    
    negative_dir = stage1_dir / "negative"
    negative_dir.mkdir(exist_ok=True)
    
    # Stage 2: Scene detection fine-tuning
    stage2_dir = base_dir / "stage2"
    stage2_dir.mkdir(exist_ok=True)
    
    return waldo_dir, negative_dir, stage2_dir

def extract_waldo_crops(
    annotations_path: str,
    project_root: Path,
    waldo_dir: Path,
    padding: float = 0.1  # Add 10% padding around boxes
) -> None:
    """Extract and save Waldo crops from annotated images."""
    # Read annotations
    df = pd.read_csv(annotations_path)
    
    print("\nExtracting Waldo crops:")
    for idx, row in df.iterrows():
        try:
            # Load image
            img_path = project_root / 'images' / row['filename']
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {row['filename']}")
                continue
            
            # Get box coordinates
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            
            # Calculate padding
            w = x2 - x1
            h = y2 - y1
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            # Add padding with boundary checks
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            # Save crop
            crop_path = waldo_dir / f"waldo_{idx:03d}.jpg"
            cv2.imwrite(str(crop_path), crop)
            
            print(f"Extracted: {crop_path.name} ({crop.shape[1]}x{crop.shape[0]})")
            
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
            continue

def prepare_stage2_data(
    annotations_path: str,
    project_root: Path,
    stage2_dir: Path
) -> None:
    """Prepare data for stage 2 fine-tuning."""
    # Copy annotations
    shutil.copy2(annotations_path, stage2_dir / "annotations.csv")
    
    # Create images directory
    images_dir = stage2_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Copy all scene images
    print("\nPreparing stage 2 data:")
    df = pd.read_csv(annotations_path)
    for filename in df['filename'].unique():
        try:
            src = project_root / 'images' / filename
            dst = images_dir / filename
            shutil.copy2(src, dst)
            print(f"Copied: {filename}")
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
            continue

def main():
    """Extract Waldo crops and prepare training data."""
    project_root = Path.cwd()
    annotations_path = "annotations/annotations_filtered.csv"
    
    print("Creating directory structure...")
    waldo_dir, negative_dir, stage2_dir = create_training_dirs()
    
    print("\nStarting data preparation:")
    print("1. Extracting Waldo crops for stage 1")
    extract_waldo_crops(annotations_path, project_root, waldo_dir)
    
    print("\n2. Preparing scene data for stage 2")
    prepare_stage2_data(annotations_path, project_root, stage2_dir)
    
    print("\nData preparation complete!")
    print(f"Stage 1 Waldo crops: {len(list(waldo_dir.glob('*.jpg')))} images")
    print("Stage 1 negative examples: Add manually")
    print(f"Stage 2 scenes: {len(list(stage2_dir.glob('images/*.jpg')))} images")
    
    print("\nNext steps:")
    print("1. Add negative examples to: training_data/stage1/negative/")
    print("2. Run stage 1 training")
    print("3. Run stage 2 fine-tuning")

if __name__ == "__main__":
    main()
