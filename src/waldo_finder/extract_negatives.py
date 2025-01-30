"""Extract negative examples (non-Waldo patches) from scene images."""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple, List
import random

def get_random_patch(
    image: np.ndarray,
    waldo_boxes: List[Tuple[int, int, int, int]],
    target_size: Tuple[int, int],
    min_distance: int = 20
) -> np.ndarray:
    """Extract a random patch that doesn't overlap with Waldo boxes."""
    height, width = image.shape[:2]
    patch_w, patch_h = target_size
    
    # Ensure patch size doesn't exceed image dimensions
    patch_w = min(patch_w, width)
    patch_h = min(patch_h, height)
    
    max_attempts = 100
    for _ in range(max_attempts):
        # Random top-left corner
        x = random.randint(0, width - patch_w)
        y = random.randint(0, height - patch_h)
        
        # Check distance from all Waldo boxes
        valid = True
        patch_box = (x, y, x + patch_w, y + patch_h)
        
        for waldo_box in waldo_boxes:
            if boxes_overlap(patch_box, waldo_box, min_distance):
                valid = False
                break
        
        if valid:
            return image[y:y + patch_h, x:x + patch_w]
    
    return None

def boxes_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    margin: int = 0
) -> bool:
    """Check if two boxes overlap (with margin)."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Add margin to second box
    x1_2 -= margin
    y1_2 -= margin
    x2_2 += margin
    y2_2 += margin
    
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

def extract_negative_examples(
    annotations_path: str,
    project_root: Path,
    negative_dir: Path,
    patches_per_image: int = 10,
    target_sizes: List[Tuple[int, int]] = [(30, 40), (40, 50), (50, 60)]
) -> None:
    """Extract negative examples from scene images."""
    # Read annotations
    df = pd.read_csv(annotations_path)
    
    print("\nExtracting negative examples:")
    negative_count = 0
    
    # Group boxes by image
    for filename in df['filename'].unique():
        try:
            # Load image
            img_path = project_root / 'images' / filename
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load {filename}")
                continue
            
            # Get all Waldo boxes for this image
            image_boxes = []
            image_df = df[df['filename'] == filename]
            for _, row in image_df.iterrows():
                box = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                image_boxes.append(box)
            
            # Extract random patches
            for i in range(patches_per_image):
                # Randomly choose a target size
                target_size = random.choice(target_sizes)
                
                patch = get_random_patch(image, image_boxes, target_size)
                if patch is not None:
                    # Save patch
                    patch_path = negative_dir / f"negative_{negative_count:03d}.jpg"
                    cv2.imwrite(str(patch_path), patch)
                    print(f"Extracted: {patch_path.name} ({patch.shape[1]}x{patch.shape[0]})")
                    negative_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return negative_count

def main():
    """Extract negative examples for stage 1 training."""
    project_root = Path.cwd()
    annotations_path = "annotations/annotations_filtered.csv"
    negative_dir = Path("training_data/stage1/negative")
    
    # Ensure negative directory exists
    negative_dir.mkdir(exist_ok=True, parents=True)
    
    print("Starting negative example extraction...")
    total_negatives = extract_negative_examples(
        annotations_path,
        project_root,
        negative_dir,
        patches_per_image=10  # Adjust to get desired number of negatives
    )
    
    print(f"\nExtraction complete!")
    print(f"Total negative examples: {total_negatives}")

if __name__ == "__main__":
    main()
