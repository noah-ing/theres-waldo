"""Prepare training data for stage 2 scene detection."""

import shutil
from pathlib import Path
import pandas as pd
import numpy as np
def train_val_split(df: pd.DataFrame, val_split: float, seed: int) -> tuple:
    """Split data into train and validation sets.
    
    Args:
        df: DataFrame to split
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df: Split DataFrames
    """
    # Set random seed
    np.random.seed(seed)
    
    # Get unique filenames and shuffle
    filenames = df['filename'].unique()
    np.random.shuffle(filenames)
    
    # Split indices
    split_idx = int(len(filenames) * (1 - val_split))
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]
    
    # Split dataframe
    train_df = df[df['filename'].isin(train_files)]
    val_df = df[df['filename'].isin(val_files)]
    
    return train_df, val_df

def prepare_stage2_data(
    annotations_file: str = 'annotations/annotations_filtered.csv',
    images_dir: str = 'images',
    output_dir: str = 'training_data/stage2',
    val_split: float = 0.2,
    seed: int = 42,
):
    """Prepare training data for stage 2 scene detection.
    
    Args:
        annotations_file: Path to filtered annotations CSV
        images_dir: Directory containing source images
        output_dir: Output directory for stage 2 data
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    
    # Read annotations
    df = pd.read_csv(annotations_file)
    
    # Split into train/val
    train_df, val_df = train_val_split(df, val_split, seed)
    
    # Create directory structure
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    for directory in [train_dir, val_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'images').mkdir(exist_ok=True)
    
    # Copy images and save annotations
    def process_split(split_df: pd.DataFrame, split_dir: Path):
        # Save annotations
        split_df.to_csv(split_dir / 'annotations.csv', index=False)
        
        # Copy images
        images_dir_path = Path(images_dir)
        for filename in split_df['filename'].unique():
            src_path = images_dir_path / filename
            dst_path = split_dir / 'images' / filename
            shutil.copy2(src_path, dst_path)
    
    # Process train split
    print("Processing training data...")
    process_split(train_df, train_dir)
    
    # Process validation split
    print("Processing validation data...")
    process_split(val_df, val_dir)
    
    print(f"Stage 2 data preparation complete:")
    print(f"- Training images: {len(train_df)}")
    print(f"- Validation images: {len(val_df)}")
    print(f"Data saved to: {output_dir}")

if __name__ == "__main__":
    prepare_stage2_data()
