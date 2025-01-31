"""
Data preparation script for scene-level Waldo detection.
Converts CSV annotations to scene-level JSON format and creates curriculum splits.
"""

import pandas as pd
import json
from pathlib import Path
import shutil
import numpy as np
from typing import Dict, List, Tuple
import argparse

def load_annotations(csv_path: str) -> pd.DataFrame:
    """Load annotations from CSV file"""
    return pd.read_csv(csv_path)

def calculate_raw_metrics(scene_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate raw difficulty metrics for a scene"""
    # Get image dimensions
    width = scene_df['width'].iloc[0]
    height = scene_df['height'].iloc[0]
    image_size = width * height
    
    # Get Waldo box
    waldo_row = scene_df[scene_df['class'] == 'waldo'].iloc[0]
    waldo_width = waldo_row['xmax'] - waldo_row['xmin']
    waldo_height = waldo_row['ymax'] - waldo_row['ymin']
    waldo_size = waldo_width * waldo_height
    
    # Calculate relative size (smaller = harder)
    rel_size = waldo_size / image_size
    
    # Calculate edge distance (larger = harder)
    x_center = (waldo_row['xmin'] + waldo_row['xmax']) / 2
    y_center = (waldo_row['ymin'] + waldo_row['ymax']) / 2
    x_rel = x_center / width
    y_rel = y_center / height
    edge_dist = np.sqrt((x_rel - 0.5)**2 + (y_rel - 0.5)**2)
    
    # Calculate aspect ratio deviation (larger = harder)
    waldo_aspect = waldo_width / waldo_height
    aspect_dev = abs(np.log(waldo_aspect))
    
    return {
        'rel_size': rel_size,
        'edge_dist': edge_dist,
        'aspect_dev': aspect_dev
    }

def calculate_scene_complexity(scene_df: pd.DataFrame, metric_stats: Dict = None) -> float:
    """
    Calculate scene complexity using percentile-based ranking
    If metric_stats is None, return raw metrics for stats calculation
    """
    metrics = calculate_raw_metrics(scene_df)
    
    if metric_stats is None:
        return metrics
    
    # Calculate percentile scores (0-1, higher = harder)
    size_score = 1.0 - (
        (metrics['rel_size'] - metric_stats['rel_size_min']) /
        (metric_stats['rel_size_max'] - metric_stats['rel_size_min'])
    )
    edge_score = (
        (metrics['edge_dist'] - metric_stats['edge_dist_min']) /
        (metric_stats['edge_dist_max'] - metric_stats['edge_dist_min'])
    )
    aspect_score = (
        (metrics['aspect_dev'] - metric_stats['aspect_dev_min']) /
        (metric_stats['aspect_dev_max'] - metric_stats['aspect_dev_min'])
    )
    
    # Weighted combination (size most important)
    return 0.6 * size_score + 0.3 * edge_score + 0.1 * aspect_score

def create_scene_annotation(scene_df: pd.DataFrame, metric_stats: Dict = None) -> Dict:
    """Convert scene DataFrame to scene annotation format"""
    scene_id = scene_df['filename'].iloc[0].split('.')[0]
    
    # Calculate complexity
    complexity = calculate_scene_complexity(scene_df, metric_stats)
    
    # Create boxes
    boxes = []
    for _, row in scene_df.iterrows():
        box = {
            'category': row['class'],
            'x1': float(row['xmin']) / row['width'],
            'y1': float(row['ymin']) / row['height'],
            'x2': float(row['xmax']) / row['width'],
            'y2': float(row['ymax']) / row['height'],
            'context_score': complexity if isinstance(complexity, float) else 0.0
        }
        boxes.append(box)
    
    return {
        'id': scene_id,
        'filename': scene_df['filename'].iloc[0],
        'width': int(scene_df['width'].iloc[0]),
        'height': int(scene_df['height'].iloc[0]),
        'complexity': complexity if isinstance(complexity, float) else 0.0,
        'boxes': boxes
    }

def assign_curriculum_level(complexity: float) -> str:
    """
    Assign curriculum level using percentile thresholds:
    - Easy: Bottom third
    - Medium: Middle third
    - Hard: Top third
    """
    if complexity < 0.33:
        return 'easy'
    elif complexity < 0.67:
        return 'medium'
    else:
        return 'hard'

def calculate_metric_stats(scenes_data: List[Dict]) -> Dict:
    """Calculate min/max stats for each metric"""
    metrics = {
        'rel_size': [],
        'edge_dist': [],
        'aspect_dev': []
    }
    
    for metrics_dict in scenes_data:
        for key in metrics:
            metrics[key].append(metrics_dict[key])
    
    return {
        'rel_size_min': min(metrics['rel_size']),
        'rel_size_max': max(metrics['rel_size']),
        'edge_dist_min': min(metrics['edge_dist']),
        'edge_dist_max': max(metrics['edge_dist']),
        'aspect_dev_min': min(metrics['aspect_dev']),
        'aspect_dev_max': max(metrics['aspect_dev'])
    }

def prepare_scene_data(
    csv_path: str,
    data_dir: str,
    val_split: float = 0.2,
    seed: int = 42
) -> None:
    """
    Prepare scene-level data from CSV annotations
    
    Args:
        csv_path: Path to CSV annotations
        data_dir: Output data directory
        val_split: Validation split ratio
        seed: Random seed
    """
    # Set random seed
    np.random.seed(seed)
    
    # Load annotations
    df = load_annotations(csv_path)
    
    # First pass: collect raw metrics
    raw_metrics = []
    for filename, scene_df in df.groupby('filename'):
        metrics = calculate_scene_complexity(scene_df)  # Returns raw metrics when stats=None
        raw_metrics.append(metrics)
    
    # Calculate metric statistics
    metric_stats = calculate_metric_stats(raw_metrics)
    
    # Second pass: create scene annotations with normalized complexities
    scenes = []
    scene_complexities = {}
    for filename, scene_df in df.groupby('filename'):
        scene = create_scene_annotation(scene_df, metric_stats)
        scenes.append(scene)
        scene_complexities[scene['id']] = scene['complexity']
    
    # Sort scenes by complexity for better curriculum
    scenes.sort(key=lambda s: s['complexity'])
    
    # Split train/val ensuring balanced difficulty distribution
    num_train = int(len(scenes) * (1 - val_split))
    stride = len(scenes) // num_train
    train_indices = list(range(0, len(scenes), stride))[:num_train]
    train_scenes = [scenes[i] for i in train_indices]
    val_scenes = [s for i, s in enumerate(scenes) if i not in train_indices]
    
    # Create curriculum splits
    curriculum = {'easy': [], 'medium': [], 'hard': []}
    for scene in train_scenes:
        level = assign_curriculum_level(scene['complexity'])
        curriculum[level].append(scene['id'])
    
    # Create output directories
    data_dir = Path(data_dir)
    (data_dir / 'images').mkdir(parents=True, exist_ok=True)
    (data_dir / 'curriculum').mkdir(parents=True, exist_ok=True)
    
    # Save annotations
    with open(data_dir / 'train_annotations.json', 'w') as f:
        json.dump({'scenes': train_scenes}, f, indent=2)
    
    with open(data_dir / 'val_annotations.json', 'w') as f:
        json.dump({'scenes': val_scenes}, f, indent=2)
    
    # Save curriculum splits
    for level, ids in curriculum.items():
        with open(data_dir / 'curriculum' / f'{level}.json', 'w') as f:
            json.dump(ids, f, indent=2)
    
    print(f"Prepared {len(train_scenes)} training and {len(val_scenes)} validation scenes")
    print("\nCurriculum distribution:")
    for level, ids in curriculum.items():
        print(f"{level}: {len(ids)} scenes")

def copy_images(image_dir: str, data_dir: str) -> None:
    """Copy images to data directory"""
    src_dir = Path(image_dir)
    dst_dir = Path(data_dir) / 'images'
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in src_dir.glob('*.jpg'):
        shutil.copy2(img_path, dst_dir / img_path.name)

def main():
    parser = argparse.ArgumentParser(description="Prepare scene-level training data")
    parser.add_argument('--csv', default='annotations/annotations_filtered.csv',
                      help='Path to CSV annotations')
    parser.add_argument('--images', default='images',
                      help='Path to source images directory')
    parser.add_argument('--output', default='data/scenes',
                      help='Output data directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    print("Preparing scene-level data...")
    prepare_scene_data(
        csv_path=args.csv,
        data_dir=args.output,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print("\nCopying images...")
    copy_images(args.images, args.output)
    
    print("\nData preparation complete!")

if __name__ == '__main__':
    main()
