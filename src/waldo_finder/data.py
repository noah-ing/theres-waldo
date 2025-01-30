"""Data loading and preprocessing utilities."""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple
import cv2

def normalize_box(box: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Normalize box coordinates to [0, 1] range.
    
    Args:
        box: Array of [x1, y1, x2, y2] coordinates
        image_size: Tuple of (height, width)
        
    Returns:
        normalized_box: Box coordinates normalized to [0, 1] range
    """
    height, width = image_size
    x1, y1, x2, y2 = box
    
    return np.array([
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
    ])

def denormalize_box(box: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert normalized box coordinates back to pixel space.
    
    Args:
        box: Array of normalized [x1, y1, x2, y2] coordinates
        image_size: Tuple of (height, width)
        
    Returns:
        pixel_box: Box coordinates in pixel space
    """
    height, width = image_size
    x1, y1, x2, y2 = box
    
    return np.array([
        x1 * width,
        y1 * height,
        x2 * width,
        y2 * height,
    ]).astype(np.int32)

def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int],
    augment: bool = False
) -> np.ndarray:
    """Load and preprocess image for model input.
    
    Args:
        image_path: Path to image file
        target_size: Desired (height, width)
        augment: Whether to apply data augmentation
        
    Returns:
        image: Preprocessed image array
    """
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size[::-1])  # cv2 uses (width, height)
    
    if augment:
        # Random brightness
        beta = np.random.uniform(-0.2, 0.2) * 255
        image = cv2.add(image, beta)
        
        # Random contrast
        alpha = np.random.uniform(0.8, 1.2)
        image = cv2.multiply(image, alpha)
        
        # Clip values to valid range
        image = np.clip(image, 0, 255)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def create_scene_dataset(
    data_path: str,
    batch_size: int,
    train: bool = True,
    shuffle_buffer: int = 1000
) -> Iterator[dict]:
    """Create dataset iterator for scene detection training.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size
        train: Whether this is training data
        shuffle_buffer: Size of shuffle buffer for training
        
    Returns:
        dataset: Iterator yielding batches of data
    """
    data_path = Path(data_path).resolve()
    image_dir = data_path / 'images'
    annotation_file = data_path / 'annotations.csv'
    
    # Verify paths exist
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Read annotations
    annotations = []
    print(f"Loading annotations from {annotation_file}")
    with open(annotation_file) as f:
        next(f)  # Skip header
        for line in f:
            filename, width, height, class_name, xmin, ymin, xmax, ymax = line.strip().split(',')
            annotations.append({
                'image_id': filename.replace('.jpg', ''),
                'box': [float(xmin), float(ymin), float(xmax), float(ymax)],
            })
    
    def generator():
        indices = list(range(len(annotations)))
        if train:
            np.random.shuffle(indices)
            
        for idx in indices:
            ann = annotations[idx]
            image_path = image_dir / f"{ann['image_id']}.jpg"
            
            # Load and preprocess image
            image = load_and_preprocess_image(
                image_path,
                target_size=(512, 512),
                augment=train,
            )
            
            # Normalize box coordinates
            box = normalize_box(
                np.array(ann['box']),
                image_size=(512, 512),
            )
            
            yield {
                'image': image,
                'boxes': box,
                'confidence': np.array([1.0], dtype=np.float32),
            }
    
    # Create batches
    examples = list(generator())
    if not examples:
        raise ValueError(f"No examples found in {data_path}")
    
    print(f"Created dataset with {len(examples)} examples")
    if train:
        np.random.shuffle(examples)
    
    # Yield batches
    num_batches = (len(examples) + batch_size - 1) // batch_size
    print(f"Will yield {num_batches} batches of size {batch_size}")
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        yield {
            'image': np.stack([x['image'] for x in batch]),
            'boxes': np.stack([x['boxes'] for x in batch]),
            'confidence': np.stack([x['confidence'] for x in batch]),
        }

def create_binary_dataset(
    data_path: str,
    batch_size: int,
    train: bool = True,
    shuffle_buffer: int = 1000
) -> Iterator[dict]:
    """Create dataset iterator for binary classification training.
    
    Args:
        data_path: Path to data directory
        batch_size: Batch size
        train: Whether this is training data
        shuffle_buffer: Size of shuffle buffer for training
        
    Returns:
        dataset: Iterator yielding batches of data
    """
    data_path = Path(data_path)
    positive_dir = data_path / 'positive'
    negative_dir = data_path / 'negative'
    
    def load_class_images(class_dir: Path, label: int):
        for image_path in class_dir.glob('*.jpg'):
            image = load_and_preprocess_image(
                image_path,
                target_size=(128, 128),
                augment=train,
            )
            yield {
                'image': image,
                'label': np.array([label], dtype=np.float32),
            }
    
    def generator():
        # Load positive and negative examples
        positives = list(load_class_images(positive_dir, 1))
        negatives = list(load_class_images(negative_dir, 0))
        
        # Combine and shuffle
        examples = positives + negatives
        if train:
            np.random.shuffle(examples)
        
        for example in examples:
            yield example
    
    # Create batches
    examples = list(generator())
    if train:
        np.random.shuffle(examples)
    
    # Yield batches
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        yield {
            'image': np.stack([x['image'] for x in batch]),
            'label': np.stack([x['label'] for x in batch]),
        }
