"""Data processing utilities for Waldo detection."""

from typing import Dict, Iterator, Tuple
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
from pathlib import Path
import pandas as pd

class WaldoDataset:
    """Dataset handler for Waldo detection with modern augmentations."""
    
    def __init__(self, 
                 data_dir: str,
                 image_size: Tuple[int, int] = (640, 640),
                 batch_size: int = 16,
                 augment: bool = True):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing images and annotations
            image_size: Target image size (height, width)
            batch_size: Batch size for training
            augment: Whether to apply augmentations
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        
        # Load annotations from project root
        project_root = Path(self.data_dir).resolve()
        while not (project_root / 'annotations').exists() and project_root.parent != project_root:
            project_root = project_root.parent
        
        # Load annotations and filter for existing images
        self.project_root = project_root  # Store for later use
        self.annotations = pd.read_csv(
            project_root / 'annotations' / 'annotations.csv'
        )
        
        # Filter annotations to only include existing images
        existing_images = set(f.name for f in (project_root / 'images').iterdir() if f.is_file() and f.suffix.lower() == '.jpg')
        self.annotations = self.annotations[self.annotations['filename'].isin(existing_images)]
        
        if len(self.annotations) == 0:
            raise ValueError("No valid images found in dataset")
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        image = cv2.imread(str(self.project_root / 'images' / image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _resize_with_aspect_ratio(self, 
                                image: np.ndarray,
                                box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image while maintaining aspect ratio."""
        height, width = image.shape[:2]
        target_h, target_w = self.image_size
        
        # Calculate scaling factor
        scale = min(target_w/width, target_h/height)
        new_w, new_h = int(width * scale), int(height * scale)
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        image = np.pad(
            image,
            ((pad_h, target_h - new_h - pad_h),
             (pad_w, target_w - new_w - pad_w),
             (0, 0)),
            mode='constant'
        )
        
        # Adjust bounding box coordinates
        box = box.copy()
        box[[0, 2]] = box[[0, 2]] * scale + pad_w
        box[[1, 3]] = box[[1, 3]] * scale + pad_h
        
        # Normalize coordinates to [0, 1]
        box = box / np.array([target_w, target_h, target_w, target_h])
        
        return image, box
    
    def _apply_augmentations(self,
                           image: np.ndarray,
                           box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply advanced augmentation techniques for better generalization."""
        if not self.augment:
            return image, box
            
        # Random horizontal/vertical flips
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            box[[0, 2]] = 1 - box[[2, 0]]  # Flip x coordinates
        if np.random.random() > 0.8:  # Less frequent vertical flip
            image = np.flipud(image)
            box[[1, 3]] = 1 - box[[3, 1]]  # Flip y coordinates
            
        # Simple but effective augmentations
        if np.random.random() > 0.5:
            # Random zoom/scale with fixed output size
            scale = np.random.uniform(0.9, 1.1)
            h, w = image.shape[:2]
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(image, (scaled_w, scaled_h))
            
            # Resize back to original size
            image = cv2.resize(scaled, (w, h))
            # No need to adjust box coordinates since we maintain original size
            
        # Convert to tensor for color augmentations
        image = tf.convert_to_tensor(image)
        
        # Color augmentations with higher intensity
        image = tf.image.random_brightness(image, 0.3)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_hue(image, 0.15)
        
        # Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 10, image.shape)
            image = image + noise
            
        # Ensure valid image range
        image = tf.clip_by_value(image, 0, 255)
        
        return np.array(image), box
    
    def _prepare_sample(self, 
                       image_path: str,
                       box: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare a single sample."""
        # Load and preprocess image
        image = self._load_image(image_path)
        image, box = self._resize_with_aspect_ratio(image, box)
        
        # Apply augmentations
        image, box = self._apply_augmentations(image, box)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return {
            'image': image,
            'boxes': box.astype(np.float32),
            'scores': np.array([1.0], dtype=np.float32),  # Ground truth score
        }
    
    def train_loader(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create training data loader."""
        while True:
            # Shuffle annotations
            shuffled = self.annotations.sample(frac=1)
            
            batch_images = []
            batch_boxes = []
            batch_scores = []
            
            for _, row in shuffled.iterrows():
                sample = self._prepare_sample(
                    row['filename'],
                    np.array([
                        row['xmin'], row['ymin'],
                        row['xmax'], row['ymax']
                    ])
                )
                
                batch_images.append(sample['image'])
                batch_boxes.append(sample['boxes'])
                batch_scores.append(sample['scores'])
                
                if len(batch_images) == self.batch_size:
                    yield {
                        'image': np.stack(batch_images),
                        'boxes': np.stack(batch_boxes),
                        'scores': np.stack(batch_scores),
                    }
                    batch_images = []
                    batch_boxes = []
                    batch_scores = []
    
    def val_loader(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create validation data loader."""
        # Use a fixed subset for validation
        val_indices = np.linspace(
            0, len(self.annotations)-1, 
            num=min(100, len(self.annotations)),
            dtype=int
        )
        val_data = self.annotations.iloc[val_indices]
        
        batch_images = []
        batch_boxes = []
        batch_scores = []
        
        for _, row in val_data.iterrows():
            sample = self._prepare_sample(
                row['filename'],
                np.array([
                    row['xmin'], row['ymin'],
                    row['xmax'], row['ymax']
                ])
            )
            
            batch_images.append(sample['image'])
            batch_boxes.append(sample['boxes'])
            batch_scores.append(sample['scores'])
            
            if len(batch_images) == self.batch_size:
                yield {
                    'image': np.stack(batch_images),
                    'boxes': np.stack(batch_boxes),
                    'scores': np.stack(batch_scores),
                }
                batch_images = []
                batch_boxes = []
                batch_scores = []
        
        # Return remaining samples
        if batch_images:
            yield {
                'image': np.stack(batch_images),
                'boxes': np.stack(batch_boxes),
                'scores': np.stack(batch_scores),
            }

def create_tf_dataset(data_dir: str,
                     split: str = 'train',
                     batch_size: int = 16) -> tf.data.Dataset:
    """Create a TensorFlow dataset for compatibility with existing pipelines."""
    dataset = WaldoDataset(data_dir, batch_size=batch_size, augment=split=='train')
    loader = dataset.train_loader() if split == 'train' else dataset.val_loader()
    
    def generator():
        for batch in loader:
            yield batch
    
    return tf.data.Dataset.from_generator(
        generator,
        output_signature={
            'image': tf.TensorSpec(shape=(None, 640, 640, 3), dtype=tf.float32),
            'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'scores': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        }
    ).prefetch(tf.data.AUTOTUNE)
