"""Data processing utilities for Waldo detection."""

from typing import Dict, Iterator, Tuple
import jax.numpy as jnp
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
        if len(box.shape) == 1:  # Single box
            x1, y1, x2, y2 = box
            box = np.array([
                x1 * scale + pad_w,
                y1 * scale + pad_h,
                x2 * scale + pad_w,
                y2 * scale + pad_h
            ])
        else:  # Multiple boxes
            box[:, [0, 2]] = box[:, [0, 2]] * scale + pad_w
            box[:, [1, 3]] = box[:, [1, 3]] * scale + pad_h
        
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
            # Handle box flipping for both single box and multiple boxes
            if len(box.shape) == 1:  # Single box
                x1, y1, x2, y2 = box
                box = np.array([1 - x2, y1, 1 - x1, y2])
            else:  # Multiple boxes
                box[:, [0, 2]] = 1 - box[:, [2, 0]]  # Flip x coordinates
                
        if np.random.random() > 0.8:  # Less frequent vertical flip
            image = np.flipud(image)
            # Handle box flipping for both single box and multiple boxes
            if len(box.shape) == 1:  # Single box
                x1, y1, x2, y2 = box
                box = np.array([x1, 1 - y2, x2, 1 - y1])
            else:  # Multiple boxes
                box[:, [1, 3]] = 1 - box[:, [3, 1]]  # Flip y coordinates
            
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
            
        # Color augmentations using OpenCV
        if np.random.random() > 0.5:
            # Random brightness
            brightness = np.random.uniform(0.7, 1.3)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            
        if np.random.random() > 0.5:
            # Random contrast
            contrast = np.random.uniform(0.7, 1.3)
            mean = np.mean(image)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=(1-contrast)*mean)
            
        if np.random.random() > 0.5:
            # Random saturation
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = np.random.uniform(0.7, 1.3)
            image_hsv[:, :, 1] = cv2.convertScaleAbs(image_hsv[:, :, 1], alpha=saturation)
            image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
            
        # Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255)
        
        return np.array(image), box
    
    def _convert_to_center_size(self, box: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h] format."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        return np.array([cx, cy, w, h])
    
    def _prepare_sample(self, 
                       image_path: str,
                       boxes: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare a single sample with multiple boxes."""
        # Load and preprocess image
        image = self._load_image(image_path)
        
        # Handle multiple boxes
        processed_boxes = []
        for box in boxes:
            # Resize and normalize box coordinates
            _, box = self._resize_with_aspect_ratio(image, box)
            processed_boxes.append(box)
        
        # Stack boxes
        boxes = np.stack(processed_boxes)
        
        # Resize image once
        image, _ = self._resize_with_aspect_ratio(image, boxes[0])
        
        # Apply augmentations (will handle all boxes)
        image, boxes = self._apply_augmentations(image, boxes)
        
        # Convert boxes to center-size format
        center_size_boxes = np.array([self._convert_to_center_size(box) for box in boxes])
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return {
            'image': image,
            'boxes': center_size_boxes.astype(np.float32),
            'scores': np.ones(len(boxes), dtype=np.float32),  # Ground truth scores
        }
    
    def train_loader(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create training data loader with multi-box support."""
        while True:
            # Group annotations by image
            grouped = self.annotations.groupby('filename')
            # Shuffle image names
            image_names = list(grouped.groups.keys())
            np.random.shuffle(image_names)
            
            batch_images = []
            batch_boxes = []
            batch_scores = []
            
            for image_name in image_names:
                # Get all boxes for this image
                image_annots = grouped.get_group(image_name)
                boxes = image_annots[['xmin', 'ymin', 'xmax', 'ymax']].values
                
                # Prepare sample with all boxes
                sample = self._prepare_sample(image_name, boxes)
                
                batch_images.append(sample['image'])
                batch_boxes.append(sample['boxes'])
                batch_scores.append(sample['scores'])
                
                if len(batch_images) == self.batch_size:
                    # Pad boxes to same length within batch
                    max_boxes = max(boxes.shape[0] for boxes in batch_boxes)
                    padded_boxes = []
                    padded_scores = []
                    
                    for boxes, scores in zip(batch_boxes, batch_scores):
                        if boxes.shape[0] < max_boxes:
                            # Pad with zeros and set corresponding scores to 0
                            pad_boxes = np.zeros((max_boxes - boxes.shape[0], 4))
                            pad_scores = np.zeros(max_boxes - boxes.shape[0])  # Remove extra dimension
                            boxes = np.vstack([boxes, pad_boxes])
                            scores = np.concatenate([scores, pad_scores])  # Keep scores 1D
                        # No reshaping needed for scores
                        padded_boxes.append(boxes)
                        padded_scores.append(scores)
                    
                    yield {
                        'image': np.stack(batch_images),
                        'boxes': np.stack(padded_boxes),
                        'scores': np.stack(padded_scores),  # Shape: [batch_size, max_boxes]
                    }
                    batch_images = []
                    batch_boxes = []
                    batch_scores = []
    
    def val_loader(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create validation data loader with multi-box support."""
        # Group annotations by image
        grouped = self.annotations.groupby('filename')
        
        # Use a fixed subset of images for validation
        val_images = list(grouped.groups.keys())
        val_indices = np.linspace(
            0, len(val_images)-1,
            num=min(100, len(val_images)),
            dtype=int
        )
        val_images = [val_images[i] for i in val_indices]
        
        batch_images = []
        batch_boxes = []
        batch_scores = []
        
        for image_name in val_images:
            # Get all boxes for this image
            image_annots = grouped.get_group(image_name)
            boxes = image_annots[['xmin', 'ymin', 'xmax', 'ymax']].values
            
            # Prepare sample with all boxes
            sample = self._prepare_sample(image_name, boxes)
            
            batch_images.append(sample['image'])
            batch_boxes.append(sample['boxes'])
            batch_scores.append(sample['scores'])
            
            if len(batch_images) == self.batch_size:
                # Pad boxes to same length within batch
                max_boxes = max(boxes.shape[0] for boxes in batch_boxes)
                padded_boxes = []
                padded_scores = []
                
                for boxes, scores in zip(batch_boxes, batch_scores):
                    if boxes.shape[0] < max_boxes:
                        pad_boxes = np.zeros((max_boxes - boxes.shape[0], 4))
                        pad_scores = np.zeros(max_boxes - boxes.shape[0])  # Remove extra dimension
                        boxes = np.vstack([boxes, pad_boxes])
                        scores = np.concatenate([scores, pad_scores])  # Keep scores 1D
                    # No reshaping needed for scores
                    padded_boxes.append(boxes)
                    padded_scores.append(scores)
                
                yield {
                    'image': np.stack(batch_images),
                    'boxes': np.stack(padded_boxes),
                    'scores': np.stack(padded_scores),  # Shape: [batch_size, max_boxes]
                }
                batch_images = []
                batch_boxes = []
                batch_scores = []
        
        # Return remaining samples
        if batch_images:
            # Pad final batch
            max_boxes = max(boxes.shape[0] for boxes in batch_boxes)
            padded_boxes = []
            padded_scores = []
            
            for boxes, scores in zip(batch_boxes, batch_scores):
                if boxes.shape[0] < max_boxes:
                    pad_boxes = np.zeros((max_boxes - boxes.shape[0], 4))
                    pad_scores = np.zeros(max_boxes - boxes.shape[0])  # Remove extra dimension
                    boxes = np.vstack([boxes, pad_boxes])
                    scores = np.concatenate([scores, pad_scores])  # Keep scores 1D
                # No reshaping needed for scores
                padded_boxes.append(boxes)
                padded_scores.append(scores)
            
            yield {
                'image': np.stack(batch_images),
                'boxes': np.stack(padded_boxes),
                'scores': np.stack(padded_scores),  # Shape: [batch_size, max_boxes]
            }
