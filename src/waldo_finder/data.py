"""Data processing utilities for Waldo detection."""

from typing import Dict, Iterator, Tuple, Optional
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
                 augment: bool = True,
                 use_unlabeled: bool = True,
                 box_format: str = 'cxcywh'):  # Use center format consistently
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
        
        # Find project root efficiently
        self.project_root = Path(data_dir).resolve()
        while not (self.project_root / 'annotations').exists() and self.project_root.parent != self.project_root:
            self.project_root = self.project_root.parent
            
        # Store parameters
        self.box_format = box_format
        self.use_unlabeled = use_unlabeled
        
        # Get all image paths first
        self.image_paths = list((self.project_root / 'images').glob('*.jpg'))
        
        # Stream annotations instead of loading all at once
        self.annotations_path = self.project_root / 'annotations' / 'annotations.csv'
        
        # Create index of labeled images
        self.labeled_images = set()
        if self.annotations_path.exists():
            for chunk in pd.read_csv(self.annotations_path, chunksize=1000):
                self.labeled_images.update(chunk['filename'].unique())
        
        if len(self.image_paths) == 0:
            raise ValueError("No valid images found in dataset")
        
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image, returning None if load fails."""
        try:
            image = cv2.imread(str(self.project_root / 'images' / image_path))
            if image is None:
                print(f"Warning: Failed to load image: {image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {str(e)}")
            return None
    
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
        """Apply enhanced augmentation techniques."""
        if not self.augment:
            return image, box
            
        height, width = image.shape[:2]
        
        # Convert box to absolute coordinates for augmentations
        abs_box = box.copy()
        abs_box = abs_box * np.array([width, height, width, height])
        
        # Random rotation (±7 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-7, 7)
            matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height), 
                                 borderMode=cv2.BORDER_REFLECT)
            
            # Rotate box corners
            if len(abs_box.shape) == 1:  # Single box
                x1, y1, x2, y2 = abs_box
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                ones = np.ones(shape=(len(corners), 1))
                corners_ones = np.hstack([corners, ones])
                rotated = corners_ones.dot(matrix.T)
                
                # Get new bounds
                x1 = np.min(rotated[:, 0])
                y1 = np.min(rotated[:, 1])
                x2 = np.max(rotated[:, 0])
                y2 = np.max(rotated[:, 1])
                
                abs_box = np.array([x1, y1, x2, y2])
        
        # Scale/zoom (0.9-1.1)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            new_w = int(width * scale)
            new_h = int(height * scale)
            image = cv2.resize(image, (new_w, new_h))
            
            # Adjust box for scale
            abs_box = abs_box * scale
            
            # Crop or pad to original size
            if scale > 1:  # Zoom in - crop
                x_start = (new_w - width) // 2
                y_start = (new_h - height) // 2
                image = image[y_start:y_start+height, x_start:x_start+width]
                abs_box -= np.array([x_start, y_start, x_start, y_start])
            else:  # Zoom out - pad
                x_start = (width - new_w) // 2
                y_start = (height - new_h) // 2
                temp = np.zeros((height, width, 3), dtype=image.dtype)
                temp[y_start:y_start+new_h, x_start:x_start+new_w] = image
                image = temp
                abs_box += np.array([x_start, y_start, x_start, y_start])
        
        # Horizontal flips
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            if len(abs_box.shape) == 1:  # Single box
                x1, y1, x2, y2 = abs_box
                abs_box = np.array([width - x2, y1, width - x1, y2])
            else:  # Multiple boxes
                abs_box[:, [0, 2]] = width - abs_box[:, [2, 0]]
        
        # Enhanced color jittering
        if np.random.random() > 0.5:
            # Brightness (0.9-1.1)
            brightness = np.random.uniform(0.9, 1.1)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        if np.random.random() > 0.5:
            # Contrast (0.9-1.1)
            contrast = np.random.uniform(0.9, 1.1)
            mean = np.mean(image)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=(1-contrast)*mean)
            
        if np.random.random() > 0.5:
            # Hue shift (±10°)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image[:, :, 0] = (image[:, :, 0] + np.random.uniform(-10, 10)) % 180
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        
        # Clip box coordinates to image bounds
        abs_box = np.clip(abs_box, 0, [width, height, width, height])
        
        # Convert box back to normalized coordinates
        box = abs_box / np.array([width, height, width, height])
        
        return np.array(image), box
    
    def _convert_box_format(self, box: np.ndarray, from_format: str, to_format: str) -> np.ndarray:
        """Convert between box formats efficiently."""
        if from_format == to_format:
            return box
            
        if from_format == 'xyxy' and to_format == 'cxcywh':
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            return np.array([x1 + w/2, y1 + h/2, w, h])
            
        if from_format == 'cxcywh' and to_format == 'xyxy':
            cx, cy, w, h = box
            return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            
        raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")
    
    def _prepare_sample(self,
                       image_path: str,
                       boxes: np.ndarray = None) -> Optional[Dict[str, np.ndarray]]:
        """Prepare a sample with optional boxes for semi-supervised learning."""
        # Load and preprocess image
        image = self._load_image(image_path)
        if image is None:
            return None
        
        # Initialize default box and score
        box = np.zeros((4,), dtype=np.float32)  # [cx, cy, w, h] format
        score = np.array([0.0], dtype=np.float32)
        
        if boxes is not None and len(boxes) > 0:
            # Convert first box to consistent format
            box = self._convert_box_format(boxes[0], 'xyxy', self.box_format)
            score = np.array([1.0], dtype=np.float32)
        
        # Resize and normalize coordinates
        image, box = self._resize_with_aspect_ratio(image, box)
        
        # Apply augmentations
        image, box = self._apply_augmentations(image, box)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return {
            'image': image,
            'boxes': box.astype(np.float32),
            'scores': score
        }
    
    def _get_boxes_for_image(self, image_name: str) -> Optional[np.ndarray]:
        """Stream and get boxes for an image efficiently."""
        try:
            boxes = []
            for chunk in pd.read_csv(self.annotations_path, chunksize=1000):
                img_annots = chunk[chunk['filename'] == image_name]
                if not img_annots.empty:
                    boxes.extend(img_annots[['xmin', 'ymin', 'xmax', 'ymax']].values)
            return np.array(boxes) if boxes else None
        except Exception as e:
            print(f"Warning: Error loading annotations for {image_name}: {str(e)}")
            return None

    def train_loader(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create training data loader with semi-supervised support."""
        while True:
            # Shuffle all image paths
            image_paths = self.image_paths.copy()
            np.random.shuffle(image_paths)
            
            batch_images = []
            batch_boxes = []
            batch_scores = []
            batch_is_labeled = []  # Track which samples are labeled
            
            for image_path in image_paths:
                image_name = image_path.name
                
                # Get boxes if image is labeled
                boxes = self._get_boxes_for_image(image_name) if image_name in self.labeled_images else None
                
                # Skip unlabeled data if not using it
                if boxes is None and not self.use_unlabeled:
                    continue
                    
                # Prepare sample
                sample = self._prepare_sample(str(image_path), boxes)
                if sample is None:  # Skip if image loading failed
                    continue
                
                batch_images.append(sample['image'])
                if boxes is not None:
                    batch_boxes.append(sample['boxes'])
                    batch_scores.append(sample['scores'])
                    batch_is_labeled.append(True)
                else:
                    # For unlabeled data, use dummy boxes and scores
                    batch_boxes.append(np.zeros((4,), dtype=np.float32))
                    batch_scores.append(np.array([0.0], dtype=np.float32))
                    batch_is_labeled.append(False)
                
                # Yield batch when full
                if len(batch_images) == self.batch_size:
                    yield {
                        'image': np.stack(batch_images),
                        'boxes': np.stack(batch_boxes),
                        'scores': np.stack(batch_scores),
                        'is_labeled': np.array(batch_is_labeled)
                    }
                    batch_images = []
                    batch_boxes = []
                    batch_scores = []
                    batch_is_labeled = []
            
            # Yield remaining samples
            if batch_images:
                yield {
                    'image': np.stack(batch_images),
                    'boxes': np.stack(batch_boxes),
                    'scores': np.stack(batch_scores),
                    'is_labeled': np.array(batch_is_labeled)
                }
    
    def val_loader(self) -> Iterator[Dict[str, np.ndarray]]:
        """Create validation data loader using only labeled data."""
        # Get validation images (20% of labeled images)
        val_images = list(self.labeled_images)
        np.random.seed(42)  # Fixed seed for consistent validation set
        val_size = max(int(len(val_images) * 0.2), 1)
        val_images = np.random.choice(val_images, size=val_size, replace=False)
        
        batch_images = []
        batch_boxes = []
        batch_scores = []
        
        for image_name in val_images:
            try:
                # Get boxes for this image
                boxes = self._get_boxes_for_image(image_name)
                if boxes is None:
                    continue
                    
                # Prepare sample
                sample = self._prepare_sample(str(self.project_root / 'images' / image_name), boxes)
                if sample is None:  # Skip if image loading failed
                    continue
                
                batch_images.append(sample['image'])
                batch_boxes.append(sample['boxes'])
                batch_scores.append(sample['scores'])
                
                # Yield batch when full
                if len(batch_images) == self.batch_size:
                    yield {
                        'image': np.stack(batch_images),
                        'boxes': np.stack(batch_boxes),
                        'scores': np.stack(batch_scores)
                    }
                    batch_images = []
                    batch_boxes = []
                    batch_scores = []
            except Exception as e:
                print(f"Warning: Error processing validation image {image_name}: {str(e)}")
                continue
        
        # Only yield if we have valid samples
        if len(batch_images) > 0:
            yield {
                'image': np.stack(batch_images),
                'boxes': np.stack(batch_boxes),
                'scores': np.stack(batch_scores)
            }
        else:
            print("Warning: No valid validation samples in batch")
