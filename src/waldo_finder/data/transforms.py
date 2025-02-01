"""
Data augmentation transforms for Waldo detection training.
"""

import torch
import torchvision.transforms as T
from typing import Dict, Optional
import random
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def convert_to_rgb(image):
    """Convert image to RGB format if needed.
    
    Args:
        image (PIL.Image): Input image in any format (RGBA, CMYK, etc.)
        
    Returns:
        PIL.Image: Image in RGB format
    """
    original_mode = image.mode
    if original_mode == 'RGBA':
        logger.debug(f"Converting RGBA image to RGB using alpha compositing")
        # Convert RGBA to RGB by compositing on white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        return background
    elif original_mode == 'CMYK':
        logger.debug(f"Converting CMYK image to RGB")
        # Convert CMYK to RGB with proper color profile handling
        try:
            # First try converting with any embedded ICC profile
            return image.convert('RGB', dither=None)
        except Exception as e:
            logger.warning(f"ICC profile conversion failed: {str(e)}, using basic conversion")
            try:
                # Try PIL's built-in conversion first
                return image.convert('RGB')
            except Exception as e:
                logger.warning(f"Built-in conversion failed: {str(e)}, using numpy conversion")
                # Last resort: manual conversion via numpy
                # Convert to numpy array and invert CMYK values
                cmyk_array = np.array(image)
                if len(cmyk_array.shape) == 3 and cmyk_array.shape[2] == 4:
                    # Proper CMYK array
                    c, m, y, k = cmyk_array[:, :, 0], cmyk_array[:, :, 1], cmyk_array[:, :, 2], cmyk_array[:, :, 3]
                    # Convert to RGB using standard CMYK->RGB formula
                    r = 255 * (1 - c/255) * (1 - k/255)
                    g = 255 * (1 - m/255) * (1 - k/255)
                    b = 255 * (1 - y/255) * (1 - k/255)
                    rgb_array = np.stack([r, g, b], axis=2).astype(np.uint8)
                    return Image.fromarray(rgb_array, 'RGB')
                else:
                    # Fallback for unexpected array shape
                    return Image.fromarray(255 - cmyk_array).convert('RGB')
    elif original_mode != 'RGB':
        # Handle any other modes (L, P, etc.)
        logger.debug(f"Converting {original_mode} image to RGB")
        return image.convert('RGB')
    
    logger.debug(f"Image already in RGB mode, no conversion needed")
    return image

class WaldoTransforms:
    """Configurable transforms for Waldo detection data augmentation"""
    
    def __init__(
        self,
        config: Dict,
        is_train: bool = True,
        img_size: int = 384
    ):
        self.config = config
        self.is_train = is_train
        self.img_size = img_size
        
        # Build transforms
        if is_train:
            self.transform = self._build_train_transforms()
        else:
            self.transform = self._build_val_transforms()
            
    def _build_train_transforms(self) -> T.Compose:
        """Build training transforms with augmentation"""
        aug_config = self.config['augmentation']
        
        transforms = [
            # Convert to RGB first
            T.Lambda(convert_to_rgb),
            
            # Resize with aspect ratio preservation
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            
            # Color augmentation
            T.ColorJitter(
                brightness=aug_config['color_jitter']['brightness'],
                contrast=aug_config['color_jitter']['contrast'],
                saturation=aug_config['color_jitter']['saturation'],
                hue=aug_config['color_jitter']['hue']
            ),
            
            # Geometric augmentation
            T.RandomAffine(
                degrees=aug_config['random_affine']['degrees'],
                translate=aug_config['random_affine']['translate'],
                scale=aug_config['random_affine']['scale']
            ),
            
            # Flips
            T.RandomHorizontalFlip(p=aug_config['random_horizontal_flip']),
            T.RandomVerticalFlip(p=aug_config['random_vertical_flip']),
            
            # Additional augmentations for robustness
            T.RandomApply([
                T.GaussianBlur(kernel_size=3)
            ], p=0.3),
            
            T.RandomApply([
                T.RandomPerspective(distortion_scale=0.3)
            ], p=0.3),
            
            # Convert to tensor and normalize
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        return T.Compose(transforms)
        
    def _build_val_transforms(self) -> T.Compose:
        """Build validation transforms (no augmentation)"""
        transforms = [
            # Convert to RGB first
            T.Lambda(convert_to_rgb),
            
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        return T.Compose(transforms)
        
    def __call__(self, img):
        """Apply transforms to image"""
        return self.transform(img)
        
class ConsistentTripletTransform:
    """Apply consistent transforms across triplet images"""
    
    def __init__(
        self,
        config: Dict,
        is_train: bool = True,
        img_size: int = 384
    ):
        self.base_transform = WaldoTransforms(config, is_train, img_size)
        
        # Additional random transforms for triplet diversity
        self.random_transform = T.RandomApply([
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            )
        ], p=0.5) if is_train else None
        
    def __call__(self, anchor, positive, negative):
        """Apply transforms consistently across triplet"""
        # Base transform (same for all)
        anchor = self.base_transform(anchor)
        positive = self.base_transform(positive)
        negative = self.base_transform(negative)
        
        # Random transform for diversity (different for each)
        if self.random_transform is not None:
            anchor = self.random_transform(anchor)
            positive = self.random_transform(positive)
            negative = self.random_transform(negative)
            
        return anchor, positive, negative
