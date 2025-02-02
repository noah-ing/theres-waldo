"""Scene-level dataset implementation with improved triplet mining,
curriculum learning, and memory efficiency optimizations."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import random
import logging
from functools import lru_cache
import multiprocessing as mp
import io
import pickle
import os
from .transforms import ConsistentTripletTransform, WaldoTransforms

logger = logging.getLogger(__name__)

class ImageCache:
    """Thread-safe image cache with atomic disk operations"""
    def __init__(self, capacity: int = 100, cache_dir: str = "data/cache"):
        self.capacity = capacity
        self.cache_dir = os.path.abspath(cache_dir)
        self.memory_cache = {}
        
        # Clean start for each process
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def __getstate__(self):
        """Custom state for worker processes"""
        return {
            'capacity': self.capacity,
            'cache_dir': self.cache_dir,
            'memory_cache': {}  # Fresh cache per worker
        }
        
    def __setstate__(self, state):
        """Restore state in worker"""
        self.capacity = state['capacity']
        self.cache_dir = state['cache_dir']
        self.memory_cache = state['memory_cache']
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with robust error handling"""
        try:
            # Try memory cache first
            if key in self.memory_cache:
                return self.memory_cache[key]
                
            # Try disk cache with atomic read
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            temp_file = os.path.join(self.cache_dir, f"{key}.tmp")
            
            if os.path.exists(cache_file):
                try:
                    # First try reading directly
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                    self.memory_cache[key] = value
                    return value
                except (EOFError, pickle.UnpicklingError) as e:
                    # File is corrupted, try backup
                    if os.path.exists(temp_file):
                        try:
                            with open(temp_file, 'rb') as f:
                                value = pickle.load(f)
                            # Restore from backup
                            import shutil
                            shutil.copy2(temp_file, cache_file)
                            self.memory_cache[key] = value
                            return value
                        except:
                            pass
                            
                    # Both files corrupted or missing
                    logger.warning(f"Cache file corrupted: {cache_file}")
                    self._cleanup_cache_files(key)
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error loading cache: {str(e)}")
                    self._cleanup_cache_files(key)
                    return None
                    
            return None
        except Exception as e:
            logger.error(f"Cache error for key {key}: {str(e)}")
            return None
            
    def _cleanup_cache_files(self, key: str):
        """Clean up corrupted cache files"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            temp_file = os.path.join(self.cache_dir, f"{key}.tmp")
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Also clear from memory cache
            self.memory_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Error cleaning cache files: {str(e)}")
        
    def set(self, key: str, value: Any):
        """Set value in cache with atomic write"""
        # Update memory cache with LRU eviction
        if len(self.memory_cache) >= self.capacity:
            if self.memory_cache:
                self.memory_cache.pop(next(iter(self.memory_cache)))
        self.memory_cache[key] = value
        
        # Update disk cache with atomic write
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        temp_file = os.path.join(self.cache_dir, f"{key}.tmp")
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Write to temporary file first
            with open(temp_file, 'wb') as f:
                pickle.dump(value, f, protocol=4)  # Use stable protocol version
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
                
            # Atomic rename for final write
            import shutil
            shutil.move(temp_file, cache_file)
            
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {str(e)}")
            self._cleanup_cache_files(key)

class SceneDataset(Dataset):
    """Enhanced dataset for Waldo detection with improved triplet mining"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: Dict[str, Any],
        split: str = 'train',
        curriculum_level: Optional[str] = None,
        triplet_mining: bool = False,
        max_triplets_per_scene: int = 10,
        cache_size: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.curriculum_level = curriculum_level
        self.triplet_mining = triplet_mining
        self.max_triplets_per_scene = max_triplets_per_scene
        
        # Setup caches with persistence using absolute paths
        cache_root = os.path.abspath("data/cache")
        
        # Initialize image cache with absolute paths
        self.image_cache = ImageCache(
            capacity=cache_size,
            cache_dir=os.path.join(cache_root, "images")
        )
        
        # Setup transforms
        self.transform = ConsistentTripletTransform(
            config,
            is_train=(split == 'train'),
            img_size=config['model']['img_size']
        ) if triplet_mining else WaldoTransforms(
            config,
            is_train=(split == 'train'),
            img_size=config['model']['img_size']
        )
        
        try:
            # Load scene annotations
            self.annotations = self._load_annotations()
            
            # Load scenes based on curriculum
            if curriculum_level is not None:
                self.scenes = self._load_curriculum_scenes()
            else:
                self.scenes = self._load_all_scenes()
                
            # Setup triplet mining if enabled
            if triplet_mining:
                self._initialize_triplets()
                
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise
            
    def _initialize_triplets(self):
        """Initialize triplets with caching"""
        # Always regenerate triplets to avoid deserialization issues
        self.triplets = self._mine_triplets()
        logger.info(f"Generated {len(self.triplets)} triplets for {self.split} split")
            
    def _load_annotations(self) -> Dict[str, Any]:
        """Load and validate scene annotations"""
        ann_file = self.data_dir / f'{self.split}_annotations.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_file}")
            
        try:
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
                
            # Validate annotations structure
            if not isinstance(annotations, dict) or 'scenes' not in annotations:
                raise ValueError("Invalid annotations format")
                
            return annotations
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in annotations file: {ann_file}")
            
    def _load_curriculum_scenes(self) -> List[Dict[str, Any]]:
        """Load and validate curriculum-specific scenes"""
        curr_file = self.data_dir / 'curriculum' / f'{self.curriculum_level}.json'
        if not curr_file.exists():
            raise FileNotFoundError(f"Curriculum file not found: {curr_file}")
            
        try:
            with open(curr_file, 'r') as f:
                scene_ids = json.load(f)
                
            # Filter scenes based on curriculum
            curriculum_scenes = [
                scene for scene in self.annotations['scenes']
                if scene['id'] in scene_ids
            ]
            
            if not curriculum_scenes:
                raise ValueError(f"No scenes found for curriculum level: {self.curriculum_level}")
                
            return curriculum_scenes
            
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in curriculum file: {curr_file}")
        
    def _load_all_scenes(self) -> List[Dict[str, Any]]:
        """Load and validate all scenes for current split"""
        scenes = self.annotations['scenes']
        if not scenes:
            raise ValueError(f"No scenes found for split: {self.split}")
        return scenes
        
    @staticmethod
    def _compute_scene_complexity(scene: Dict[str, Any]) -> float:
        """Compute scene complexity score based on multiple factors"""
        # Basic complexity from number of objects
        num_objects = len(scene['boxes'])
        
        # Additional complexity factors
        num_similar_objects = sum(1 for box in scene['boxes']
                                if box.get('similarity_score', 0) > 0.7)
        
        avg_box_size = np.mean([
            (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
            for box in scene['boxes']
        ])
        
        # Combine factors into complexity score
        complexity = (
            0.5 * num_objects +
            0.3 * num_similar_objects +
            0.2 * (1 - avg_box_size)  # Smaller objects = more complex
        )
        
        return complexity
        
    def _mine_triplets(self) -> List[Dict[str, Any]]:
        """Enhanced triplet mining with complexity-aware sampling"""
        triplets = []
        
        # Collect and analyze scenes
        all_waldos = []
        scene_complexities = {}
        
        for scene in self.scenes:
            # Compute comprehensive scene complexity
            complexity = self._compute_scene_complexity(scene)
            scene_complexities[scene['id']] = complexity
            
            # Collect Waldo instances with metadata
            for box in scene['boxes']:
                if box['category'] == 'waldo':
                    box_with_meta = box.copy()
                    box_with_meta.update({
                        'id': scene['id'],
                        'complexity': complexity,
                        'context_score': box.get('context_score', 0.0),
                        'similarity_score': box.get('similarity_score', 0.0)
                    })
                    all_waldos.append(box_with_meta)
        
        if not all_waldos:
            raise ValueError("No Waldo instances found in dataset")
            
        # Sort scenes by complexity for curriculum sampling
        complexity_range = max(scene_complexities.values()) - min(scene_complexities.values())
        
        triplets = []
        for anchor in all_waldos:
            # Normalize complexity to 0-1 range
            norm_complexity = (anchor['complexity'] - min(scene_complexities.values())) / complexity_range
            
            # Dynamic triplet generation based on complexity
            num_triplets = max(
                3,
                int(self.max_triplets_per_scene * (0.5 + 0.5 * norm_complexity))
            )
            
            # Find similar Waldos for positives
            similar_waldos = [
                w for w in all_waldos
                if w['id'] != anchor['id'] and
                abs(w['complexity'] - anchor['complexity']) / complexity_range < 0.2 and
                w['similarity_score'] > 0.7
            ]
            
            # Generate triplets with curriculum-aware sampling
            for _ in range(num_triplets):
                triplet = self._generate_curriculum_triplet(
                    anchor, similar_waldos, all_waldos, norm_complexity
                )
                if triplet:
                    triplets.append(triplet)
                    
                if len(triplets) >= self.max_triplets_per_scene * len(self.scenes):
                    break
                        
        # Shuffle triplets for better training
        random.shuffle(triplets)
        return triplets[:self.max_triplets_per_scene * len(self.scenes)]
        
    def _generate_curriculum_triplet(
        self,
        anchor: Dict[str, Any],
        similar_waldos: List[Dict[str, Any]],
        all_waldos: List[Dict[str, Any]],
        norm_complexity: float
    ) -> Optional[Dict[str, Any]]:
        """Generate a single triplet with curriculum-aware sampling"""
        try:
            # Select positive example
            if similar_waldos:
                positive = random.choice(similar_waldos)
            else:
                # Create synthetic positive with controlled variations
                positive = anchor.copy()
                # Add curriculum-aware jitter
                jitter = 0.05 + (0.15 * norm_complexity)
                w = positive['x2'] - positive['x1']
                h = positive['y2'] - positive['y1']
                scale = 1.0 + random.uniform(-jitter, jitter)
                positive.update({
                    'x1': max(0.0, positive['x1'] + (1-scale)*w/2),
                    'x2': min(1.0, positive['x2'] - (1-scale)*w/2),
                    'y1': max(0.0, positive['y1'] + (1-scale)*h/2),
                    'y2': min(1.0, positive['y2'] - (1-scale)*h/2)
                })
            
            # Select negative based on curriculum difficulty
            other_waldos = [w for w in all_waldos if w['id'] != anchor['id']]
            if not other_waldos:
                return None
                
            if norm_complexity < 0.3:  # Easy: very different Waldos
                candidates = [
                    w for w in other_waldos
                    if abs(w['complexity'] - anchor['complexity']) > 0.3 * (max(w['complexity'] for w in all_waldos) - min(w['complexity'] for w in all_waldos))
                ]
            elif norm_complexity < 0.7:  # Medium: mix of similar and different
                if random.random() < 0.7:
                    candidates = [
                        w for w in other_waldos
                        if abs(w['complexity'] - anchor['complexity']) < 0.2 * (max(w['complexity'] for w in all_waldos) - min(w['complexity'] for w in all_waldos))
                    ]
                else:
                    candidates = other_waldos
            else:  # Hard: similar Waldos
                candidates = [
                    w for w in other_waldos
                    if abs(w['complexity'] - anchor['complexity']) < 0.1 * (max(w['complexity'] for w in all_waldos) - min(w['complexity'] for w in all_waldos))
                    and w['similarity_score'] > 0.8
                ]
            
            negative = random.choice(candidates if candidates else other_waldos)
            
            return {
                'scene_id': anchor['id'],
                'anchor': anchor,
                'positive': positive,
                'negative': negative,
                'difficulty': norm_complexity,
                'complexity': anchor['complexity']
            }
            
        except Exception as e:
            logger.warning(f"Error generating triplet: {str(e)}")
            return None
        
    def _load_image(self, img_path: Path) -> Image.Image:
        """Load and cache image file with shared memory and format validation"""
        try:
            # Convert Path to string for dict key
            path_str = str(img_path)
            
            # Check cache first
            cached_img = self.image_cache.get(path_str)
            if cached_img is not None:
                try:
                    img = Image.open(io.BytesIO(cached_img))
                    logger.debug(f"Loaded cached image {path_str} with mode {img.mode}")
                    return img
                except Exception as e:
                    logger.warning(f"Failed to load cached image {path_str}: {str(e)}")
                    # Clear corrupted cache entry
                    self.image_cache.memory_cache.pop(path_str, None)
                
            # Load image if not in cache
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            # Load and cache image bytes
            try:
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
            except Exception as e:
                raise IOError(f"Failed to read image file {img_path}: {str(e)}")
                
            try:
                img = Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                raise ValueError(f"Failed to decode image {img_path}: {str(e)}")
                
            # Cache the image bytes
            self.image_cache.set(path_str, img_bytes)
            
            # Log image format for debugging
            logger.debug(f"Loaded new image {path_str} with mode {img.mode}")
            
            # Validate image format
            if img.mode not in ['RGB', 'RGBA', 'CMYK']:
                logger.warning(f"Unexpected image mode {img.mode} for {path_str}")
                
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
            
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.triplets if self.triplet_mining else self.scenes)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item with error handling"""
        try:
            if self.triplet_mining:
                return self._get_triplet_item(idx)
            return self._get_scene_item(idx)
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            raise
        
    def _get_scene_item(self, idx: int) -> Dict[str, Any]:
        """Get a full scene with annotations and metadata"""
        scene = self.scenes[idx]
        
        # Load and transform image
        img_path = self.data_dir / 'images' / f"{scene['id']}.jpg"
        image = self._load_image(img_path)
        image = self.transform(image)
        
        # Prepare tensors without gradients initially
        boxes = torch.tensor([
            [box['x1'], box['y1'], box['x2'], box['y2']]
            for box in scene['boxes']
        ], dtype=torch.float32)
        
        labels = torch.tensor([
            1 if box['category'] == 'waldo' else 0
            for box in scene['boxes']
        ], dtype=torch.long)
        
        # Compute scales from box dimensions
        scales = torch.tensor([
            [(box['x2'] - box['x1']), (box['y2'] - box['y1'])]
            for box in scene['boxes']
        ], dtype=torch.float32)
        
        # Enhanced metadata
        metadata = {
            'scales': scales,
            'context_scores': torch.tensor([
                box.get('context_score', 0.0)
                for box in scene['boxes']
            ], dtype=torch.float32),
            'similarity_scores': torch.tensor([
                box.get('similarity_score', 0.0)
                for box in scene['boxes']
            ], dtype=torch.float32),
            'scene_complexity': self._compute_scene_complexity(scene),
            'scene_id': scene['id']
        }
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            **metadata
        }
        
    def _get_triplet_item(self, idx: int) -> Dict[str, Any]:
        """Get a triplet with enhanced metadata"""
        triplet = self.triplets[idx]
        
        # Load scene image
        img_path = self.data_dir / 'images' / f"{triplet['scene_id']}.jpg"
        image = self._load_image(img_path)
        
        # Extract and transform patches
        patches = {
            'anchor': self._extract_patch(image, triplet['anchor']),
            'positive': self._extract_patch(image, triplet['positive']),
            'negative': self._extract_patch(image, triplet['negative'])
        }
        
        # Apply consistent transforms
        transformed = self.transform(
            patches['anchor'],
            patches['positive'],
            patches['negative']
        )
        
        # Enhanced metadata
        metadata = {
            'scene_id': triplet['scene_id'],
            'difficulty': triplet['difficulty'],
            'complexity': triplet['complexity'],
            'similarity_scores': {
                'anchor': triplet['anchor'].get('similarity_score', 0.0),
                'positive': triplet['positive'].get('similarity_score', 0.0),
                'negative': triplet['negative'].get('similarity_score', 0.0)
            }
        }
        
        return {
            'anchor': transformed[0],
            'positive': transformed[1],
            'negative': transformed[2],
            **metadata
        }
        
    def _extract_patch(self, image: Image.Image, box: Dict[str, Any]) -> Image.Image:
        """Extract and validate image patch"""
        try:
            # Convert normalized coordinates to pixels
            w, h = image.size
            x1 = int(box['x1'] * w)
            y1 = int(box['y1'] * h)
            x2 = int(box['x2'] * w)
            y2 = int(box['y2'] * h)
            
            # Validate coordinates
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"Invalid box coordinates: {box}")
                
            # Extract patch
            patch = image.crop((x1, y1, x2, y2))
            return patch
            
        except Exception as e:
            logger.error(f"Error extracting patch: {str(e)}")
            raise

def build_dataloader(
    data_dir: Union[str, Path],
    config: Dict[str, Any],
    split: str = 'train',
    curriculum_level: Optional[str] = None,
    triplet_mining: bool = False
) -> DataLoader:
    """Build an optimized dataloader with the specified configuration"""
    try:
        # Create dataset with optimizations
        dataset = SceneDataset(
            data_dir=data_dir,
            config=config,
            split=split,
            curriculum_level=curriculum_level,
            triplet_mining=triplet_mining,
            max_triplets_per_scene=config['data']['max_triplets_per_scene'],
            cache_size=config['data']['cache_size']
        )
        
        # Configure dataloader with optimized settings
        dataloader_config = {
            'batch_size': config['data']['batch_size'],
            'num_workers': config['data']['num_workers'],
            'shuffle': (split == 'train'),
            'pin_memory': config['data']['pin_memory'],
            'persistent_workers': config['data']['persistent_workers'],
            'prefetch_factor': config['data']['prefetch_factor']
        }
        
        return DataLoader(dataset, **dataloader_config)
        
    except Exception as e:
        logger.error(f"Error building dataloader: {str(e)}")
        raise
