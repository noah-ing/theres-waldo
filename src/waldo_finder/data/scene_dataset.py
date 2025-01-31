"""
Scene-level dataset implementation with triplet mining and curriculum learning support.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import random
from einops import rearrange, repeat

class SceneDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        img_size: int = 384,
        transform: Optional[nn.Module] = None,
        curriculum_level: Optional[str] = None,
        triplet_mining: bool = False,
        max_triplets_per_scene: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.curriculum_level = curriculum_level
        self.triplet_mining = triplet_mining
        self.max_triplets_per_scene = max_triplets_per_scene
        
        # Load scene annotations
        self.annotations = self._load_annotations()
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
        
        # Load curriculum data if specified
        if curriculum_level is not None:
            self.scenes = self._load_curriculum_scenes()
        else:
            self.scenes = self._load_all_scenes()
            
        # Setup triplet mining if enabled
        if triplet_mining:
            self.triplets = self._mine_triplets()
            
    def _load_annotations(self) -> Dict:
        """Load scene annotations from JSON file"""
        ann_file = self.data_dir / f'{self.split}_annotations.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_file}")
            
        with open(ann_file, 'r') as f:
            return json.load(f)
            
    def _load_curriculum_scenes(self) -> List[Dict]:
        """Load scenes for current curriculum level"""
        curr_file = self.data_dir / 'curriculum' / f'{self.curriculum_level}.json'
        if not curr_file.exists():
            raise FileNotFoundError(f"Curriculum file not found: {curr_file}")
            
        with open(curr_file, 'r') as f:
            scene_ids = json.load(f)
            
        return [
            scene for scene in self.annotations['scenes']
            if scene['id'] in scene_ids
        ]
        
    def _load_all_scenes(self) -> List[Dict]:
        """Load all scenes for current split"""
        return self.annotations['scenes']
        
    def _mine_triplets(self) -> List[Dict]:
        """Mine triplets for contrastive learning"""
        triplets = []
        
        for scene in self.scenes:
            # Get Waldo boxes in this scene and add scene id
            waldo_boxes = []
            for box in scene['boxes']:
                if box['category'] == 'waldo':
                    box_with_id = box.copy()
                    box_with_id['id'] = scene['id']
                    waldo_boxes.append(box_with_id)
            
            # Get non-Waldo boxes (hard negatives) and add scene id
            hard_negatives = []
            for box in scene['boxes']:
                if box['category'] != 'waldo':
                    box_with_id = box.copy()
                    box_with_id['id'] = scene['id']
                    hard_negatives.append(box_with_id)
            
            # For pre-training, treat each Waldo box as both anchor and positive
            # and use other scenes' Waldo boxes as negatives
            for i, scene_i in enumerate(self.scenes):
                waldo_i = next((box for box in scene_i['boxes'] if box['category'] == 'waldo'), None)
                if waldo_i is None:
                    continue
                    
                # Add scene id to box
                waldo_i = waldo_i.copy()
                waldo_i['id'] = scene_i['id']
                
                # Use Waldo boxes from other scenes as negatives
                for j, scene_j in enumerate(self.scenes):
                    if i == j:
                        continue
                        
                    waldo_j = next((box for box in scene_j['boxes'] if box['category'] == 'waldo'), None)
                    if waldo_j is None:
                        continue
                        
                    # Add scene id to negative box
                    waldo_j = waldo_j.copy()
                    waldo_j['id'] = scene_j['id']
                    
                    # Create triplet using same box as anchor and positive
                    triplets.append({
                        'scene_id': scene_i['id'],
                        'anchor': waldo_i,
                        'positive': waldo_i,  # Same box as anchor
                        'negative': waldo_j
                    })
                        
        return triplets
        
    def __len__(self) -> int:
        if self.triplet_mining:
            return len(self.triplets)
        return len(self.scenes)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.triplet_mining:
            return self._get_triplet_item(idx)
        return self._get_scene_item(idx)
        
    def _get_scene_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a full scene with annotations"""
        scene = self.scenes[idx]
        
        # Load image
        img_path = self.data_dir / 'images' / f"{scene['id']}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Prepare boxes and labels
        boxes = torch.tensor([
            [box['x1'], box['y1'], box['x2'], box['y2']]
            for box in scene['boxes']
        ], dtype=torch.float32)
        
        labels = torch.tensor([
            1 if box['category'] == 'waldo' else 0
            for box in scene['boxes']
        ], dtype=torch.long)
        
        # Prepare context scores (scene complexity indicators)
        context_scores = torch.tensor([
            box.get('context_score', 0.0)
            for box in scene['boxes']
        ], dtype=torch.float32)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'context_scores': context_scores,
            'scene_id': scene['id']
        }
        
    def _get_triplet_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a triplet of anchor, positive, and negative examples"""
        triplet = self.triplets[idx]
        scene_id = triplet['scene_id']
        
        # Load scene image
        img_path = self.data_dir / 'images' / f"{scene_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Extract patches
        anchor_patch = self._extract_patch(image, triplet['anchor'])
        positive_patch = self._extract_patch(image, triplet['positive'])
        negative_patch = self._extract_patch(image, triplet['negative'])
        
        # Apply transforms
        anchor_patch = self.transform(anchor_patch)
        positive_patch = self.transform(positive_patch)
        negative_patch = self.transform(negative_patch)
        
        return {
            'anchor': anchor_patch,
            'positive': positive_patch,
            'negative': negative_patch,
            'scene_id': scene_id
        }
        
    def _extract_patch(self, image: Image.Image, box: Dict) -> Image.Image:
        """Extract a patch from an image given a bounding box"""
        x1, y1, x2, y2 = (
            box['x1'], box['y1'],
            box['x2'], box['y2']
        )
        
        # Convert normalized coordinates to pixels
        w, h = image.size
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
        
        # Extract patch
        patch = image.crop((x1, y1, x2, y2))
        return patch

def build_dataloader(
    data_dir: Union[str, Path],
    split: str = 'train',
    batch_size: int = 32,
    img_size: int = 384,
    transform: Optional[nn.Module] = None,
    curriculum_level: Optional[str] = None,
    triplet_mining: bool = False,
    max_triplets_per_scene: int = 10,
    num_workers: int = 4,
    shuffle: bool = True,
    persistent_workers: bool = False
) -> DataLoader:
    """Build a dataloader with the specified configuration"""
    dataset = SceneDataset(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        transform=transform,
        curriculum_level=curriculum_level,
        triplet_mining=triplet_mining,
        max_triplets_per_scene=max_triplets_per_scene
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers
    )
