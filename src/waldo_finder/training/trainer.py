"""
Training pipeline for scene-level Waldo detection, implementing self-supervised
pre-training, contrastive learning, and curriculum progression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp import GradScaler
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import wandb
from pathlib import Path

from ..models.hierarchical_vit import HierarchicalViT
from ..models.attention import GlobalLocalAttention
from ..models.detection_head import WaldoDetectionHead, DetectionLoss
from ..data.scene_dataset import build_dataloader

class WaldoTrainer(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
        pretrain: bool = True
    ):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        self.save_hyperparameters()
        
        # Build model components
        self.build_model()
        
        # Setup loss functions
        self.setup_losses()
        
        # Initialize training state
        self.curr_curriculum_level = 0
        self.curriculum_levels = ['easy', 'medium', 'hard']
        
        # Setup mixed precision
        self.scaler = GradScaler('cuda')
        
    def build_model(self):
        """Initialize model architecture"""
        # Vision Transformer backbone
        self.backbone = HierarchicalViT(
            img_size=self.config['model']['img_size'],
            patch_size=self.config['model']['patch_size'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            hidden_dim=self.config['model']['hidden_dim'],
            mlp_dim=self.config['model']['mlp_dim'],
            dropout=self.config['model']['dropout'],
            attention_dropout=self.config['model']['attention_dropout'],
            num_levels=self.config['model']['num_levels'],
            pool_ratios=self.config['model']['pool_ratios']
        )
        
        # Global-Local attention
        self.attention = GlobalLocalAttention(
            dim=self.config['model']['hidden_dim'],
            num_heads=self.config['model']['num_heads'],
            head_dim=self.config['model']['hidden_dim'] // self.config['model']['num_heads'],
            dropout=self.config['model']['dropout'],
            window_size=7
        )
        
        # Detection head
        self.detection_head = WaldoDetectionHead(
            in_dim=self.config['model']['hidden_dim'],
            hidden_dim=self.config['detection']['hidden_dim'],
            num_scales=self.config['model']['num_levels'],
            min_size=self.config['detection']['min_size'],
            max_size=self.config['detection']['max_size']
        )
        
    def setup_losses(self):
        """Initialize loss functions"""
        # Detection losses
        self.detection_loss = DetectionLoss(
            box_weight=self.config['loss']['box_weight'],
            scale_weight=self.config['loss']['scale_weight'],
            context_weight=self.config['loss']['context_weight'],
            conf_weight=self.config['loss']['conf_weight']
        )
        
        # Contrastive loss for pre-training
        self.contrastive_loss = nn.TripletMarginLoss(
            margin=self.config['loss']['contrastive_margin']
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Optional[List[torch.Tensor]]]:
        # Get hierarchical features
        features, feature_maps = self.backbone(x, return_features=True)
        
        # Apply global-local attention
        features = self.attention(features)
        
        # Get detections
        detections = self.detection_head(features, feature_maps)
        
        return detections, feature_maps
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.pretrain:
            return self._pretrain_step(batch, batch_idx)
        return self._detection_step(batch, batch_idx)
        
    def _pretrain_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Self-supervised pre-training step"""
        with autocast('cuda'):
            # Get triplet embeddings
            anchor_feat = self.backbone(batch['anchor'])[0]
            positive_feat = self.backbone(batch['positive'])[0]
            negative_feat = self.backbone(batch['negative'])[0]
            
            # Compute contrastive loss
            loss = self.contrastive_loss(
                anchor_feat,
                positive_feat,
                negative_feat
            )
            
        # Log metrics
        self.log('train/pretrain_loss', loss)
        
        return loss
        
    def _detection_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Detection training step"""
        with autocast():
            # Forward pass
            predictions, _ = self(batch['image'])
            
            # Compute losses
            losses = self.detection_loss(
                predictions=predictions,
                targets={
                    'boxes': batch['boxes'],
                    'scales': batch['scales'],
                    'context_scores': batch['context_scores'],
                    'confidence': batch['labels'].float()
                }
            )
            
        # Log metrics
        for name, value in losses.items():
            self.log(f'train/{name}', value)
            
        return losses['loss']
        
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step"""
        if self.pretrain:
            return self._pretrain_validation_step(batch, batch_idx)
        return self._detection_validation_step(batch, batch_idx)
        
    def _pretrain_validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step for pre-training"""
        with torch.no_grad():
            # Get triplet embeddings
            anchor_feat = self.backbone(batch['anchor'])[0]
            positive_feat = self.backbone(batch['positive'])[0]
            negative_feat = self.backbone(batch['negative'])[0]
            
            # Compute contrastive loss
            loss = self.contrastive_loss(
                anchor_feat,
                positive_feat,
                negative_feat
            )
            
        self.log('val/pretrain_loss', loss)
        
        return {'val_loss': loss}
        
    def _detection_validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step for detection"""
        with torch.no_grad():
            # Forward pass
            predictions, _ = self(batch['image'])
            
            # Compute losses
            losses = self.detection_loss(
                predictions=predictions,
                targets={
                    'boxes': batch['boxes'],
                    'scales': batch['scales'],
                    'context_scores': batch['context_scores'],
                    'confidence': batch['labels'].float()
                }
            )
            
            # Post-process predictions
            boxes, scores = self.detection_head.post_process(
                predictions,
                conf_threshold=self.config['detection']['conf_threshold'],
                nms_threshold=self.config['detection']['nms_threshold']
            )
            
        # Log metrics
        for name, value in losses.items():
            self.log(f'val/{name}', value)
            
        return {
            'val_loss': losses['loss'],
            'boxes': boxes,
            'scores': scores,
            'targets': batch['boxes']
        }
        
    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler"""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['optimizer']['learning_rate'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['optimizer']['max_epochs'],
            eta_min=self.config['optimizer']['min_lr']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
        
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return build_dataloader(
            data_dir=self.config['data']['data_dir'],
            split='train',
            batch_size=self.config['data']['batch_size'],
            img_size=self.config['model']['img_size'],
            curriculum_level=self.curriculum_levels[self.curr_curriculum_level]
            if not self.pretrain else None,
            triplet_mining=self.pretrain,
            num_workers=self.config['data']['num_workers'],
            persistent_workers=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return build_dataloader(
            data_dir=self.config['data']['data_dir'],
            split='val',
            batch_size=self.config['data']['batch_size'],
            img_size=self.config['model']['img_size'],
            triplet_mining=self.pretrain,
            num_workers=self.config['data']['num_workers'],
            shuffle=False,
            persistent_workers=True
        )
        
    def on_train_epoch_end(self):
        """Handle curriculum progression"""
        if not self.pretrain:
            # Check if ready to advance curriculum
            val_loss = self.trainer.callback_metrics.get('val_loss')
            if val_loss is not None:
                if val_loss < self.config['curriculum']['advance_threshold']:
                    self.advance_curriculum()
                    
    def advance_curriculum(self):
        """Advance to next curriculum level if available"""
        if self.curr_curriculum_level < len(self.curriculum_levels) - 1:
            self.curr_curriculum_level += 1
            level = self.curriculum_levels[self.curr_curriculum_level]
            print(f"Advancing to curriculum level: {level}")
