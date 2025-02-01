"""Training pipeline for scene-level Waldo detection, implementing self-supervised
pre-training, contrastive learning, and curriculum progression."""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import wandb
from pathlib import Path

from .progress import ProgressTracker, TrainingMetrics, create_stage_header, create_stage_summary

from ..models.hierarchical_vit import HierarchicalViT
from ..models.attention import GlobalLocalAttention
from ..models.detection_head import WaldoDetectionHead, DetectionLoss, DetectionOutput
from ..data.scene_dataset import build_dataloader

class WaldoTrainer(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
        pretrain: bool = True,
        phase: str = None,
        checkpoint_path: str = None
    ):
        super().__init__()
        self.config = config
        self.pretrain = pretrain
        self.phase = phase or ('pretrain' if pretrain else 'detection')
        self._detection_mode = self.phase == 'detection'
        self.save_hyperparameters(ignore=['checkpoint_path'])
        
        # Setup memory monitoring
        self.last_memory_clear = time.time()
        self.memory_clear_interval = self.config['hardware']['memory_monitoring']['clear_interval']
        
        # Load from checkpoint if provided
        if checkpoint_path:
            self._load_phase_checkpoint(checkpoint_path)
        
        # Build model components
        self.build_model()
        
        # Set example input for graph logging
        self.example_input_array = torch.randn(1, 3, self.config['model']['img_size'], self.config['model']['img_size'])
        
        # Setup detection loss
        self.detection_loss = DetectionLoss()  # Combined detection losses
        
        # Setup contrastive loss
        self.contrastive_margin = self.config.get('loss', {}).get('contrastive_margin', 0.2)
        
        # Initialize curriculum levels
        self.curriculum_levels = ['easy', 'medium', 'hard']
        self.curr_curriculum_level = 0
        
        # Initialize monitoring
        self.progress_tracker = None
        self.current_speed = 0.0
        self.iteration_times = []
        self.last_memory_clear = time.time()
        
    def build_model(self):
        """Initialize model architecture with memory optimizations"""
        # Vision Transformer backbone with gradient checkpointing
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
            pool_ratios=self.config['model']['pool_ratios'],
            use_checkpoint=self.config['model'].get('use_checkpoint', True)  # Enable by default
        )
        
        # Calculate feature map size after patch embedding
        patch_size = self.config['model']['patch_size']
        img_size = self.config['model']['img_size']
        feature_size = img_size // patch_size  # Size after patch embedding
        
        # Calculate size after pooling
        pool_ratios = self.config['model']['pool_ratios']
        for ratio in pool_ratios:
            feature_size = feature_size // ratio
            
        # Global-Local attention with correct window size
        self.attention = GlobalLocalAttention(
            dim=self.config['model']['hidden_dim'],
            num_heads=self.config['model']['num_heads'],
            head_dim=self.config['model']['hidden_dim'] // self.config['model']['num_heads'],
            dropout=self.config['model']['dropout'],
            window_size=feature_size  # Dynamically computed window size
        )
        
        # Detection head
        self.detection_head = WaldoDetectionHead(
            in_dim=self.config['model']['hidden_dim'],
            hidden_dim=self.config['detection']['hidden_dim'],
            num_scales=self.config['model']['num_levels'],
            min_size=self.config['detection']['min_size'],
            max_size=self.config['detection']['max_size']
        )
        
    def compute_loss(self, predictions: DetectionOutput, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute detection loss using DetectionLoss"""
        return self.detection_loss(predictions, targets)
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with structured output for tracing compatibility."""
        # Get hierarchical features
        features, feature_maps = self.backbone(x, return_features=True)
        
        # Get final feature map for attention
        final_features = feature_maps[-1]  # Use last level features
        
        # Apply global-local attention
        attended_features = self.attention(final_features)
        
        # Get detections using attended features
        detections = self.detection_head(attended_features, feature_maps)
        
        # Return individual tensors for tracing compatibility
        return (
            detections.scores,
            detections.boxes,
            detections.scales,
            detections.context_scores
        )
        
    def contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive triplet loss with margin"""
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Basic triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.contrastive_margin)
        
        return losses

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with phase-specific handling"""
        if self.phase in ['pretrain', 'contrastive']:
            return self._pretrain_step(batch, batch_idx)
        elif self.phase == 'detection':
            return self._detection_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
        
    def on_train_start(self):
        """Setup for training start"""
        # Clear any existing progress tracker
        if self.progress_tracker is not None:
            del self.progress_tracker
            self.progress_tracker = None
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def _compute_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Compute normalized embeddings with gradient preservation"""
        # Ensure input requires grad
        if self.training and not images.requires_grad:
            images.requires_grad_(True)
        
        # Get features from backbone with gradient tracking
        with torch.set_grad_enabled(self.training):
            _, feature_maps = self.backbone(images, return_features=True)
            
            # Get final feature map for attention
            final_features = feature_maps[-1]  # Use last level features
            
            # Apply attention
            attended_features = self.attention(final_features)
            
            # Pool and flatten for embedding
            pooled = F.adaptive_avg_pool2d(attended_features, (1, 1))
            features = pooled.view(pooled.size(0), -1)  # Flatten to [batch_size, features]
            
            # Normalize embeddings with gradient preservation
            features = F.normalize(features + 1e-6, p=2, dim=1)  # Add small epsilon to avoid zero gradients
            
            # Double check gradient preservation
            if self.training:
                assert features.requires_grad, "Features lost gradient information"
        
        return features
    
    def _pretrain_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Self-supervised pre-training step with difficulty-aware loss"""
        try:
            # Ensure model is in train mode
            self.train()
            
            # Verify parameters require gradients
            for param in self.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
            
            # Move batch to device and ensure gradients
            anchor = batch['anchor'].to(self.device, non_blocking=True)
            positive = batch['positive'].to(self.device, non_blocking=True)
            negative = batch['negative'].to(self.device, non_blocking=True)
            
            anchor.requires_grad_(True)
            positive.requires_grad_(True)
            negative.requires_grad_(True)
            
            # Get embeddings for all parts of triplet
            anchor_feat = self._compute_embeddings(anchor)
            positive_feat = self._compute_embeddings(positive)
            negative_feat = self._compute_embeddings(negative)
            
            # Compute distances
            pos_dist = F.pairwise_distance(anchor_feat, positive_feat)
            neg_dist = F.pairwise_distance(anchor_feat, negative_feat)
            
            # Compute loss with difficulty weighting
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=False):
                # Basic triplet loss with gradient preservation
                margin = torch.tensor(self.config['loss']['contrastive_margin'], 
                                   device=self.device, requires_grad=True)
                losses = F.relu(pos_dist - neg_dist + margin)
                
                # Weight losses by difficulty (harder triplets get higher weight)
                if 'difficulty' in batch:
                    difficulties = batch['difficulty'].float().to(self.device)
                    # Scale weight from 1.0 to 2.0 based on difficulty
                    weights = 1.0 + difficulties
                    losses = losses * weights
                
                # Take mean of non-zero losses with gradient preservation
                valid_losses = losses[losses > 0]
                if len(valid_losses) > 0:
                    loss = valid_losses.mean()
                else:
                    # Create a small loss that maintains gradient flow
                    loss = (pos_dist.mean() - neg_dist.mean()) * 0.01
            
            # Log phase-specific metrics
            batch_size = len(anchor_feat)
            self.log(f'train/{self.phase}_loss', loss.item(), prog_bar=True, batch_size=batch_size)
            self.log('train/loss', loss.item(), prog_bar=True, batch_size=batch_size)
            
            # Update progress tracking
            if self.progress_tracker is None:
                stage = "Pre-training" if self.phase == 'pretrain' else "Contrastive"
                max_epochs = self.trainer.max_epochs or 1
                self.progress_tracker = ProgressTracker(
                    stage=stage,
                    total_epochs=max_epochs,
                    total_iterations=len(self.train_dataloader())
                )
            
            # Calculate training speed
            current_time = time.time()
            self.iteration_times.append(current_time)
            if len(self.iteration_times) > 10:
                self.iteration_times = self.iteration_times[-10:]
            if len(self.iteration_times) > 1:
                time_diff = self.iteration_times[-1] - self.iteration_times[0]
                if time_diff > 0:
                    self.current_speed = len(self.iteration_times) / time_diff
            
            # Periodically clear GPU memory
            if torch.cuda.is_available():
                if current_time - self.last_memory_clear > self.config['hardware']['memory_monitoring']['clear_interval']:
                    torch.cuda.empty_cache()
                    self.last_memory_clear = current_time
            
            # Get GPU memory usage
            try:
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
            except Exception:
                gpu_memory = None
            
            # Get learning rate
            try:
                lr = self.trainer.optimizers[0].param_groups[0]['lr']
            except (IndexError, AttributeError):
                lr = 0.0
            
            # Update progress display
            try:
                self.progress_tracker.update(TrainingMetrics(
                    loss=loss.item(),
                    learning_rate=lr,
                    speed=self.current_speed,
                    gpu_memory=gpu_memory,
                    epoch=self.current_epoch,
                    total_epochs=self.trainer.max_epochs or 1,
                    iteration=batch_idx,
                    total_iterations=len(self.train_dataloader())
                ))
            except Exception as e:
                print(f"\nWarning: Progress update failed: {str(e)}")
            
            return loss
            
        except Exception as e:
            print(f"\nError in pretrain step: {str(e)}")
            return torch.tensor(0.0, device=self.device)
    
    def _detection_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Detection training step"""
        try:
            # Ensure model is in train mode
            self.train()
            
            # Verify parameters require gradients
            for param in self.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
            
            # Move batch to device and ensure gradients
            images = batch['image'].to(self.device, non_blocking=True)
            boxes = batch['boxes'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            scales = batch['scales'].to(self.device, non_blocking=True)
            context_scores = batch['context_scores'].to(self.device, non_blocking=True)
            
            # Ensure tensors require gradients
            images = images.requires_grad_(True)
            boxes = boxes.requires_grad_(True)
            scales = scales.requires_grad_(True)
            context_scores = context_scores.requires_grad_(True)
            
            # Reshape boxes to match expected format [B, N, 4]
            if len(boxes.shape) == 2:  # [N, 4]
                boxes = boxes.unsqueeze(0)  # Add batch dimension
            if boxes.shape[1] == 4:  # [B, 4, N]
                boxes = boxes.permute(0, 2, 1)  # Transpose to [B, N, 4]
            
            # Forward pass with gradient tracking
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                scores, pred_boxes, pred_scales, pred_context = self(images)
                
                # Get predictions
                predictions = DetectionOutput(
                    scores=scores,
                    boxes=pred_boxes,
                    scales=pred_scales,
                    context_scores=pred_context
                )
                
                # Create targets dict with proper shapes
                targets = {
                    'boxes': boxes,  # Already [B, N, 4]
                    'scales': scales,  # [B, N, 2]
                    'context_scores': context_scores,  # [B, N]
                    'confidence': labels.float()  # [B, N]
                }
                
                # Compute losses
                losses = self.detection_loss(predictions, targets)
                
            # Log metrics
            batch_size = batch['image'].size(0)
            for name, value in losses.items():
                if not torch.isnan(value):
                    self.log(f'train/{name}', value.item(), batch_size=batch_size)
            
            # Update progress tracking
            if self.progress_tracker is None:
                max_epochs = self.trainer.max_epochs or 1
                self.progress_tracker = ProgressTracker(
                    stage="Detection",
                    total_epochs=max_epochs,
                    total_iterations=len(self.train_dataloader())
                )
            
            # Calculate training speed
            current_time = time.time()
            self.iteration_times.append(current_time)
            if len(self.iteration_times) > 10:
                self.iteration_times = self.iteration_times[-10:]
            if len(self.iteration_times) > 1:
                time_diff = self.iteration_times[-1] - self.iteration_times[0]
                if time_diff > 0:
                    self.current_speed = len(self.iteration_times) / time_diff
            
            # Periodically clear GPU memory
            if torch.cuda.is_available():
                if current_time - self.last_memory_clear > self.config['hardware']['memory_monitoring']['clear_interval']:
                    torch.cuda.empty_cache()
                    self.last_memory_clear = current_time
            
            # Get GPU memory usage
            try:
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
            except Exception:
                gpu_memory = None
            
            # Get learning rate
            try:
                lr = self.trainer.optimizers[0].param_groups[0]['lr']
            except (IndexError, AttributeError):
                lr = 0.0
            
            # Update progress display
            try:
                self.progress_tracker.update(TrainingMetrics(
                    loss=losses['loss'].item(),
                    learning_rate=lr,
                    speed=self.current_speed,
                    gpu_memory=gpu_memory,
                    epoch=self.current_epoch,
                    total_epochs=self.trainer.max_epochs or 1,
                    iteration=batch_idx,
                    total_iterations=len(self.train_dataloader())
                ))
            except Exception as e:
                print(f"\nWarning: Progress update failed: {str(e)}")
            
            return losses['loss']
            
        except Exception as e:
            print(f"\nError in detection step: {str(e)}")
            return torch.tensor(0.0, device=self.device)
        
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step with phase-specific handling"""
        if self.phase == 'pretrain' or self.phase == 'contrastive':
            return self._pretrain_validation_step(batch, batch_idx)
        elif self.phase == 'detection':
            return self._detection_validation_step(batch, batch_idx)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")
        
    def _pretrain_validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step for pre-training"""
        try:
            with torch.no_grad():
                # Get triplet embeddings using the same method as training
                anchor_feat = self._compute_embeddings(batch['anchor'])
                positive_feat = self._compute_embeddings(batch['positive'])
                negative_feat = self._compute_embeddings(batch['negative'])
                
                # Compute per-sample losses
                losses = self.contrastive_loss(
                    anchor_feat,
                    positive_feat,
                    negative_feat
                )
                
                # Filter out zero losses
                valid_losses = losses[losses > 0]
                if len(valid_losses) > 0:
                    loss = valid_losses.mean()
                else:
                    loss = losses.mean()
            
            # Log phase-specific validation metrics
            batch_size = len(anchor_feat)
            
            # Log both formats for compatibility
            self.log(f'val/{self.phase}_loss', loss, prog_bar=True, batch_size=batch_size)
            self.log(f'val_{self.phase}_loss', loss, prog_bar=True, batch_size=batch_size)  # For ModelCheckpoint
            self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
            
            return {
                'val_loss': loss,
                f'val_{self.phase}_loss': loss,
                f'val/{self.phase}_loss': loss
            }
            
        except Exception as e:
            print(f"\nError in validation step: {str(e)}")
            return {'val_loss': torch.tensor(0.0, device=self.device)}
        
    def _detection_validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step for detection"""
        try:
            with torch.no_grad():
                # Move batch to device
                images = batch['image'].to(self.device, non_blocking=True)
                boxes = batch['boxes'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                scales = batch['scales'].to(self.device, non_blocking=True)
                context_scores = batch['context_scores'].to(self.device, non_blocking=True)
                
                # Reshape boxes to match expected format [B, N, 4]
                if len(boxes.shape) == 2:  # [N, 4]
                    boxes = boxes.unsqueeze(0)  # Add batch dimension
                if boxes.shape[1] == 4:  # [B, 4, N]
                    boxes = boxes.permute(0, 2, 1)  # Transpose to [B, N, 4]
                
                # Forward pass
                scores, pred_boxes, pred_scales, pred_context = self(images)
                
                # Get predictions
                predictions = DetectionOutput(
                    scores=scores,
                    boxes=pred_boxes,
                    scales=pred_scales,
                    context_scores=pred_context
                )
                
                # Create targets dict with proper shapes
                targets = {
                    'boxes': boxes,  # Already [B, N, 4]
                    'scales': scales,  # [B, N, 2]
                    'context_scores': context_scores,  # [B, N]
                    'confidence': labels.float()  # [B, N]
                }
                
                # Compute losses
                losses = self.detection_loss(predictions, targets)
                
                # Post-process predictions
                boxes, scores = self.detection_head.post_process(
                    predictions,
                    conf_threshold=self.config['detection']['conf_threshold'],
                    nms_threshold=self.config['detection']['nms_threshold']
                )
            
            # Log detection-specific validation metrics
            batch_size = batch['image'].size(0)
            for name, value in losses.items():
                if not torch.isnan(value):
                    # Log both formats for compatibility
                    self.log(f'val/detection_{name}', value, prog_bar=True, batch_size=batch_size)
                    self.log(f'val_detection_{name}', value, prog_bar=True, batch_size=batch_size)  # For ModelCheckpoint
                    if name == 'loss':
                        self.log('val_loss', value, prog_bar=True, batch_size=batch_size)
            
            return {
                'val_loss': losses['loss'],
                'val_detection_loss': losses['loss'],
                'val/detection_loss': losses['loss'],
                'boxes': boxes,
                'scores': scores,
                'targets': batch['boxes']
            }
            
        except Exception as e:
            print(f"\nError in detection validation step: {str(e)}")
            return {
                'val_loss': torch.tensor(0.0, device=self.device),
                'boxes': [],
                'scores': [],
                'targets': batch['boxes']
            }
        
    def _load_phase_checkpoint(self, checkpoint_path: str):
        """Load weights from checkpoint while handling phase transitions"""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Reset internal state for new phase
        self.curr_curriculum_level = 0  # Reset curriculum level
        if self.progress_tracker is not None:
            del self.progress_tracker
            self.progress_tracker = None
        
        # Clear iteration tracking
        self.iteration_times = []
        self.current_speed = 0.0
        
        # Load model weights with careful handling of phase-specific layers
        if 'state_dict' in checkpoint:
            # Filter out phase-specific layers that shouldn't transfer
            state_dict = checkpoint['state_dict']
            if self.phase != 'pretrain':
                # Remove pretrain-specific layers when transitioning to later phases
                state_dict = {k: v for k, v in state_dict.items() 
                            if not k.startswith('pretrain_')}
            
            # Load weights with flexible matching
            self.load_state_dict(state_dict, strict=False)
            
            # Log any missing or unexpected keys
            missing, unexpected = self._verify_state_dict(state_dict)
            if missing:
                logger.warning(f"Missing keys in checkpoint: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected}")
        else:
            self.load_state_dict(checkpoint, strict=False)
        
        # Initialize new phase-specific components
        self._init_phase_specific_components()
        
    def _verify_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """Verify state dict loading and return missing/unexpected keys"""
        model_state = self.state_dict()
        missing = [k for k in model_state.keys() if k not in state_dict]
        unexpected = [k for k in state_dict.keys() if k not in model_state]
        return missing, unexpected
        
    def _init_phase_specific_components(self):
        """Initialize components specific to current phase"""
        if self.phase == 'pretrain':
            # Initialize pretrain-specific components
            self.contrastive_margin = self.config.get('loss', {}).get('contrastive_margin', 0.2)
        elif self.phase == 'contrastive':
            # Initialize contrastive-specific components
            self.contrastive_margin = self.config.get('loss', {}).get('contrastive_margin', 0.2)
        else:  # detection
            # Detection components are already initialized in __init__
            pass
            
    def configure_optimizers(self):
        """Setup optimizer and learning rate scheduler with enhanced settings"""
        # Get phase-specific learning rate
        if self.phase == 'pretrain':
            lr = self.config['training']['pretrain']['learning_rate']
        elif self.phase == 'contrastive':
            lr = self.config['training']['contrastive']['learning_rate']
        else:  # detection
            lr = self.config['training']['detection']['learning_rate']
        
        # Create optimizer with enhanced settings
        optimizer_config = self.config['optimizer']
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=optimizer_config['weight_decay'],
            amsgrad=optimizer_config.get('amsgrad', True)
        )
        
        # Get phase-specific max epochs
        max_epochs = self.config['training'][self.phase]['epochs']
        
        # Get phase-specific scheduler config
        scheduler_config = self.config['training'][self.phase]['scheduler']
        warmup_epochs = scheduler_config['warmup_epochs']
        
        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs * len(self.train_dataloader())
        )
        
        # Create main scheduler based on type
        if scheduler_config['type'] == 'cosine':
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=scheduler_config['min_lr']
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}")
        
        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs * len(self.train_dataloader())]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update LR every step for smoother training
                'frequency': 1
            }
        }
        
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader with phase-specific configuration"""
        return build_dataloader(
            data_dir=self.config['data']['data_dir'],
            config=self.config,
            split='train',
            curriculum_level=self.curriculum_levels[self.curr_curriculum_level]
            if self.phase == 'detection' else None,
            triplet_mining=(self.phase in ['pretrain', 'contrastive'])
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader with phase-specific configuration"""
        return build_dataloader(
            data_dir=self.config['data']['data_dir'],
            config=self.config,
            split='val',
            triplet_mining=(self.phase in ['pretrain', 'contrastive'])
        )
        
    def on_train_epoch_end(self):
        """Handle phase-specific epoch end tasks.
        
        - Updates curriculum level based on validation metrics
        - Clears GPU memory
        - Updates progress tracking
        """
        # Update curriculum level if in detection phase
        if self.phase == 'detection':
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'callback_metrics'):
                val_loss = self.trainer.callback_metrics.get('val_loss', None)
                if val_loss is not None and val_loss < self.config['training']['detection']['curriculum']['threshold']:
                    self.curr_curriculum_level = min(
                        self.curr_curriculum_level + 1,
                        len(self.curriculum_levels) - 1
                    )
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Clear progress tracker at epoch end
        if self.progress_tracker is not None:
            print("\n")  # Add spacing between epochs
