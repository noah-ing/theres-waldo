"""
Training script for scene-level Waldo detection model.
Implements multi-stage training with pre-training and curriculum learning.
"""

import os
import torch
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
from typing import Dict, Optional

from .trainer import WaldoTrainer

def setup_callbacks(config: DictConfig, stage: str) -> Dict:
    """Setup training callbacks"""
    callbacks = []
    
    # Determine monitor metric based on stage
    if stage == 'pretrain':
        monitor_metric = 'train/pretrain_loss'
    elif stage == 'contrastive':
        monitor_metric = 'train/contrastive_loss'
    else:
        monitor_metric = 'val/detection_loss'
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        save_top_k=config.checkpoint.save_top_k,
        monitor=monitor_metric,
        mode='min',
        save_last=config.checkpoint.save_last
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        mode='min',
        patience=10,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    return callbacks

def setup_logger(config: DictConfig) -> Optional[WandbLogger]:
    """Setup WandB logger if enabled"""
    if config.logging.get('use_wandb', False):
        return WandbLogger(
            project=config.logging.project,
            name=config.logging.name,
            save_dir=config.logging.save_dir,
            log_model=True
        )
    return None

def train_stage(
    config: DictConfig,
    stage: str,
    pretrain: bool = False,
    ckpt_path: Optional[str] = None
) -> str:
    """Run a training stage and return the best checkpoint path"""
    print(f"\nStarting {stage} training stage...")
    
    # Initialize trainer
    model = WaldoTrainer(config=config, pretrain=pretrain)
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(config, stage)
    logger = setup_logger(config)
    
    # Configure trainer
    trainer_kwargs = {
        'max_epochs': config.training[stage].epochs,
        'accelerator': config.hardware.accelerator,
        'devices': config.hardware.devices,
        'strategy': config.hardware.strategy,
        'precision': config.hardware.precision,
        'callbacks': callbacks,
        'gradient_clip_val': config.optimizer.gradient_clip,
    }
    
    # Only add logger if wandb is enabled
    if logger is not None:
        trainer_kwargs['logger'] = logger
        trainer_kwargs['log_every_n_steps'] = config.logging.log_every_n_steps
        
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    trainer.fit(
        model,
        ckpt_path=ckpt_path
    )
    
    # Return best checkpoint path
    return trainer.checkpoint_callback.best_model_path

@hydra.main(config_path="../../../config", config_name="scene_model", version_base="1.1")
def main(config: DictConfig):
    """Main training pipeline"""
    print("Starting Waldo detection training pipeline...")
    
    # Create output directories
    os.makedirs(config.logging.save_dir, exist_ok=True)
    os.makedirs(config.checkpoint.dirpath, exist_ok=True)
    
    best_ckpt = None
    
    # Pre-training stage
    if config.training.pretrain.enabled:
        best_ckpt = train_stage(
            config=config,
            stage='pretrain',
            pretrain=True
        )
        print(f"Pre-training complete. Best checkpoint: {best_ckpt}")
    
    # Contrastive learning stage
    if config.training.contrastive.enabled:
        best_ckpt = train_stage(
            config=config,
            stage='contrastive',
            pretrain=True,
            ckpt_path=best_ckpt
        )
        print(f"Contrastive learning complete. Best checkpoint: {best_ckpt}")
    
    # Detection training stage
    if config.training.detection.enabled:
        best_ckpt = train_stage(
            config=config,
            stage='detection',
            pretrain=False,
            ckpt_path=best_ckpt
        )
        print(f"Detection training complete. Best checkpoint: {best_ckpt}")
    
    print("\nTraining pipeline complete!")
    print(f"Final best checkpoint: {best_ckpt}")
    
    # Close wandb run if it was used
    if config.logging.get('use_wandb', False):
        wandb.finish()

if __name__ == "__main__":
    main()
