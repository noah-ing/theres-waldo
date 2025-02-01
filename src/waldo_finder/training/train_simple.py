"""
Enhanced training script for Waldo detection with improved monitoring,
error handling, and resource management.
"""

import os
import sys
import torch
import hydra
import logging
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    DeviceStatsMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .simple_trainer import SimpleTripletTrainer
from ..data.scene_dataset import build_dataloader

logger = logging.getLogger(__name__)

def setup_callbacks(config: DictConfig) -> list:
    """Setup comprehensive training callbacks"""
    callbacks = [
        # Checkpoint saving
        ModelCheckpoint(
            dirpath=config.checkpoint.dirpath,
            filename=config.checkpoint.filename,
            save_top_k=config.checkpoint.save_top_k,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last,
            every_n_epochs=config.checkpoint.every_n_epochs
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='step'),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            min_delta=1e-4
        ),
        
        # Resource monitoring
        DeviceStatsMonitor(),
        
        # Rich progress bar
        RichProgressBar()
    ]
    
    return callbacks

def setup_loggers(config: DictConfig):
    """Setup multiple loggers for comprehensive tracking"""
    loggers = [
        # CSV logging
        CSVLogger(
            save_dir="logs",
            name="waldo_triplet",
            flush_logs_every_n_steps=100
        ),
        
        # TensorBoard logging
        TensorBoardLogger(
            save_dir="logs",
            name="waldo_triplet",
            log_graph=True
        )
    ]
    
    return loggers

def verify_gpu_availability():
    """Verify and log GPU availability and specifications"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_capability = torch.cuda.get_device_capability()
        
        logger.info(f"Using GPU: {device_name}")
        logger.info(f"GPU Memory: {memory_gb:.1f}GB")
        logger.info(f"CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}")
        
        return True
    else:
        logger.warning("No GPU detected! Training will be slow on CPU.")
        return False

def setup_training_strategy(config: DictConfig, has_gpu: bool):
    """Configure training strategy based on available resources"""
    if has_gpu and config.training.devices > 1:
        return DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
    return config.training.strategy

@hydra.main(config_path="../../../config", config_name="simple_train", version_base="1.1")
def main(config: DictConfig):
    """Main training function with enhanced error handling and monitoring"""
    try:
        logger.info("\n" + "="*80)
        logger.info("Starting Enhanced Waldo Detection Training")
        logger.info("="*80 + "\n")
        
        # Verify hardware
        has_gpu = verify_gpu_availability()
        
        # Create output directories
        os.makedirs(config.checkpoint.dirpath, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize model with mining strategy
        model = SimpleTripletTrainer(
            config=config,
            margin=config.model.margin,
            mining_strategy=config.model.mining_strategy
        )
        
        # Setup training infrastructure
        callbacks = setup_callbacks(config)
        loggers = setup_loggers(config)
        strategy = setup_training_strategy(config, has_gpu)
        
        # Configure trainer with enhanced monitoring
        trainer = pl.Trainer(
            max_epochs=config.optimizer.max_epochs,
            accelerator=config.training.accelerator,
            devices=config.training.devices,
            strategy=strategy,
            precision=config.training.precision,
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=config.training.log_every_n_steps,
            val_check_interval=config.training.val_check_interval,
            gradient_clip_val=config.optimizer.grad_clip,
            gradient_clip_algorithm=config.training.gradient_clip_algorithm,
            deterministic=True,
            benchmark=True
        )
        
        # Create dataloaders with enhanced configuration
        train_loader = build_dataloader(
            data_dir=config.data.data_dir,
            config=config,
            split='train',
            triplet_mining=True
        )
        
        val_loader = build_dataloader(
            data_dir=config.data.data_dir,
            config=config,
            split='val',
            triplet_mining=True
        )
        
        # Train model with progress tracking
        logger.info("Starting training...")
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Log training results
        best_model_score = trainer.checkpoint_callback.best_model_score
        logger.info("\nTraining completed successfully!")
        logger.info(f"Best validation loss: {best_model_score:.4f}")
        logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        
        # Save final metrics
        final_metrics = {
            'best_val_loss': float(best_model_score),
            'total_epochs': trainer.current_epoch,
            'global_step': trainer.global_step
        }
        
        return final_metrics
        
    except Exception as e:
        logger.error("\nError during training:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
