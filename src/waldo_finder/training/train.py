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
    RichProgressBar,
    StochasticWeightAveraging
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .trainer import WaldoTrainer
from ..data.scene_dataset import build_dataloader

logger = logging.getLogger(__name__)

def setup_callbacks(config: DictConfig, phase: str) -> list:
    """Setup comprehensive training callbacks for a specific phase"""
    # Create phase-specific checkpoint directory
    phase_dir = os.path.join(config.checkpoint.dirpath, phase)
    os.makedirs(phase_dir, exist_ok=True)
    
    callbacks = [
        # Checkpoint saving with phase-specific settings
        ModelCheckpoint(
            dirpath=phase_dir,
            filename=f"{phase}-{config.checkpoint.filename}",
            save_top_k=config.checkpoint.save_top_k,
            monitor=f"val_{phase}_loss",  # Phase-specific metric
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last,
            every_n_epochs=config.checkpoint.get('every_n_epochs', 1)
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='step'),
        
        # Early stopping with phase-specific patience
        EarlyStopping(
            monitor='val_loss',
            patience=config.training[phase].get('patience', 10),
            mode='min',
            min_delta=1e-4
        ),
        
        # Resource monitoring
        DeviceStatsMonitor(),
        
        # Rich progress bar with memory stats
        RichProgressBar(leave=True)
    ]
    
    # Add SWA for better generalization in later epochs
    if config.training[phase].get('swa_enabled', False):
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=0.8,  # Start at 80% of training
                swa_lrs=config.training[phase].learning_rate * 0.1
            )
        )
    
    return callbacks

def setup_loggers(config: DictConfig):
    """Setup multiple loggers for comprehensive tracking"""
    loggers = [
        # CSV logging with frequent updates
        CSVLogger(
            save_dir="logs",
            name="waldo_detection",
            flush_logs_every_n_steps=50  # More frequent flushing
        ),
        
        # TensorBoard logging
        TensorBoardLogger(
            save_dir="logs",
            name="waldo_detection",
            log_graph=True
        )
    ]
    
    return loggers

def verify_gpu_availability(config: DictConfig) -> bool:
    """Verify and log GPU availability and specifications"""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_capability = torch.cuda.get_device_capability()
            
            logger.info(f"Using GPU: {device_name}")
            logger.info(f"GPU Memory: {memory_gb:.1f}GB")
            logger.info(f"CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}")
            
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = config.hardware.get('benchmark', True)
            torch.backends.cudnn.deterministic = config.hardware.get('deterministic', False)
            
            return True
        else:
            logger.warning("No GPU detected! Training will be slow on CPU.")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU: {str(e)}")
        return False

def setup_training_strategy(config: DictConfig, has_gpu: bool):
    """Configure training strategy based on available resources"""
    if has_gpu and isinstance(config.hardware.devices, list) and len(config.hardware.devices) > 1:
        return DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True  # Optimize for static graphs
        )
    return "auto"  # Default to auto strategy for single GPU

@hydra.main(config_path="../../../config", config_name="test_run", version_base="1.1")
def main(config: DictConfig):
    """Main training function with enhanced error handling and monitoring"""
    try:
        logger.info("\n" + "="*80)
        logger.info("Starting Enhanced Waldo Detection Training")
        logger.info("="*80 + "\n")
        
        # Verify hardware and setup environment
        has_gpu = verify_gpu_availability(config)
        
        # Create output directories
        os.makedirs(config.checkpoint.dirpath, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Setup base training infrastructure
        strategy = setup_training_strategy(config, has_gpu)
        base_trainer_kwargs = {
            'accelerator': config.hardware.accelerator,
            'devices': config.hardware.devices,
            'strategy': strategy,
            'precision': config.hardware.precision,
            'gradient_clip_val': config.hardware.get('gradient_clip_val', 1.0),
            'gradient_clip_algorithm': "norm",
            'accumulate_grad_batches': config.hardware.get('accumulate_grad_batches', 1),
            'deterministic': config.hardware.get('deterministic', False),
            'benchmark': config.hardware.get('benchmark', True),
            'val_check_interval': 0.5,
            'enable_progress_bar': True,
            'enable_model_summary': True
        }

        # Phase 1: Pretraining with memory optimization
        if config.training.pretrain.enabled:
            logger.info("\n" + "="*40)
            logger.info("Phase 1: Pretraining")
            logger.info("="*40)
            
            pretrain_model = WaldoTrainer(config=config, pretrain=True)
            pretrain_trainer = pl.Trainer(
                **base_trainer_kwargs,
                max_epochs=config.training.pretrain.epochs,
                callbacks=setup_callbacks(config, "pretrain"),
                logger=setup_loggers(config),
                log_every_n_steps=config.logging.log_every_n_steps
            )
            
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
            
            pretrain_trainer.fit(pretrain_model, train_loader, val_loader)
            pretrain_ckpt = pretrain_trainer.checkpoint_callback.best_model_path
            
            # Clear memory after phase
            del train_loader, val_loader, pretrain_model
            if has_gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"Pretraining completed. Best checkpoint: {pretrain_ckpt}")
            
        # Phase 2: Contrastive Learning
        if config.training.contrastive.enabled:
            logger.info("\n" + "="*40)
            logger.info("Phase 2: Contrastive Learning")
            logger.info("="*40)
            
            # Load model from pretrain checkpoint with error handling
            if config.training.pretrain.enabled:
                if not os.path.exists(pretrain_ckpt):
                    raise ValueError(f"Pretrain checkpoint not found: {pretrain_ckpt}")
                try:
                    contrastive_model = WaldoTrainer.load_from_checkpoint(
                        pretrain_ckpt,
                        config=config,
                        pretrain=False,
                        phase='contrastive'  # Explicitly set phase
                    )
                    logger.info(f"Successfully loaded pretrain checkpoint: {pretrain_ckpt}")
                except Exception as e:
                    logger.error(f"Failed to load pretrain checkpoint: {str(e)}")
                    raise
            else:
                contrastive_model = WaldoTrainer(config=config, pretrain=False, phase='contrastive')
            
            contrastive_trainer = pl.Trainer(
                **base_trainer_kwargs,
                max_epochs=config.training.contrastive.epochs,
                callbacks=setup_callbacks(config, "contrastive"),
                logger=setup_loggers(config),
                log_every_n_steps=config.logging.log_every_n_steps
            )
            
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
            
            contrastive_trainer.fit(contrastive_model, train_loader, val_loader)
            contrastive_ckpt = contrastive_trainer.checkpoint_callback.best_model_path
            
            # Clear memory after phase
            del train_loader, val_loader, contrastive_model
            if has_gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"Contrastive learning completed. Best checkpoint: {contrastive_ckpt}")
            
        # Phase 3: Detection Training
        if config.training.detection.enabled:
            logger.info("\n" + "="*40)
            logger.info("Phase 3: Detection Training")
            logger.info("="*40)
            
            # Load model from appropriate checkpoint with error handling
            if config.training.contrastive.enabled or config.training.pretrain.enabled:
                ckpt_path = contrastive_ckpt if config.training.contrastive.enabled else pretrain_ckpt
                if not os.path.exists(ckpt_path):
                    raise ValueError(f"Checkpoint not found: {ckpt_path}")
                try:
                    detection_model = WaldoTrainer.load_from_checkpoint(
                        ckpt_path,
                        config=config,
                        pretrain=False,
                        phase='detection'  # Explicitly set phase
                    )
                    logger.info(f"Successfully loaded checkpoint: {ckpt_path}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {str(e)}")
                    raise
            else:
                detection_model = WaldoTrainer(config=config, pretrain=False, phase='detection')
            
            detection_trainer = pl.Trainer(
                **base_trainer_kwargs,
                max_epochs=config.training.detection.epochs,
                callbacks=setup_callbacks(config, "detection"),
                logger=setup_loggers(config),
                log_every_n_steps=config.logging.log_every_n_steps
            )
            
            train_loader = build_dataloader(
                data_dir=config.data.data_dir,
                config=config,
                split='train',
                triplet_mining=False
            )
            val_loader = build_dataloader(
                data_dir=config.data.data_dir,
                config=config,
                split='val',
                triplet_mining=False
            )
            
            detection_trainer.fit(detection_model, train_loader, val_loader)
            detection_ckpt = detection_trainer.checkpoint_callback.best_model_path
            
            # Clear memory after phase
            del train_loader, val_loader, detection_model
            if has_gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"Detection training completed. Best checkpoint: {detection_ckpt}")
        
        # Collect final metrics from all phases
        final_metrics = {
            'phases_completed': [],
            'checkpoints': {}
        }
        
        if config.training.pretrain.enabled:
            final_metrics['phases_completed'].append('pretrain')
            final_metrics['checkpoints']['pretrain'] = pretrain_ckpt
            
        if config.training.contrastive.enabled:
            final_metrics['phases_completed'].append('contrastive')
            final_metrics['checkpoints']['contrastive'] = contrastive_ckpt
            
        if config.training.detection.enabled:
            final_metrics['phases_completed'].append('detection')
            final_metrics['checkpoints']['detection'] = detection_ckpt
        
        logger.info("\nTraining Pipeline Completed Successfully!")
        logger.info("="*80)
        logger.info("Phases Completed: " + ", ".join(final_metrics['phases_completed']))
        logger.info("\nBest Checkpoints:")
        for phase, ckpt in final_metrics['checkpoints'].items():
            logger.info(f"{phase.capitalize()}: {ckpt}")
        logger.info("="*80)
        
        return final_metrics
        
    except Exception as e:
        logger.error("\nError during training:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        raise
    finally:
        # Final cleanup
        if has_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
