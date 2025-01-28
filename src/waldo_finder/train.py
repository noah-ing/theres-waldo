"""Advanced training script for Waldo detector with SOTA practices."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import hydra
from omegaconf import DictConfig
import wandb
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import common_utils

from waldo_finder.model import (
    create_train_state,
    train_step,
    eval_step,
)
from waldo_finder.data import WaldoDataset

class EarlyStopping:
    """Early stopping handler with patience and minimum improvement threshold."""
    
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

@hydra.main(config_path="../../config", config_name="train", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """Main training loop with modern ML practices.
    
    Args:
        cfg: Hydra configuration object
    """
    # Enhanced wandb initialization with more metadata
    wandb.init(
        project="waldo-finder",
        name=cfg.experiment_name,
        config=dict(cfg),
        tags=["SOTA", "ViT", "Object-Detection"],
        notes="Advanced training with mixed precision, EMA, and gradient accumulation"
    )
    
    # Set up data loaders
    train_dataset = WaldoDataset(
        data_dir=cfg.data.train_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        augment=True,
    )
    
    val_dataset = WaldoDataset(
        data_dir=cfg.data.val_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        augment=False,
    )
    
    print("Starting model initialization and JAX compilation (this may take a few minutes)...")
    
    # Initialize model
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    # Calculate training parameters
    steps_per_epoch = len(train_dataset.annotations) // cfg.training.batch_size
    num_train_steps = steps_per_epoch * cfg.training.num_epochs
    warmup_steps = steps_per_epoch * cfg.training.warmup_epochs
    
    print(f"Training schedule:")
    print(f"- {steps_per_epoch} steps per epoch")
    print(f"- {cfg.training.num_epochs} total epochs")
    print(f"- {cfg.training.warmup_epochs} warmup epochs ({warmup_steps} steps)")
    print(f"- {num_train_steps} total training steps")
    print(f"- {cfg.training.learning_rate} peak learning rate")
    print(f"- {cfg.training.batch_size} batch size")
    
    print("\nCreating train state...")
    state = create_train_state(
        init_rng,
        learning_rate=cfg.training.learning_rate,
        model_kwargs=dict(cfg.model),
        num_train_steps=num_train_steps,
        warmup_epochs=cfg.training.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    # Initialize advanced training components
    if cfg.training.mixed_precision:
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None
    
    if cfg.training.ema:
        ema_decay = cfg.training.ema_decay
        def ema_update(params, new_params):
            return jax.tree_map(
                lambda p1, p2: p1 * ema_decay + (1 - ema_decay) * p2,
                params, new_params
            )
        ema_params = state.params
    else:
        ema_update = None
        ema_params = None
    
    early_stopping = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta
    )
    
    print("\nStarting advanced training loop...")
    
    # Enhanced training loop with advanced practices
    best_val_loss = float('inf')
    train_loader = train_dataset.train_loader()
    grad_accumulation_steps = cfg.training.gradient_accumulation_steps
    
    for epoch in range(cfg.training.num_epochs):
        state = state.replace(dropout_rng=jax.random.fold_in(state.dropout_rng, epoch))
        # Enhanced training with gradient accumulation and mixed precision
        train_metrics = []
        accumulated_grads = None
        train_pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}')
        
        for step in train_pbar:
            batch = next(train_loader)
            rng, step_rng = jax.random.split(rng)
            
            # Mixed precision training step
            if dynamic_scale:
                state, metrics, dynamic_scale = train_step_mixed_precision(
                    state, batch, step_rng, dynamic_scale)
            else:
                state, metrics = train_step(state, batch, step_rng)
            
            # Update EMA parameters
            if ema_update is not None and (step + 1) % grad_accumulation_steps == 0:
                ema_params = ema_update(ema_params, state.params)
            
            train_metrics.append(metrics)
            
            # Enhanced progress bar with more metrics
            recent_metrics = train_metrics[-100:]
            avg_metrics = {
                k: float(np.mean([m[k] for m in recent_metrics]))
                for k in recent_metrics[0].keys()
            }
            train_pbar.set_postfix({
                'loss': f"{avg_metrics['loss']:.4f}",
                'giou_loss': f"{avg_metrics.get('giou_loss', 0):.4f}",
                'score_loss': f"{avg_metrics.get('score_loss', 0):.4f}"
            })
        
        # Calculate epoch metrics
        train_epoch_metrics = {
            k: float(np.mean([m[k] for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        
        # Enhanced validation with EMA if enabled
        val_metrics = []
        eval_params = ema_params if ema_update is not None else state.params
        
        for batch in val_dataset.val_loader():
            outputs = eval_step(state.replace(params=eval_params), batch)
            
            # Calculate comprehensive validation metrics
            val_loss = jnp.mean(jnp.abs(outputs['boxes'] - batch['boxes']))
            score_accuracy = jnp.mean(jnp.abs(outputs['scores'] - batch['scores']))
            val_metrics.append({
                'val_loss': val_loss,
                'val_score_accuracy': score_accuracy
            })
        
        val_epoch_metrics = {
            k: float(np.mean([m[k] for m in val_metrics]))
            for k in val_metrics[0].keys()
        }
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            **train_epoch_metrics,
            **val_epoch_metrics,
        })
        
        # Check early stopping
        if early_stopping(val_epoch_metrics['val_loss']):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
            
        # Enhanced model saving with more metadata
        if val_epoch_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_epoch_metrics['val_loss']
            
            # Enhanced model saving with EMA and training state
            model_dir = Path(cfg.training.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            from flax.serialization import to_bytes
            import pickle
            
            save_dict = {
                'params': to_bytes(eval_params),  # Save EMA params if enabled
                'opt_state': to_bytes(state.opt_state),
                'training_state': {
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'dynamic_scale': dynamic_scale.state_dict() if dynamic_scale else None,
                    'ema_params': to_bytes(ema_params) if ema is not None else None,
                }
            }
            
            with open(model_dir / 'model.pkl', 'wb') as f:
                pickle.dump(save_dict, f)
            
            # Enhanced wandb artifact logging
            artifact = wandb.Artifact(
                name=f"{cfg.experiment_name}-model",
                type="model",
                description=f"Best model checkpoint (val_loss: {best_val_loss:.4f})"
            )
            artifact.metadata = {
                'epoch': epoch + 1,
                'val_loss': float(best_val_loss),
                'train_loss': float(train_epoch_metrics['loss']),
                'architecture': {
                    'num_layers': cfg.model.num_layers,
                    'hidden_dim': cfg.model.hidden_dim,
                    'num_heads': cfg.model.num_heads,
                }
            }
            artifact.add_file(str(model_dir / 'model.pkl'))
            wandb.log_artifact(artifact)
        
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_epoch_metrics['loss']:.4f}")
        print(f"Val Loss: {val_epoch_metrics['val_loss']:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
    
    wandb.finish()

if __name__ == '__main__':
    train()
