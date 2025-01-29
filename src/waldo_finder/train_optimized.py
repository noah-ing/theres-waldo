"""Enhanced training script with advanced techniques for Waldo detector."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU-only mode

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
import optax

from waldo_finder.model_optimized import (
    create_optimized_train_state,
    compute_optimized_loss,
    EnhancedWaldoDetector
)
from waldo_finder.data import WaldoDataset

class EnhancedEarlyStopping:
    """Enhanced early stopping with more sophisticated criteria."""
    
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = 'min',
        baseline: Optional[float] = None,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode not in ['min', 'max']:
            raise ValueError(f"mode {mode} is unknown")
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta *= 1 if mode == 'min' else -1
    
    def __call__(self, current: float, weights: Optional[Dict] = None) -> bool:
        if self.baseline is not None and self.monitor_op(self.baseline, current):
            return True
            
        if self.best_score is None:
            self.best_score = current
            if self.restore_best_weights:
                self.best_weights = weights
        elif self.monitor_op(current - self.min_delta, self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = current
            if self.restore_best_weights:
                self.best_weights = weights
            self.counter = 0
        return False

@hydra.main(config_path="../../config", config_name="train_optimized", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """Enhanced training loop with advanced ML practices."""
    
    # Initialize wandb if available
    use_wandb = False
    try:
        if os.environ.get('WANDB_MODE') != 'disabled':
            wandb.init(
                project="waldo-finder",
                name=cfg.experiment_name,
                config=dict(cfg),
                tags=["CPU-Training", "ViT", "Enhanced-Training"],
                notes="Advanced training configuration with regularization"
            )
            use_wandb = True
    except:
        print("Wandb not available, continuing without logging")
    
    # Enhanced dataset loading with augmentation
    train_dataset = WaldoDataset(
        data_dir=cfg.data.train_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        augment=True
    )
    
    val_dataset = WaldoDataset(
        data_dir=cfg.data.val_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        augment=False
    )
    
    print("\nInitializing enhanced training pipeline...")
    print("1/5: Setting up JAX with optimizations...")
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    print("2/5: Creating enhanced model state...")
    steps_per_epoch = len(train_dataset.annotations) // cfg.training.batch_size
    num_train_steps = steps_per_epoch * cfg.training.num_epochs
    warmup_steps = steps_per_epoch * cfg.training.warmup_epochs
    
    # Create learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.training.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_train_steps,
        end_value=cfg.training.learning_rate * 0.01
    )
    
    state = create_optimized_train_state(
        init_rng,
        learning_rate=cfg.training.learning_rate,
        model_kwargs=dict(cfg.model),
        num_train_steps=num_train_steps,
        warmup_steps=warmup_steps
    )
    
    print("3/5: Pre-compiling JAX functions...")
    # Create dummy batch with multi-box support
    max_boxes_per_image = 6  # Based on our dataset (image 36.jpg has 6 boxes)
    dummy_batch = {
        'image': np.zeros((cfg.training.batch_size, *cfg.data.image_size, 3), dtype=np.float32),
        'boxes': np.zeros((cfg.training.batch_size, max_boxes_per_image, 4), dtype=np.float32),
        'scores': np.zeros((cfg.training.batch_size, max_boxes_per_image), dtype=np.float32),  # Remove extra dimension
    }
    # Set first box in each image as valid
    dummy_batch['scores'][:, 0] = 1.0  # Updated indexing
    
    # Enhanced training step with multi-box support and box constraints
    @jax.jit
    def enhanced_train_step(state, batch, rng):
        def loss_fn(params):
            return compute_optimized_loss(
                params, batch, state, rng,
                consistency_weight=cfg.training.regularization.consistency_weight,
                size_target=cfg.training.box_constraints.size_target,
                size_penalty=cfg.training.box_constraints.size_penalty
            )
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        
        # Apply gradient clipping
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -cfg.training.regularization.gradient_clip_norm,
                             cfg.training.regularization.gradient_clip_norm),
            grads
        )
        
        # Update state
        state = state.apply_gradients(grads=grads)
        
        # Update EMA parameters if enabled
        if cfg.training.ema:
            ema_decay = cfg.training.ema_decay
            if state.ema_params is None:
                state = state.replace(ema_params=state.params)
            else:
                new_ema = jax.tree_util.tree_map(
                    lambda ema, p: ema * ema_decay + (1 - ema_decay) * p,
                    state.ema_params,
                    state.params
                )
                state = state.replace(ema_params=new_ema)
        
        return state, metrics
    
    # Enhanced evaluation step with deterministic mode
    @jax.jit
    def enhanced_eval_step(state, batch):
        params = state.ema_params if state.ema_params is not None else state.params
        outputs = state.apply_fn(
            {'params': params},
            batch['image'],
            training=False,
            deterministic=True,  # Explicitly set deterministic mode
            augment=False
        )
        return outputs
    
    print("4/5: Compiling training functions...")
    _ = enhanced_train_step(state, dummy_batch, jax.random.PRNGKey(0))
    _ = enhanced_eval_step(state, dummy_batch)
    
    print("5/5: Setting up training monitors...")
    
    # Enhanced early stopping
    early_stopping = EnhancedEarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        mode=cfg.training.early_stopping.mode,
        baseline=cfg.training.early_stopping.baseline,
        restore_best_weights=cfg.training.early_stopping.restore_best_weights
    )
    
    print(f"\nEnhanced training configuration:")
    print(f"- {steps_per_epoch} steps per epoch")
    print(f"- {cfg.training.num_epochs} total epochs")
    print(f"- {warmup_steps} warmup steps")
    print(f"- {cfg.training.batch_size} batch size")
    print(f"- {cfg.training.gradient_accumulation_steps}x gradient accumulation")
    print(f"- {len(train_dataset.annotations)} training images")
    
    print("\nStarting enhanced training loop...")
    
    best_val_loss = float('inf')
    train_loader = train_dataset.train_loader()
    
    for epoch in range(cfg.training.num_epochs):
        # Training phase with enhanced monitoring
        train_metrics = []
        train_pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}')
        
        for step in train_pbar:
            batch = next(train_loader)
            rng, step_rng = jax.random.split(rng)
            
            # Enhanced training step
            state, metrics = enhanced_train_step(state, batch, step_rng)
            train_metrics.append(metrics)
            
            # Update progress bar with detailed metrics
            if step % cfg.training.monitoring.log_every_n_steps == 0:
                recent_metrics = train_metrics[-100:]
                avg_metrics = {
                    k: float(np.mean([m[k] for m in recent_metrics]))
                    for k in recent_metrics[0].keys()
                }
                train_pbar.set_postfix({
                    'loss': f"{avg_metrics['loss']:.4f}",
                    'giou': f"{avg_metrics['giou_loss']:.4f}",
                    'center': f"{avg_metrics['center_loss']:.4f}",
                    'size': f"{avg_metrics['size_loss']:.4f}",
                    'penalty': f"{avg_metrics['size_penalty_loss']:.4f}"
                })
        
        # Calculate epoch metrics
        train_epoch_metrics = {
            k: float(np.mean([m[k] for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        
        # Enhanced validation with EMA parameters
        val_metrics = []
        for batch in val_dataset.val_loader():
            outputs = enhanced_eval_step(state, batch)
            val_loss = compute_optimized_loss(
                state.params, batch, state, rng,
                consistency_weight=0.0  # Disable consistency loss for validation
            )[1]
            val_metrics.append(val_loss)
        
        val_epoch_metrics = {
            f"val_{k}": float(np.mean([m[k] for m in val_metrics]))
            for k in val_metrics[0].keys()
        }
        
        # Log metrics
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **train_epoch_metrics,
                **val_epoch_metrics,
                'learning_rate': float(lr_schedule(state.step))  # Get learning rate from schedule directly
            })
        
        # Enhanced model saving
        val_loss = val_epoch_metrics['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model with comprehensive metadata
            model_dir = Path(cfg.training.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            from flax.serialization import to_bytes
            import pickle
            
            save_dict = {
                'params': to_bytes(state.ema_params or state.params),
                'opt_state': to_bytes(state.opt_state),
                'training_state': {
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'ema_params': to_bytes(state.ema_params) if state.ema_params else None
                },
                'config': dict(cfg)
            }
            
            with open(model_dir / 'model_optimized.pkl', 'wb') as f:
                pickle.dump(save_dict, f)
            
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"{cfg.experiment_name}-model-optimized",
                    type="model",
                    description=f"Best model checkpoint (val_loss: {best_val_loss:.4f})"
                )
                artifact.add_file(str(model_dir / 'model_optimized.pkl'))
                wandb.log_artifact(artifact)
        
        # Check early stopping with enhanced criteria
        if early_stopping(val_loss, state.params):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            if early_stopping.restore_best_weights:
                state = state.replace(params=early_stopping.best_weights)
            break
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_epoch_metrics['loss']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    train()
