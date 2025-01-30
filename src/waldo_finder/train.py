"""Training script for Waldo detector optimized for CPU."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU-only mode before JAX import

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
    train_step_mixed_precision,
    eval_step,
    compute_loss,
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
    # Initialize wandb if available and not disabled
    use_wandb = False
    try:
        if os.environ.get('WANDB_MODE') != 'disabled':
            wandb.init(
                project="waldo-finder",
                name=cfg.experiment_name,
                config=dict(cfg),
                tags=["CPU-Training", "ViT", "Object-Detection"],
                notes="CPU-optimized training configuration"
            )
            use_wandb = True
    except:
        print("Wandb not available, continuing without logging")
    
    # Set up data loaders with semi-supervised support
    train_dataset = WaldoDataset(
        data_dir=cfg.data.train_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        augment=True,
        use_unlabeled=True,
        box_format='cxcywh'  # Use consistent format
    )
    
    val_dataset = WaldoDataset(
        data_dir=cfg.data.val_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        augment=False,
        use_unlabeled=False,  # Validation only on labeled data
        box_format='cxcywh'
    )
    
    print("Starting initialization...")
    print("1/5: Setting up JAX...")
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    print("2/5: Creating initial state...")
    # Calculate training parameters considering all available images
    steps_per_epoch = (len(train_dataset.image_paths) + cfg.training.batch_size - 1) // cfg.training.batch_size
    num_train_steps = steps_per_epoch * cfg.training.num_epochs
    warmup_steps = steps_per_epoch * cfg.training.warmup_epochs
    
    # Create model kwargs with data config
    model_kwargs = {
        **dict(cfg.model),
        'data': dict(cfg.data)  # Include data config for image size
    }
    
    state = create_train_state(
        init_rng,
        learning_rate=cfg.training.learning_rate,
        model_kwargs=model_kwargs,
        num_train_steps=num_train_steps,
        warmup_epochs=cfg.training.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    print("3/5: Pre-compiling JAX functions...")
    # Pre-compile JAX functions with dummy data
    dummy_batch = {
        'image': np.zeros((cfg.training.batch_size, *cfg.data.image_size, 3), dtype=np.float32),
        'boxes': np.zeros((cfg.training.batch_size, 4), dtype=np.float32),
        'scores': np.ones((cfg.training.batch_size, 1), dtype=np.float32),
        'is_labeled': np.ones((cfg.training.batch_size,), dtype=bool)
    }
    dummy_rng = jax.random.PRNGKey(0)
    
    # Compile steps ahead of time with proper RNG keys
    print("  - Compiling train_step...")
    _ = jax.jit(train_step)(state, dummy_batch, dummy_rng)
    print("  - Compiling eval_step...")
    _ = jax.jit(eval_step)(state.replace(dropout_rng=dummy_rng), dummy_batch)
    print("4/5: JAX compilation complete")
    
    print("5/5: Preparing data loaders...")
    
    print(f"\nTraining schedule:")
    print(f"- {steps_per_epoch} steps per epoch")
    print(f"- {cfg.training.num_epochs} total epochs")
    print(f"- {cfg.training.warmup_epochs} warmup epochs ({warmup_steps} steps)")
    print(f"- {num_train_steps} total training steps")
    print(f"- {cfg.training.learning_rate} peak learning rate")
    print(f"- {cfg.training.batch_size} batch size")
    print(f"- {len(train_dataset.image_paths)} total images")
    print(f"- {len(train_dataset.labeled_images)} labeled images")
    
    # Temporarily disable mixed precision until we get stable training
    dynamic_scale = None
    
    if cfg.training.ema:
        ema_decay = cfg.training.ema_decay
        def ema_update(params, new_params):
            return jax.tree.map(
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
        train_pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}', ncols=80)
        
        for step in train_pbar:
            try:
                batch = next(train_loader)
            except (StopIteration, ValueError) as e:
                print(f"\nWarning: Batch loading failed - {str(e)}")
                continue
            rng, step_rng = jax.random.split(rng)
            
            # Handle labeled and unlabeled data separately
            labeled_mask = batch.pop('is_labeled')
            step_metrics = {}
            
            if np.any(labeled_mask):
                # Process labeled data
                labeled_batch = {
                    'image': batch['image'][labeled_mask],
                    'boxes': batch['boxes'][labeled_mask],
                    'scores': batch['scores'][labeled_mask]
                }
                if dynamic_scale:
                    state, metrics, dynamic_scale = train_step_mixed_precision(
                        state, labeled_batch, step_rng, dynamic_scale)
                else:
                    state, metrics = train_step(state, labeled_batch, step_rng)
                
                # Track labeled metrics
                step_metrics.update({f'labeled_{k}': v for k, v in metrics.items()})
            
            if np.any(~labeled_mask) and epoch >= cfg.training.warmup_epochs:
                # Process unlabeled data with pseudo-labeling
                unlabeled_batch = {
                    'image': batch['image'][~labeled_mask],
                    'boxes': batch['boxes'][~labeled_mask],
                    'scores': batch['scores'][~labeled_mask]
                }
                
                # Generate pseudo-labels using EMA model
                if ema_params is not None:
                    # Get predictions from EMA model
                    pseudo_outputs = eval_step(
                        state.replace(params=ema_params), 
                        unlabeled_batch
                    )
                    
                    # Filter high-confidence predictions
                    conf_threshold = 0.9
                    high_conf_mask = pseudo_outputs['scores'] > conf_threshold
                    
                    if np.any(high_conf_mask):
                        # Create batch with pseudo-labels
                        pseudo_batch = {
                            'image': unlabeled_batch['image'][high_conf_mask],
                            'boxes': pseudo_outputs['boxes'][high_conf_mask],
                            'scores': pseudo_outputs['scores'][high_conf_mask]
                        }
                        
                        # Train on pseudo-labeled data with reduced weight
                        if dynamic_scale:
                            state, pseudo_metrics, dynamic_scale = train_step_mixed_precision(
                                state, pseudo_batch, step_rng, dynamic_scale)
                        else:
                            state, pseudo_metrics = train_step(state, pseudo_batch, step_rng)
                        
                        # Track pseudo-label metrics
                        step_metrics.update({f'pseudo_{k}': v * 0.5 for k, v in pseudo_metrics.items()})
            
            # Update EMA parameters if we processed any data
            if step_metrics and ema_update is not None and (step + 1) % grad_accumulation_steps == 0:
                ema_params = ema_update(ema_params, state.params)
            
            # Only append metrics if we processed some data
            if step_metrics:
                train_metrics.append(step_metrics)
            
            # Clean progress output with safe metric averaging
            recent_metrics = train_metrics[-100:]
            if recent_metrics:
                # Handle each metric separately to avoid shape issues
                avg_metrics = {}
                for k in recent_metrics[0].keys():
                    try:
                        values = [float(m[k]) for m in recent_metrics]  # Convert to float immediately
                        avg_metrics[k] = float(np.mean(values))
                    except (ValueError, TypeError) as e:
                        print(f"\nWarning: Could not average metric {k}: {str(e)}")
                        avg_metrics[k] = 0.0
                
                train_pbar.set_postfix({
                    'L': f"{avg_metrics.get('labeled_loss', 0.0):.3f}",
                    'G': f"{avg_metrics.get('labeled_giou_loss', 0.0):.3f}",
                    'B': f"{avg_metrics.get('labeled_l1_loss', 0.0):.3f}",
                    'C': f"{avg_metrics.get('labeled_score_loss', 0.0):.3f}",
                    'P': f"{avg_metrics.get('pseudo_loss', 0.0):.3f}"
                }, refresh=True)
        
        # Calculate epoch metrics safely with explicit float conversion
        train_epoch_metrics = {}
        if train_metrics:
            for k in train_metrics[0].keys():
                try:
                    values = [float(m[k]) for m in train_metrics]  # Convert to float immediately
                    train_epoch_metrics[k] = float(np.mean(values))
                except (ValueError, TypeError) as e:
                    print(f"\nWarning: Could not average epoch metric {k}: {str(e)}")
                    train_epoch_metrics[k] = 0.0
        
        # Precision-focused validation with comprehensive metrics
        val_metrics = []
        eval_params = ema_params if ema_update is not None else state.params
        valid_batches = 0
        
        try:
            for batch in val_dataset.val_loader():
                try:
                    outputs = eval_step(state.replace(params=eval_params), batch)
                    
                    # Use same loss computation as training for consistency
                    val_loss, val_batch_metrics = compute_loss(
                        eval_params,
                        batch,
                        state.replace(params=eval_params),
                        state.dropout_rng
                    )
                    
                    # Track validation metrics
                    # Convert metrics to scalars immediately
                    scalar_metrics = {}
                    for k, v in val_batch_metrics.items():
                        try:
                            scalar_metrics[f"val_{k}"] = float(jnp.mean(v))
                        except (ValueError, TypeError) as e:
                            print(f"\nWarning: Could not convert validation metric {k} to scalar: {str(e)}")
                            scalar_metrics[f"val_{k}"] = 0.0
                    val_metrics.append(scalar_metrics)
                    valid_batches += 1
                except Exception as batch_e:
                    print(f"\nWarning: Validation batch failed - {str(batch_e)}")
                    continue
            
            # Calculate validation metrics safely with explicit float conversion
            val_epoch_metrics = {}
            if val_metrics:
                print(f"\nProcessed {valid_batches} valid validation batches")
                for k in val_metrics[0].keys():
                    try:
                        values = [float(m[k]) for m in val_metrics]  # Convert to float immediately
                        val_epoch_metrics[k] = float(np.mean(values))
                    except (ValueError, TypeError) as e:
                        print(f"\nWarning: Could not average validation metric {k}: {str(e)}")
                        val_epoch_metrics[k] = 0.0
            else:
                print("\nWarning: No valid validation batches collected")
                # Use training metrics as fallback, but mark them clearly
                val_epoch_metrics = {
                    f"val_{k}": float(v) for k, v in train_epoch_metrics.items()
                }
                print("Using training metrics as validation fallback")
        except Exception as e:
            print(f"\nWarning: Validation loop failed - {str(e)}")
            # Use training metrics as fallback, but mark them clearly
            val_epoch_metrics = {
                f"val_{k}": float(v) for k, v in train_epoch_metrics.items()
            }
            print("Using training metrics as validation fallback")
        
        # Simple, effective logging
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **train_epoch_metrics,
                **val_epoch_metrics,
                'learning_rate': cfg.training.learning_rate  # Use configured learning rate
            })
        
        # Check early stopping with safe metric access
        val_loss = val_epoch_metrics.get('val_loss')
        if val_loss is not None:
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        else:
            print("\nWarning: Skipping early stopping check - no validation loss available")
            
        # Enhanced model saving with safe metric access
        val_loss = val_epoch_metrics.get('val_loss')
        if val_loss is not None and val_loss < best_val_loss:
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
                    'ema_params': to_bytes(ema_params) if ema_update is not None else None,
                }
            }
            
            with open(model_dir / 'model.pkl', 'wb') as f:
                pickle.dump(save_dict, f)
            
            # Log artifacts if wandb is available
            if use_wandb:
                artifact = wandb.Artifact(
                    name=f"{cfg.experiment_name}-model",
                    type="model",
                    description=f"Best model checkpoint (val_loss: {best_val_loss:.4f})"
                )
                artifact.metadata = {
                    'epoch': epoch + 1,
                    'val_loss': float(best_val_loss),
                    'train_loss': float(train_epoch_metrics.get('labeled_loss', 0.0)),
                    'architecture': {
                        'num_layers': cfg.model.num_layers,
                        'hidden_dim': cfg.model.hidden_dim,
                        'num_heads': cfg.model.num_heads,
                    }
                }
                artifact.add_file(str(model_dir / 'model.pkl'))
                wandb.log_artifact(artifact)
        
        # Clean epoch summary focused on finding Waldo
        print(f"\n{'='*30}")
        print(f"Epoch {epoch+1}")
        print(f"{'='*30}")
        
        # Pattern recognition with safe metric access
        conf_train = 1 - train_epoch_metrics.get('labeled_score_loss', 0)
        conf_val = 1 - val_epoch_metrics.get('val_score_loss', 0)
        print(f"Finding Waldo: {conf_val*100:.1f}% confident (train: {conf_train*100:.1f}%)")
        
        # Box quality (secondary)
        iou_val = 1 - val_epoch_metrics.get('val_giou_loss', 0)
        print(f"Box Quality:   {iou_val*100:.1f}% accurate")
        
        # Pseudo-label stats (if available)
        if 'pseudo_score_loss' in train_epoch_metrics:
            pseudo_conf = 1 - train_epoch_metrics['pseudo_score_loss']
            pseudo_iou = 1 - train_epoch_metrics['pseudo_giou_loss']
            print(f"Pseudo Labels: {pseudo_conf*100:.1f}% confident, {pseudo_iou*100:.1f}% accurate")
        
        # Model improvement
        if val_epoch_metrics['val_loss'] < best_val_loss:
            improvement = (best_val_loss - val_epoch_metrics['val_loss']) / best_val_loss * 100
            print(f"Improved:      {improvement:.1f}%")
        
        print(f"{'='*30}\n")
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    train()
