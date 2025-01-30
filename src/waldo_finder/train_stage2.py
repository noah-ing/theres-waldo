"""Stage 2 training script for scene detection with transfer learning."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from pathlib import Path
import hydra
from omegaconf import DictConfig
import wandb

from waldo_finder.model import SceneDetector, WaldoClassifier
from waldo_finder.data import create_scene_dataset
from waldo_finder.losses import compute_total_loss
import shutil
from waldo_finder.train_utils import (
    TrainState,
    create_train_state,
    save_checkpoint,
    load_checkpoint,
    create_optimizer,
    MODEL_VERSION,
)

def load_stage1_model(checkpoint_path: str) -> WaldoClassifier:
    """Load pre-trained stage 1 model for feature extraction."""
    classifier = WaldoClassifier()
    variables = load_checkpoint(checkpoint_path, check_version=False)  # Don't check version for stage1
    return classifier.bind({
        'params': variables['params'],
        'batch_stats': variables['batch_stats'],
    })

def train_step(state: TrainState, batch: dict, config: DictConfig, version: str = "2.0.0"):
    """Single training step with gradient computation."""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    
    def loss_fn(params):
        variables = {
            'params': params, 
            'batch_stats': state.batch_stats,
        }
        (pred_boxes, pred_conf), new_batch_stats = state.apply_fn(
            variables,
            batch['image'],
            train=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng},
        )
        loss, metrics = compute_total_loss(
            pred_boxes,
            batch['boxes'],
            pred_conf,
            batch['confidence'],
            config,
        )
        return loss, (metrics, new_batch_stats)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_batch_stats)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_batch_stats,
        dropout_rng=new_dropout_rng,
    )
    return state, loss, metrics

def validate_step(state: TrainState, batch: dict, config: DictConfig, version: str = "2.0.0"):
    """Single validation step."""
    variables = {
        'params': state.params, 
        'batch_stats': state.batch_stats,
    }
    (pred_boxes, pred_conf), _ = state.apply_fn(
        variables,
        batch['image'],
        train=False,
        mutable=['batch_stats'],
        rngs={'dropout': state.dropout_rng},
    )
    loss, metrics = compute_total_loss(
        pred_boxes,
        batch['boxes'],
        pred_conf,
        batch['confidence'],
        config,
    )
    return loss, metrics

@hydra.main(config_path="../../config", config_name="stage2_train")
def main(config: DictConfig):
    """Main training function."""
    print("\nWARNING: Stage 2 model architecture has been updated to version 2.0.0")
    print("This version uses a new coordinate parameterization and is not compatible")
    print("with previous checkpoints. Clearing checkpoint directory...\n")
    
    # Clear checkpoint directory
    checkpoint_dir = Path(hydra.utils.get_original_cwd()) / config.training.checkpoint_dir
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    # Initialize wandb if enabled
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.run_name,
            mode=config.wandb.mode,
            config=dict(config),
        )
        print("Initialized wandb for experiment tracking")
    else:
        print("Running without wandb experiment tracking")
    
    # Get paths relative to project root
    orig_cwd = hydra.utils.get_original_cwd()
    train_path = Path(orig_cwd) / config.data.train_path
    val_path = Path(orig_cwd) / config.data.val_path
    
    # Load pre-trained stage 1 model
    checkpoint_path = Path(hydra.utils.get_original_cwd()) / config.model.stage1_checkpoint
    feature_extractor = load_stage1_model(str(checkpoint_path))
    
    # Initialize scene detector with pre-trained features
    model = SceneDetector(
        feature_extractor=feature_extractor,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_dim=config.model.mlp_dim,
        dropout_rate=config.model.dropout_rate
    )
    
    # Create optimizer and training state
    optimizer = create_optimizer(config)
    rng = jax.random.PRNGKey(config.training.seed)
    state = create_train_state(rng, model, optimizer, config)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        # Training - create fresh dataset for each epoch
        train_ds = create_scene_dataset(
            train_path,
            config.data.batch_size,
            train=True,
        )
        
        for step, batch in enumerate(train_ds):
            state, loss, metrics = train_step(state, batch, config, version=MODEL_VERSION)
            
            # Print training progress with detailed metrics
            print(f"\nEpoch {epoch}, Step {step}")
            print(f"  Loss: {loss:.4f}")
            print("  Metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {float(v):.4f}")
            
            if config.wandb.enabled:
                wandb.log({
                    'train/loss': loss,
                    **{f'train/{k}': v for k, v in metrics.items()}
                })
        
        # Validation - create fresh dataset for each epoch
        val_ds = create_scene_dataset(
            val_path,
            config.data.batch_size,
            train=False,
        )
        
        val_losses = []
        val_metrics_list = []
        
        # Run validation
        for batch in val_ds:
            loss, metrics = validate_step(state, batch, config, version=MODEL_VERSION)
            val_losses.append(loss)
            val_metrics_list.append(metrics)
        
        # Only compute validation metrics if we have validation samples
        if val_losses:
            val_loss = float(jnp.mean(jnp.array(val_losses)))
            avg_metrics = {}
            if val_metrics_list:
                metric_keys = val_metrics_list[0].keys()
                avg_metrics = {
                    k: float(jnp.mean(jnp.array([m[k] for m in val_metrics_list])))
                    for k in metric_keys
                }
            
            # Print validation results with detailed metrics
            print(f"\nValidation Results:")
            print(f"  Loss: {val_loss:.4f}")
            if avg_metrics:
                print("  Metrics:")
                for k, v in avg_metrics.items():
                    print(f"    {k}: {v:.4f}")
        else:
            print("\nNo validation samples available")
            val_loss = float('inf')
            avg_metrics = {}
        
        if config.wandb.enabled:
            wandb.log({
                'val/loss': val_loss,
                **{f'val/{k}': v for k, v in avg_metrics.items()}
            })
        
        # Early stopping
        if val_loss < best_val_loss - config.training.early_stopping.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(state, config.training.checkpoint_dir)
        else:
            patience_counter += 1
            
        if patience_counter >= config.training.early_stopping.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    if config.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
