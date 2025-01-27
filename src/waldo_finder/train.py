"""Training script for Waldo detector with modern ML practices."""

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import wandb
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np

from waldo_finder.model import (
    create_train_state,
    train_step,
    eval_step,
)
from waldo_finder.data import WaldoDataset

@hydra.main(config_path="../../config", config_name="train", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """Main training loop with modern ML practices.
    
    Args:
        cfg: Hydra configuration object
    """
    # Initialize wandb
    wandb.init(
        project="waldo-finder",
        name=cfg.experiment_name,
        config=dict(cfg),
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
    
    print("\nStarting training loop...")
    
    # Training loop with modern practices
    best_val_loss = float('inf')
    train_loader = train_dataset.train_loader()
    
    for epoch in range(cfg.training.num_epochs):
        # Training
        train_metrics = []
        train_pbar = tqdm(range(steps_per_epoch), desc=f'Epoch {epoch+1}')
        
        for _ in train_pbar:
            batch = next(train_loader)
            rng, step_rng = jax.random.split(rng)
            
            state, metrics = train_step(state, batch, step_rng)
            train_metrics.append(metrics)
            
            # Update progress bar
            avg_loss = np.mean([m['loss'] for m in train_metrics[-100:]])
            train_pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Calculate epoch metrics
        train_epoch_metrics = {
            k: float(np.mean([m[k] for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        
        # Validation
        val_metrics = []
        for batch in val_dataset.val_loader():
            outputs = eval_step(state, batch)
            
            # Calculate validation metrics
            val_loss = jnp.mean(jnp.abs(outputs['boxes'] - batch['boxes']))
            val_metrics.append({'val_loss': val_loss})
        
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
        
        # Save best model
        if val_epoch_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_epoch_metrics['val_loss']
            
            # Save model using Flax serialization
            model_dir = Path(cfg.training.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert JAX arrays to numpy and save
            from flax.serialization import to_bytes, from_bytes
            import pickle
            
            serialized_params = to_bytes(state.params)
            serialized_opt_state = to_bytes(state.opt_state)
            
            with open(model_dir / 'model.pkl', 'wb') as f:
                pickle.dump({
                    'params': serialized_params,
                    'opt_state': serialized_opt_state,
                }, f)
            
            # Log best model to wandb using artifact
            artifact = wandb.Artifact(
                name=f"{cfg.experiment_name}-model",
                type="model",
                description="Best model checkpoint"
            )
            artifact.add_file(str(model_dir / 'model.pkl'))
            wandb.log_artifact(artifact)
        
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_epoch_metrics['loss']:.4f}")
        print(f"Val Loss: {val_epoch_metrics['val_loss']:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
    
    wandb.finish()

if __name__ == '__main__':
    train()
