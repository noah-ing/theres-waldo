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

@hydra.main(config_path="../config", config_name="train")
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
    
    # Initialize model
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    state = create_train_state(
        init_rng,
        learning_rate=cfg.training.learning_rate,
        model_kwargs=dict(cfg.model),
    )
    
    # Training loop with modern practices
    best_val_loss = float('inf')
    train_loader = train_dataset.train_loader()
    
    for epoch in range(cfg.training.num_epochs):
        # Training
        train_metrics = []
        train_pbar = tqdm(range(cfg.training.steps_per_epoch), desc=f'Epoch {epoch+1}')
        
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
            
            with open(model_dir / 'best_model.pkl', 'wb') as f:
                import pickle
                pickle.dump(state, f)
            
            # Log best model to wandb
            wandb.save(str(model_dir / 'best_model.pkl'))
        
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_epoch_metrics['loss']:.4f}")
        print(f"Val Loss: {val_epoch_metrics['val_loss']:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
    
    wandb.finish()

if __name__ == '__main__':
    train()
