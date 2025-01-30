"""Training utilities for model training."""

import jax
import flax.linen as nn
import optax
from flax.training import train_state
from pathlib import Path
from typing import Any, Dict
import pickle

class TrainState(train_state.TrainState):
    """Custom train state with batch stats and dropout RNG."""
    batch_stats: Dict[str, Any]
    dropout_rng: jax.random.PRNGKey

def create_train_state(rng: jax.random.PRNGKey, model: nn.Module, optimizer: optax.GradientTransformation, config: dict) -> TrainState:
    """Create initial training state.
    
    Args:
        rng: PRNG key
        model: Flax model
        optimizer: Optax optimizer
        config: Training configuration
        
    Returns:
        state: Initial training state
    """
    # Initialize model
    variables = model.init(
        rng,
        jax.numpy.ones([1, *config.data.image_size, 3]),
        train=False,
    )
    
    # Split RNG for dropout
    dropout_rng = jax.random.fold_in(rng, 1)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        batch_stats=variables.get('batch_stats', {}),
        dropout_rng=dropout_rng,
    )

# Model version to prevent loading incompatible checkpoints
MODEL_VERSION = "2.0.0"  # Increment on architecture changes

def save_checkpoint(state: TrainState, checkpoint_dir: str):
    """Save model checkpoint as pickle file with version."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    with open(checkpoint_path / 'checkpoint.pkl', 'wb') as f:
        pickle.dump({
            'version': MODEL_VERSION,
            'params': state.params,
            'batch_stats': state.batch_stats,
        }, f)

def load_checkpoint(checkpoint_path: str, check_version: bool = True) -> Dict[str, Any]:
    """Load model checkpoint with optional version check.
    
    Args:
        checkpoint_path: Path to checkpoint file
        check_version: Whether to enforce version check (False for stage1 checkpoints)
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    if check_version:
        if 'version' not in checkpoint:
            raise ValueError(f"Checkpoint at {checkpoint_path} has no version info")
        if checkpoint['version'] != MODEL_VERSION:
            raise ValueError(
                f"Checkpoint version mismatch: expected {MODEL_VERSION}, "
                f"got {checkpoint['version']}"
            )
    
    # Handle legacy checkpoints without version info
    if isinstance(checkpoint, dict) and 'params' in checkpoint:
        return {
            'params': checkpoint['params'],
            'batch_stats': checkpoint.get('batch_stats', {}),
        }
    
    # For very old checkpoints that were just params
    return {
        'params': checkpoint,
        'batch_stats': {},
    }

def create_learning_rate_schedule(
    total_steps: int,
    learning_rate: float,
    warmup_steps: int,
) -> optax.Schedule:
    """Create learning rate schedule with warmup and cosine decay.
    
    Args:
        total_steps: Total number of training steps
        learning_rate: Peak learning rate
        warmup_steps: Number of warmup steps
        
    Returns:
        schedule: Learning rate schedule function
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    
    decay_fn = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps - warmup_steps,
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[warmup_steps],
    )

def create_optimizer(config: dict) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule and gradient clipping.
    
    Args:
        config: Training configuration
        
    Returns:
        optimizer: Configured optimizer
    """
    schedule = create_learning_rate_schedule(
        total_steps=config.training.total_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
    )
    
    return optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=config.training.weight_decay,
        ),
    )
