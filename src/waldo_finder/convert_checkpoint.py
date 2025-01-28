"""Convert checkpoint model to pickle format for inference."""

from pathlib import Path
import jax
import tensorflow as tf
from flax.serialization import msgpack_serialize
from waldo_finder.model import create_train_state

def convert_checkpoint(ckpt_dir: str, output_path: str):
    """Convert checkpoint to pickle format.
    
    Args:
        ckpt_dir: Directory containing checkpoint files
        output_path: Path to save pickle file
    """
    print(f"Loading checkpoint from {ckpt_dir}")
    
    # Initialize model with same configuration as training
    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        rng,
        learning_rate=1e-4,  # Doesn't matter for inference
        model_kwargs={
            'num_heads': 12,
            'num_layers': 12,
            'hidden_dim': 768,
            'mlp_dim': 3072,
            'dropout_rate': 0.1,
        }
    )
    
    # Load checkpoint
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")
        
    print(f"Found checkpoint: {ckpt_path}")
    
    # Create checkpoint reader
    reader = tf.train.load_checkpoint(ckpt_path)
    
    # Serialize parameters using msgpack
    serialized_params = msgpack_serialize(state.params)
    
    # Save serialized parameters
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(serialized_params)
    
    print(f"Saved converted model to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', help='Directory containing checkpoint files')
    parser.add_argument('output_path', help='Path to save pickle file')
    args = parser.parse_args()
    
    convert_checkpoint(args.ckpt_dir, args.output_path)
