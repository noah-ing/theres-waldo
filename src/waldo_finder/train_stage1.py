"""Stage 1 training: Learn Waldo's appearance from cropped examples."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU-only mode

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import hydra
from omegaconf import DictConfig
import wandb
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import numpy as np
from flax.training import train_state
import optax
import cv2
import pickle

class Stage1Dataset:
    """Dataset for Waldo appearance learning."""
    
    def __init__(self, 
                 data_dir: str,
                 image_size: Tuple[int, int] = (128, 128),
                 batch_size: int = 16,
                 augment: bool = True,
                 val_split: float = 0.2):
        """Initialize dataset for binary classification."""
        # Convert relative path to absolute using Hydra's original working directory
        original_cwd = Path(hydra.utils.get_original_cwd())
        self.data_dir = (original_cwd / data_dir).resolve()
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        
        # Load all image paths
        waldo_dir = self.data_dir / "waldo"
        negative_dir = self.data_dir / "negative"
        
        self.waldo_images = list(waldo_dir.glob("*.jpg"))
        self.negative_images = list(negative_dir.glob("*.jpg"))
        
        if len(self.waldo_images) == 0:
            raise ValueError("No Waldo images found in dataset")
        if len(self.negative_images) == 0:
            raise ValueError("No negative examples found in dataset")
            
        # Split into train/val
        np.random.seed(42)
        n_val_waldo = max(int(len(self.waldo_images) * val_split), 1)
        n_val_neg = max(int(len(self.negative_images) * val_split), 1)
        
        self.val_waldo = np.random.choice(self.waldo_images, n_val_waldo, replace=False)
        self.train_waldo = [x for x in self.waldo_images if x not in self.val_waldo]
        
        self.val_negative = np.random.choice(self.negative_images, n_val_neg, replace=False)
        self.train_negative = [x for x in self.negative_images if x not in self.val_negative]
        
        print(f"\nDataset loaded:")
        print(f"Training: {len(self.train_waldo)} Waldo, {len(self.train_negative)} negative")
        print(f"Validation: {len(self.val_waldo)} Waldo, {len(self.val_negative)} negative")
    
    def _load_and_preprocess(self, image_path: Path) -> Optional[np.ndarray]:
        """Load and preprocess a single image."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Failed to load {image_path}")
                return None
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.image_size)
            
            # Apply augmentations if enabled
            if self.augment:
                # Random horizontal flip
                if np.random.random() > 0.5:
                    image = np.fliplr(image)
                
                # Random rotation ±10 degrees
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-10, 10)
                    h, w = image.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT)
                
                # Color jittering
                if np.random.random() > 0.5:
                    # Brightness
                    alpha = np.random.uniform(0.6, 1.4)
                    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
                
                if np.random.random() > 0.5:
                    # Contrast
                    alpha = np.random.uniform(0.6, 1.4)
                    mean = np.mean(image)
                    image = cv2.convertScaleAbs(image, alpha=alpha, 
                                              beta=(1-alpha)*mean)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _prepare_batch(self, 
                      waldo_paths: List[Path],
                      negative_paths: List[Path],
                      batch_size: int) -> Dict[str, np.ndarray]:
        """Prepare a balanced batch of Waldo and negative examples."""
        # Determine number of each class
        n_waldo = batch_size // 2
        n_negative = batch_size - n_waldo
        
        # Randomly sample paths
        batch_waldo = np.random.choice(waldo_paths, n_waldo)
        batch_negative = np.random.choice(negative_paths, n_negative)
        
        # Load and preprocess images
        images = []
        labels = []
        
        for path in batch_waldo:
            image = self._load_and_preprocess(path)
            if image is not None:
                images.append(image)
                labels.append(1)  # Waldo class
                
        for path in batch_negative:
            image = self._load_and_preprocess(path)
            if image is not None:
                images.append(image)
                labels.append(0)  # Negative class
        
        if not images:
            raise ValueError("No valid images in batch")
            
        # Stack into arrays
        images = np.stack(images)
        labels = np.array(labels, dtype=np.float32)
        
        return {
            'image': images,
            'label': labels[:, None]  # Add channel dimension
        }
    
    def train_loader(self) -> Dict[str, np.ndarray]:
        """Create training data loader."""
        while True:
            try:
                batch = self._prepare_batch(
                    self.train_waldo,
                    self.train_negative,
                    self.batch_size
                )
                yield batch
            except Exception as e:
                print(f"Error creating training batch: {str(e)}")
                continue
    
    def val_loader(self) -> Dict[str, np.ndarray]:
        """Create validation data loader."""
        while True:
            try:
                batch = self._prepare_batch(
                    self.val_waldo,
                    self.val_negative,
                    self.batch_size
                )
                yield batch
            except Exception as e:
                print(f"Error creating validation batch: {str(e)}")
                continue

class BinaryClassifier(train_state.TrainState):
    """Training state with dropout RNG."""
    dropout_rng: Any

def create_binary_classifier(rng: jnp.ndarray,
                           learning_rate: float,
                           model_kwargs: Dict,
                           num_train_steps: Optional[int] = None,
                           warmup_epochs: Optional[int] = None,
                           steps_per_epoch: Optional[int] = None) -> BinaryClassifier:
    """Creates binary classifier for Waldo detection."""
    # Create learning rate schedule
    if num_train_steps and warmup_epochs and steps_per_epoch:
        warmup_steps = warmup_epochs * steps_per_epoch
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps
        )
        cosine_steps = max(num_train_steps - warmup_steps, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=cosine_steps
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps]
        )
    else:
        schedule_fn = learning_rate

    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    )
    
    # Initialize model
    model = hydra.utils.instantiate(model_kwargs)
    variables = model.init(
        rng,
        jnp.ones((1, *model_kwargs.image_size, 3)),
        training=False
    )
    
    return BinaryClassifier.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        dropout_rng=rng
    )

def compute_loss_and_metrics(logits: jnp.ndarray,
                           labels: jnp.ndarray,
                           class_weight: float = 10.0) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute loss and metrics for binary classification."""
    # Get predictions
    probs = jax.nn.sigmoid(logits)
    preds = (probs > 0.5).astype(jnp.float32)
    
    # Compute weighted binary cross entropy
    bce = optax.sigmoid_binary_cross_entropy(logits, labels)
    weights = jnp.where(labels == 1, class_weight, 1.0)
    loss = jnp.mean(bce * weights)
    
    # Compute metrics
    accuracy = jnp.mean((preds == labels).astype(jnp.float32))
    
    # Compute precision/recall for Waldo class
    tp = jnp.sum((preds == 1) & (labels == 1))
    fp = jnp.sum((preds == 1) & (labels == 0))
    fn = jnp.sum((preds == 0) & (labels == 1))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return loss, metrics

@jax.jit
def train_step(state: BinaryClassifier,
               batch: Dict[str, jnp.ndarray],
               class_weight: float) -> Tuple[BinaryClassifier, Dict[str, jnp.ndarray]]:
    """Single training step."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['image'],
            training=True,
            rngs={'dropout': state.dropout_rng}
        )
        loss, metrics = compute_loss_and_metrics(logits, batch['label'], class_weight)
        return loss, metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

@jax.jit
def eval_step(state: BinaryClassifier,
              batch: Dict[str, jnp.ndarray],
              class_weight: float) -> Dict[str, jnp.ndarray]:
    """Evaluation step."""
    logits = state.apply_fn(
        {'params': state.params},
        batch['image'],
        training=False,
        rngs={'dropout': state.dropout_rng}
    )
    _, metrics = compute_loss_and_metrics(logits, batch['label'], class_weight)
    return metrics

@hydra.main(config_path="../../config", config_name="stage1_train", version_base=None)
def train(cfg: DictConfig) -> None:
    """Train Waldo appearance classifier."""
    # Initialize wandb if available
    use_wandb = False
    try:
        if os.environ.get('WANDB_MODE') != 'disabled':
            wandb.init(
                project="waldo-finder",
                name=f"{cfg.experiment_name}-stage1",
                config=dict(cfg),
                tags=["Stage1", "Appearance-Learning"],
            )
            use_wandb = True
    except:
        print("Wandb not available, continuing without logging")
    
    # Set up data loaders
    dataset = Stage1Dataset(
        data_dir=cfg.data.train_dir,
        image_size=tuple(cfg.data.image_size),
        batch_size=cfg.training.batch_size,
        augment=cfg.data.augment,
        val_split=cfg.data.val_split
    )
    
    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    
    # Initialize model and training state
    print("\nInitializing model...")
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, init_rng = jax.random.split(rng)
    
    # Calculate training parameters
    steps_per_epoch = (
        len(dataset.train_waldo) + len(dataset.train_negative)
    ) // cfg.training.batch_size
    
    num_train_steps = steps_per_epoch * cfg.training.num_epochs
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    
    state = create_binary_classifier(
        init_rng,
        learning_rate=cfg.training.learning_rate,
        model_kwargs=cfg.model,
        num_train_steps=num_train_steps,
        warmup_epochs=cfg.training.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    # Training loop
    best_val_f1 = 0.0
    early_stopping_counter = 0
    
    print("\nStarting training...")
    for epoch in range(cfg.training.num_epochs):
        # Training
        state = state.replace(dropout_rng=jax.random.fold_in(state.dropout_rng, epoch))
        train_metrics = []
        
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}") as pbar:
            for step in pbar:
                try:
                    # Get next batch
                    batch = next(train_loader)
                    
                    # Training step
                    state, metrics = train_step(
                        state, batch,
                        cfg.training.classification.positive_class_weight
                    )
                    train_metrics.append(metrics)
                    
                    # Update progress bar
                    if len(train_metrics) >= 10:
                        avg_loss = float(np.mean([m['loss'] for m in train_metrics[-10:]]))
                        avg_f1 = float(np.mean([m['f1'] for m in train_metrics[-10:]]))
                        pbar.set_postfix({
                            'loss': f"{avg_loss:.3f}",
                            'f1': f"{avg_f1:.3f}"
                        })
                except Exception as e:
                    print(f"\nError during training step: {str(e)}")
                    continue
        
        # Calculate epoch metrics
        train_epoch_metrics = {
            k: float(np.mean([m[k] for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        
        # Validation
        val_metrics = []
        n_val_batches = max(
            (len(dataset.val_waldo) + len(dataset.val_negative)) 
            // cfg.training.batch_size,
            1
        )
        
        for _ in range(n_val_batches):
            try:
                batch = next(val_loader)
                metrics = eval_step(
                    state, batch,
                    cfg.training.classification.positive_class_weight
                )
                val_metrics.append(metrics)
            except Exception as e:
                print(f"\nError during validation: {str(e)}")
                continue
        
        val_epoch_metrics = {
            k: float(np.mean([m[k] for m in val_metrics]))
            for k in val_metrics[0].keys()
        }
        
        # Log metrics
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                **{f"train_{k}": v for k, v in train_epoch_metrics.items()},
                **{f"val_{k}": v for k, v in val_epoch_metrics.items()}
            })
        
        # Print epoch summary
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}")
        print(f"{'='*40}")
        print(f"Training:")
        print(f"- Loss: {train_epoch_metrics['loss']:.4f}")
        print(f"- Accuracy: {train_epoch_metrics['accuracy']:.4f}")
        print(f"- F1: {train_epoch_metrics['f1']:.4f}")
        print(f"\nValidation:")
        print(f"- Loss: {val_epoch_metrics['loss']:.4f}")
        print(f"- Accuracy: {val_epoch_metrics['accuracy']:.4f}")
        print(f"- F1: {val_epoch_metrics['f1']:.4f}")
        
        # Save best model
        if val_epoch_metrics['f1'] > best_val_f1:
            best_val_f1 = val_epoch_metrics['f1']
            early_stopping_counter = 0
            
            # Save model with absolute path
            model_dir = Path(hydra.utils.get_original_cwd()) / cfg.training.model_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as pickle file
            checkpoint_path = model_dir / 'best_model.pkl'
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'batch_stats': {},  # Empty dict since we don't use batch norm
                }, f)
            
            print(f"\nSaved new best model (F1: {best_val_f1:.4f})")
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= cfg.training.early_stopping.patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    train()
