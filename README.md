# There's Waldo: Advanced Object Detection

A state-of-the-art solution for finding Waldo in "Where's Waldo?" puzzles using Vision Transformers implemented in JAX.

## Features

- CPU-optimized Vision Transformer implementation
- Advanced regularization techniques:
  - GridMask augmentation for structured dropout
  - Consistency regularization between views
  - Enhanced attention with relative position encoding
  - Stochastic depth with linear scaling
- Sophisticated training pipeline:
  - Cyclic learning rate with warmup and restarts
  - Enhanced EMA parameter averaging
  - Advanced gradient accumulation
  - Comprehensive metric tracking
- Memory-efficient design:
  - Optimized batch processing
  - Smart gradient handling
  - Enhanced parameter updates

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Unix/MacOS
venv\Scripts\activate     # Windows

# Install package
pip install -e .
```

## Training

Two training options are available:

### Standard Training
```bash
python -m waldo_finder.train
```

### Enhanced Training (Recommended)
```bash
python -m waldo_finder.train_optimized
```

The enhanced training includes:
- Advanced regularization techniques
- Improved optimization strategies
- Better monitoring and checkpointing

## Configuration

Configuration files are located in the `config/` directory:

- `train.yaml`: Basic training configuration
- `train_optimized.yaml`: Enhanced training settings
- `model/vit_base.yaml`: Base model architecture
- `model/vit_base_optimized.yaml`: Enhanced model architecture

## Project Structure

```
├── config/                 # Configuration files
├── src/waldo_finder/      # Core implementation
│   ├── model.py           # Base ViT implementation
│   ├── model_optimized.py # Enhanced ViT with regularization
│   ├── train.py           # Basic training pipeline
│   ├── train_optimized.py # Advanced training pipeline
│   ├── data.py           # Data loading and preprocessing
│   └── inference.py      # Inference and visualization
├── images/                # Training data
└── annotations/          # Bounding box annotations
```

## Model Architecture

The Vision Transformer (ViT) architecture includes:
- 8 transformer layers
- 8 attention heads
- 512-dim hidden states
- 2048-dim MLP layers
- Advanced regularization suite
- Enhanced coordinate prediction

## Training Pipeline

The enhanced training pipeline features:
- Gradient accumulation (8 steps)
- Cyclic learning rate scheduling
- Advanced early stopping
- EMA parameter averaging
- Comprehensive metrics
- Enhanced checkpointing

## Requirements

- Python 3.x
- JAX (CPU version)
- Flax
- Hydra
- Additional dependencies in requirements.txt

## License

MIT License - see LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.
