# There's Waldo: Advanced Object Detection

A state-of-the-art solution for finding Waldo in "Where's Waldo?" puzzles using Vision Transformers implemented in JAX.

## Features

- CPU-optimized Vision Transformer implementation
- Advanced Detection System:
  - Multi-box detection support
  - Center-size prediction format
  - Box size constraints and penalties
  - Enhanced confidence scoring
- Advanced Regularization:
  - GridMask augmentation for structured dropout
  - Consistency regularization between views
  - Enhanced attention with relative position encoding
  - Stochastic depth with linear scaling
- Sophisticated Training:
  - Robust shape handling and broadcasting
  - Multi-box batch processing
  - Advanced gradient accumulation
  - Comprehensive metric tracking
- Memory-efficient Design:
  - Optimized tensor operations
  - Smart shape management
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
- Multi-box detection support
- Advanced shape handling
- Box size constraints
- Improved regularization
- Enhanced monitoring

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
- Multi-box detection heads
- Center-size prediction
- Box size constraints:
  - Width/height in [0.1, 0.4]
  - Target size of 0.15
  - Size-aware penalties

## Training Pipeline

The enhanced training pipeline features:
- Multi-box batch processing
- Robust shape handling
- Gradient accumulation (8 steps)
- Advanced early stopping
- EMA parameter averaging
- Loss Components:
  - Center point loss (1.5x weight)
  - Size control loss (1.0x weight)
  - IoU quality loss (2.0x weight)
  - Size penalty loss (0.5x weight)
  - Confidence loss (0.5x weight)

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
