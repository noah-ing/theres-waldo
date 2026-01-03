# Where's Waldo Detector 

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art computer vision system that uses advanced deep learning to find Waldo in complex scenes. Built with a hierarchical Vision Transformer architecture and context-aware attention mechanisms, this system achieves precise localization through scale-aware detection and scene-level understanding.

## Technical Highlights

### Hierarchical Architecture
- **Vision Transformer Backbone (41.6M params)**
 * Multi-scale processing with 32x32 patches
 * Progressive feature pooling
 * 12 transformer layers
 * Scale-aware embeddings
 * Mixed precision training

- **Context-Aware Attention (1.1M params)**
 * Global scene attention
 * Local region focus
 * Cross-scale interactions
 * Spatial relationships
 * Window attention

- **Detection System (1.8M params)**
 * Location regression (weight: 2.0)
 * Scale prediction (weight: 1.0)
 * Context scoring (weight: 1.5)
 * Confidence estimation (weight: 1.0)
 * NMS post-processing

### Training Pipeline
- **Multi-Stage Training** (️ Currently Debugging)
 * Pre-training phase (30 epochs, planned)
 * Contrastive learning (40 epochs, planned)
 * Detection fine-tuning (early stopping at 3/60 epochs)
 * Curriculum progression (not yet validated)

- **Known Issues**
 * Training speed: 0.18-0.21 it/s
 * Batch count: 3 batches/epoch
 * Loss plateauing at 0.000001
 * Potential gradient vanishing
 * Data loading bottlenecks

- **Curriculum Learning** (Planned)
 * Easy: 10 distractors, 5% min size
 * Medium: 20 distractors, 3% min size
 * Hard: 30 distractors, 2% min size
 * Progression threshold: 0.25
 * Patience: 3 epochs

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/noah-ing/theres-waldo.git
cd theres-waldo

# Create virtual environment
python -m venv venv
source venv/bin/activate # Unix
venv\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt

# Development install
pip install -e .
```

### Basic Usage
```bash
# Find Waldo in an image
python -m waldo_finder.inference --image path/to/image.jpg --model checkpoints/latest.ckpt

# Train the model
python -m waldo_finder.training.train --config-name train
```

## Technical Architecture

### Core Components
- **Hierarchical Vision Transformer**
 * Multi-scale processing
 * Progressive feature pooling
 * Skip connections
 * Feature pyramids

- **Context-Aware Attention**
 * Global scene features
 * Local region features
 * Cross-scale features
 * Spatial relations

- **Detection Head**
 * Box coordinates
 * Scale factors
 * Context scores
 * Confidence values

### Training Strategy
- **Three-Phase Pipeline** (️ Under Development)
 * Pattern learning (not yet started)
 * Discrimination (not yet started)
 * Localization (stopped at 3/60 epochs)
 * Curriculum advancement (pending)

- **Current Focus**
 * Debugging data pipeline issues
 * Investigating loss function behavior
 * Optimizing batch processing
 * Monitoring gradient flow

### Performance Optimization
- **Hardware Utilization**
 * GPU acceleration
 * Mixed precision (16-bit)
 * Gradient accumulation (x2)
 * Memory monitoring
 * Cache size: 100

## Configuration

### Model Architecture
```yaml
model:
 img_size: 384
 patch_size: 32
 num_layers: 12
 num_heads: 8
 hidden_dim: 512
 mlp_dim: 2048
 dropout: 0.1
```

### Training Settings
```yaml
training:
 pretrain:
 epochs: 30
 batch_size: 8
 learning_rate: 2e-4

 contrastive:
 epochs: 40
 batch_size: 8
 learning_rate: 1e-4
 mining:
 k: 3000

 detection:
 epochs: 60
 batch_size: 4
 learning_rate: 5e-5
```

## System Requirements

### Hardware
- CUDA-capable GPU recommended
- 16GB+ RAM
- SSD recommended
- Multi-core processor

### Software
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.x (for GPU)
- cuDNN 8.x

## ️ Project Structure
```
theres-waldo/
├── src/waldo_finder/
│ ├── models/
│ │ ├── hierarchical_vit.py
│ │ ├── attention.py
│ │ └── detection_head.py
│ ├── training/
│ │ ├── train.py
│ │ └── trainer.py
│ ├── data/
│ │ ├── scene_dataset.py
│ │ └── prepare_data.py
│ └── inference/
│ └── predict.py
├── config/
│ ├── scene_model.yaml
│ └── model/
│ └── vit_base.yaml
├── data/
│ └── scenes/
└── outputs/
 ├── checkpoints/
 └── logs/
```

## License
MIT License - see [LICENSE](LICENSE)

## Acknowledgments
- Martin Handford for the "Where's Waldo?" series
- PyTorch team for excellent ML tools
- Vision Transformer authors for inspiration

## Contact
For questions and feedback:
- Open an issue
