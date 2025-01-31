# Where's Waldo Detector 🔍

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art computer vision system that uses advanced deep learning to find Waldo in complex scenes. Built with a hierarchical Vision Transformer architecture and context-aware attention mechanisms, this system achieves precise localization through scale-aware detection and scene-level understanding.

![Where's Waldo Detection](docs/docs.png)

## 🌟 Technical Highlights

### Hierarchical Architecture
- **Vision Transformer Backbone (41.6M params)**
  * Multi-scale processing with 32x32 patches
  * Progressive feature pooling
  * 12 transformer layers
  * Scale-aware embeddings
  * Mixed precision training

- **Context-Aware Attention (3.9M params)**
  * Global scene attention
  * Local region focus
  * Cross-scale interactions
  * Spatial relationships
  * Window attention

- **Detection System (331K params)**
  * Location regression
  * Scale prediction
  * Context scoring
  * Confidence estimation
  * NMS post-processing

### Training Pipeline
- **Multi-Stage Training**
  * Pre-training phase (10 epochs)
  * Contrastive learning (10 epochs)
  * Detection fine-tuning (20 epochs)
  * Curriculum progression

- **Data Organization**
  * 29 total scenes
  * 23 training scenes
  * 6 validation scenes
  * Curriculum split:
    - Easy: 2 scenes (8.7%)
    - Medium: 14 scenes (60.9%)
    - Hard: 7 scenes (30.4%)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/noah-ing/theres-waldo.git
cd theres-waldo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Development install
pip install -e .
```

### Basic Usage
```bash
# Find Waldo in an image
python -m waldo_finder.inference --image images/1.jpg --model checkpoints/latest.ckpt

# Train the model
python -m waldo_finder.training.train --config-name scene_model
```

## 📖 Technical Architecture

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
- **Multi-Stage Pipeline**
  * Self-supervised pre-training
  * Contrastive learning
  * Detection fine-tuning
  * Curriculum progression

### Performance Optimization
- **Hardware Utilization**
  * GPU acceleration
  * Mixed precision (16-bit)
  * Persistent workers
  * Memory efficiency

## 🔧 Configuration

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
    epochs: 10
    batch_size: 32
    learning_rate: 0.0001
  
  contrastive:
    epochs: 10
    batch_size: 16
    learning_rate: 0.00005
  
  detection:
    epochs: 20
    batch_size: 4
    learning_rate: 0.00003
```

## 📊 System Requirements

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

## 🛠️ Project Structure
```
theres-waldo/
├── src/waldo_finder/
│   ├── models/
│   │   ├── hierarchical_vit.py
│   │   ├── attention.py
│   │   └── detection_head.py
│   ├── training/
│   │   ├── train.py
│   │   └── trainer.py
│   ├── data/
│   │   ├── scene_dataset.py
│   │   └── prepare_data.py
│   └── inference/
│       └── predict.py
├── config/
│   ├── scene_model.yaml
│   └── model/
│       └── vit_base.yaml
├── data/
│   └── scenes/
└── outputs/
    ├── checkpoints/
    └── logs/
```

## 📝 License
MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments
- Martin Handford for the "Where's Waldo?" series
- PyTorch team for excellent ML tools
- Vision Transformer authors for inspiration

## 📧 Contact
For questions and feedback:
- Open an issue
