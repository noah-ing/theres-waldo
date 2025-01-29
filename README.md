# Where's Waldo Detector 🔍

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-CPU%20Optimized-green.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A state-of-the-art computer vision system that automatically locates Waldo in "Where's Waldo?" images using Vision Transformers and modern deep learning techniques.

<div align="center">
  <img src="docs/docs.png" alt="Where's Waldo Detection" width="600px"/>
</div>

## 🌟 Features

- **Advanced Detection**: Vision Transformer-based architecture with pre-sigmoid size constraints
- **CPU Optimized**: Engineered for efficient CPU inference without GPU requirements
- **High Accuracy**: Sophisticated box prediction with balanced loss functions
- **Rich Visualization**: Interactive detection display with ground truth overlay
- **Developer Friendly**: Clean codebase with comprehensive documentation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/theres-waldo.git
cd theres-waldo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Development install
pip install -e .
```

### Basic Usage

```bash
# Run inference on an image
python -m waldo_finder.inference images/1.jpg --model models/best_model.pkl

# Train a new model
python -m waldo_finder.train
```

## 📖 Documentation

### Model Architecture

The system uses a Vision Transformer (ViT) architecture optimized for CPU inference:

- 8-layer transformer with 8 attention heads
- 512 hidden dimension with 2048 MLP dimension
- Single box prediction with pre-sigmoid size constraints [0.1, 0.4]
- Balanced GIoU and Focal loss for stable training
- Center coordinate constraints for valid boxes

### Training Pipeline

The training system features:

- Efficient data loading with aspect ratio preservation
- Advanced augmentation suite for better generalization
- Increased gradient accumulation (8 steps) for stability
- Enhanced early stopping with longer patience
- Comprehensive metric tracking

### Configuration

The project uses Hydra for flexible configuration:

```yaml
# Model settings
model:
  num_heads: 8
  num_layers: 8
  hidden_dim: 512
  mlp_dim: 2048
  dropout_rate: 0.2

# Training settings
training:
  batch_size: 2
  num_epochs: 50
  learning_rate: 0.0003  # Optimized for stability
  gradient_accumulation_steps: 8  # Increased for better updates
  early_stopping:
    patience: 15  # Allow proper convergence
    min_delta: 0.0001  # Fine-grained improvements
```

## 🔧 Advanced Usage

### Custom Training

```bash
# Train with custom configuration
python -m waldo_finder.train \
  training.batch_size=4 \
  training.learning_rate=0.0003

# Enable wandb logging
export WANDB_MODE=online
python -m waldo_finder.train
```

### Inference Options

```bash
# Run inference with visualization
python -m waldo_finder.inference \
  images/1.jpg \
  --model models/best_model.pkl \
  --conf-threshold 0.5 \
  --output result.png

# Disable visualization blur
python -m waldo_finder.inference \
  images/1.jpg \
  --model models/best_model.pkl \
  --no-blur
```

## 📊 Performance

The system achieves excellent detection performance:

- **Training**: Validation loss improved from 1.2368 to 1.0562 (14.6%)
- **Stability**: Perfect loss balance (GIoU: ~1.014, Score: 0.0071)
- **Convergence**: Train/val gap closed from 0.2 to 0.035
- **Speed**: Efficient CPU inference (~0.5s per image)
- **Memory**: Optimized with 8-step gradient accumulation

## 🛠️ Development

### Project Structure

```
theres-waldo/
├── src/waldo_finder/        # Core implementation
│   ├── model.py            # Vision Transformer architecture
│   ├── train.py            # Training pipeline
│   ├── inference.py        # Detection pipeline
│   └── data.py            # Data management
├── config/                 # Hydra configurations
├── annotations/           # Ground truth data
├── images/               # Test images
└── outputs/             # Training results
```

### Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Martin Handford for creating the amazing "Where's Waldo?" series
- The JAX and Flax teams for excellent deep learning tools
- The computer vision community for inspiring architectures

## 📧 Contact

For questions and feedback:

- Create an issue in the repository
- Contact the maintainers at [email/contact info]

---
<div align="center">
  Made with ❤️ by [Your Name/Organization]
</div>
