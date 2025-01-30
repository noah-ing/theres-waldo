# Where's Waldo Detector 🔍

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-CPU%20Optimized-green.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A specialized computer vision system with one mission: FIND WALDO. Using advanced face detection and deep learning, this system is designed specifically to locate Waldo in complex "Where's Waldo?" scenes.

<div align="center">
  <img src="docs/docs.png" alt="Where's Waldo Detection" width="600px"/>
</div>

## 🌟 Features

- **Scale-Aware Detection**: Vision Transformer with scale-aware attention for precise localization
- **Enhanced Augmentations**: Advanced geometric and color transformations with box preservation
- **High Precision**: Strict IoU threshold (0.7) and high confidence requirement (0.9)
- **Robust Pipeline**: Comprehensive error handling and validation stability
- **CPU Optimized**: Efficient detection without GPU requirements

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
# Find Waldo in an image
python -m waldo_finder.inference images/1.jpg --model models/best_model.pkl

# Train the Waldo detector
python -m waldo_finder.train
```

## 📖 How It Works

### Scale-Aware Detection

The system uses a specialized Vision Transformer architecture:

- **Scale-Aware Network**:
  * Scale-based feature extraction
  * Layer normalization for stability
  * Enhanced augmentations
  * Box size constraints

- **Precise Detection**:
  1. Extract scale-aware features
  2. Enforce strict confidence threshold (0.9)
  3. Ensure precise localization (IoU > 0.7)

- **Box Handling**:
  * Tight size constraints [0.05-0.15]
  * Scale-based adaptation
  * Box-preserving processing
  * Coordinate transformations

### Training Strategy

The training process focuses on robust detection:

- Enhanced augmentations for better generalization:
  * Random rotations (±7°)
  * Scale/zoom variations (0.9-1.1)
  * Advanced color jittering
  * Box-preserving transformations

- Optimized loss components:
  * GIoU loss (8.0) for precise boxes
  * L1 loss (4.0) for coordinates
  * Confidence (0.1) for calibration
  * Size (2.0) for constraints

### Configuration

Key settings optimized for finding Waldo:

```yaml
# Model settings
model:
  num_heads: 8
  num_layers: 8
  hidden_dim: 512
  mlp_dim: 2048
  dropout_rate: 0.4

# Training configuration
training:
  batch_size: 4
  num_epochs: 150
  learning_rate: 0.00003  # Carefully tuned
  warmup_epochs: 12      # Extended warmup
  early_stopping:
    patience: 30        # Enhanced stability
    min_delta: 0.0001  # Precise improvements
```

## 🔧 Advanced Usage

### Training Options

```bash
# Train with custom settings
python -m waldo_finder.train \
  training.confidence_threshold=0.9 \
  training.iou_threshold=0.7

# Enable wandb logging
export WANDB_MODE=online
python -m waldo_finder.train
```

### Detection Options

```bash
# Find Waldo with visualization
python -m waldo_finder.inference \
  images/1.jpg \
  --model models/best_model.pkl \
  --conf-threshold 0.9 \
  --output found_waldo.png

# Show full scene
python -m waldo_finder.inference \
  images/1.jpg \
  --model models/best_model.pkl \
  --no-blur
```

## 📊 Success Metrics

The system prioritizes precise detection:

- **Scale Awareness**: Adaptive feature extraction for varying sizes
- **Detection Rate**: Optimized for reliable Waldo detection
- **Location Accuracy**: Strict IoU (0.7) for precise localization
- **Speed**: Efficient CPU-based detection

## 🛠️ Development

### Project Structure

```
theres-waldo/
├── src/waldo_finder/        # Core implementation
│   ├── model.py            # Scale-aware architecture
│   ├── train.py            # Enhanced training pipeline
│   ├── inference.py        # Detection system
│   └── data.py            # Augmentation & data handling
├── config/                 # Training configurations
├── annotations/           # Ground truth locations
├── images/               # Test scenes
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
  Made with ❤️ by Silo-22
</div>
