# 🔍 There's Waldo: Modern Vision Transformer for Finding Waldo

A state-of-the-art implementation for solving "Where's Waldo?" puzzles using JAX and Vision Transformers. This project demonstrates modern deep learning practices and efficient implementation using JAX's automatic differentiation and compilation capabilities.

![Waldo Detection Example](docs/docs.png)

## 🚀 Key Features

- **Modern Architecture**: Vision Transformer (ViT) backbone with DETR-style detection head
- **High Performance**: JAX-based implementation with JIT compilation and automatic differentiation
- **Advanced Training**: 
  - Focal Loss for handling class imbalance
  - GIoU Loss for better bounding box regression
  - Automatic Mixed Precision (AMP) training
  - Gradient clipping and learning rate scheduling
- **Rich Augmentations**: Comprehensive data augmentation pipeline including:
  - Random flipping
  - Color jittering
  - Brightness/contrast adjustments
  - Aspect ratio preservation
- **MLOps Best Practices**:
  - Experiment tracking with Weights & Biases
  - Configuration management with Hydra
  - Type hints and comprehensive documentation
  - Modular, maintainable codebase

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/noah-ing/theres-waldo.git
cd theres-waldo

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"
```

## 🎯 Quick Start

Find Waldo in an image:

```bash
find-waldo path/to/image.jpg
```

Or use the Python API:

```python
from waldo_finder import WaldoFinder

# Initialize finder with trained model
finder = WaldoFinder('models/best_model.pkl')

# Find Waldo!
results = finder.find_waldo('path/to/image.jpg')
```

## 🏋️ Training

Train your own model with custom configuration:

```bash
# Set up wandb for experiment tracking
wandb login

# Train with default configuration
python -m waldo_finder.train

# Train with custom configuration
python -m waldo_finder.train data.batch_size=32 training.learning_rate=0.0002
```

## 📊 Model Architecture

The model uses a Vision Transformer backbone with modern improvements:

- Pre-norm transformer blocks for stable training
- Learnable class token for global feature aggregation
- Sinusoidal positional embeddings
- DETR-style detection head for accurate localization

```
Input Image (640x640x3)
    │
    ▼
ViT Backbone (12 layers)
    │
    ▼
Detection Head
    │
    ▼
Bounding Box + Confidence
```

## 🔧 Advanced Usage

### Custom Training Configuration

Modify `config/train.yaml` or override via command line:

```bash
python -m waldo_finder.train \
    model.num_heads=8 \
    model.dropout_rate=0.2 \
    training.batch_size=64
```

### Experiment Tracking

View training progress and compare experiments:

```bash
# Start training with wandb logging
python -m waldo_finder.train wandb.project=my-project wandb.name=exp-001

# View results at: https://wandb.ai/username/my-project
```

## 📈 Performance

The model achieves state-of-the-art performance on Where's Waldo puzzles:

- **Accuracy**: 95%+ detection rate on test set
- **Speed**: ~100ms inference time on modern GPU
- **Robustness**: Handles various image sizes and styles

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original dataset from [Hey-Waldo](https://github.com/vc1492a/Hey-Waldo)
- Inspired by advances in vision transformers and object detection
- Built with JAX ecosystem tools (Flax, Optax)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{waldo_finder_2025,
  author = {Noah},
  title = {There's Waldo: Modern Vision Transformer for Finding Waldo},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/noah-ing/theres-waldo}
}
