# 🔍 There's Waldo: Modern Vision Transformer for Finding Waldo

A state-of-the-art implementation for solving "Where's Waldo?" puzzles using JAX and Vision Transformers. This project demonstrates modern deep learning practices and efficient implementation using JAX's automatic differentiation and compilation capabilities.

![Waldo Detection Example](docs/docs.png)

## 🚀 Project Status (January 2025)

### Completed Upgrades
- Modernized codebase with JAX/Flax implementation
- Added Vision Transformer (ViT) backbone with DETR-style detection
- Implemented modern training practices (Focal Loss, GIoU Loss)
- Added configuration management with Hydra
- Set up experiment tracking with Weights & Biases
- Added comprehensive data augmentation pipeline

### Current Setup
- Virtual environment created and activated
- Core dependencies installed:
  - JAX/Flax ecosystem
  - TensorFlow for data handling
  - OpenCV and Pillow for image processing
  - Wandb for experiment tracking
  - Hydra for configuration
  - Pandas for data management

### Next Steps Needed
1. Fix Hydra configuration path issue
2. Complete model training with the new architecture
3. Convert existing TensorFlow model weights (optional)
4. Test and validate the new implementation

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/noah-ing/theres-waldo.git
cd theres-waldo

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -e .
```

## 📊 Model Architecture

The model uses a Vision Transformer backbone with modern improvements:
- Pre-norm transformer blocks for stable training
- Learnable class token for global feature aggregation
- Sinusoidal positional embeddings
- DETR-style detection head for accurate localization

## 🏃‍♂️ Running the Model

### Current TensorFlow Model (Legacy)
The original TensorFlow implementation is temporarily unavailable due to compatibility issues with TensorFlow 2.x.

### New JAX Implementation (In Progress)
Training pipeline is set up but requires configuration fixes. Stay tuned for updates.

## 📈 Dataset

The project uses a curated dataset of Where's Waldo images with:
- 30+ annotated images
- Precise bounding box coordinates
- Various scene complexities
- Multiple image resolutions

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
