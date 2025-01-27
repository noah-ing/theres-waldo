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

### Current Progress
1. ✅ Fixed Hydra configuration issues
2. ✅ Implemented smart data loading with validation
3. ✅ Set up model serialization with msgpack
4. 🔄 Currently training the model (check wandb dashboard for progress)

### Next Steps
1. Model Evaluation & Refinement
   - Complete initial training run
   - Analyze training metrics from wandb
   - Fine-tune hyperparameters if needed
   - Validate model performance on test images

2. Inference Pipeline
   - Create user-friendly inference script
   - Add visualization tools for predictions
   - Implement confidence thresholding
   - Add support for different image sizes

3. Documentation & Demo
   - Create example notebook
   - Add model performance metrics
   - Include sample predictions
   - Document training process

4. Deployment
   - Create Docker container
   - Add CI/CD pipeline
   - Set up model versioning
   - Create web demo (optional)

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

### New JAX Implementation (Active Development)
Training pipeline is operational with:
- Vision Transformer backbone
- Modern training practices (GIoU + Focal Loss)
- Real-time metric tracking via wandb
- Efficient data loading and augmentation
- Automatic model checkpointing

Check the wandb dashboard for live training progress and metrics.

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
