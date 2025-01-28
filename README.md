# 🔍 There's Waldo: Modern Vision Transformer for Finding Waldo

A state-of-the-art implementation for solving "Where's Waldo?" puzzles using JAX and Vision Transformers. This project demonstrates modern deep learning practices and efficient implementation using JAX's automatic differentiation and compilation capabilities.

![Waldo Detection Example](docs/docs.png)

## 🚀 Project Status (January 2025)

### Latest Memory & Training Optimizations (January 2025)
- Memory-Efficient Architecture:
  - Optimized 12-layer Vision Transformer (down from 24)
  - Efficient 12-head attention mechanism
  - Balanced 768-dim hidden states (reduced from 1024)
  - Memory-efficient 3072-dim MLP layers
  - Enhanced gradient checkpointing
- Advanced Training Pipeline:
  - Smart gradient accumulation (16 steps)
  - Optimized batch size (8) with effective batch size of 128
  - Multiple dataset passes per epoch for better utilization
  - Improved data augmentation with shape consistency
  - Stable mixed precision training
- Sophisticated Optimization:
  - Lion optimizer with tuned learning rate (0.0002)
  - Efficient warmup and cosine decay scheduling
  - Enhanced dropout and regularization
  - Exponential Moving Average (EMA) of weights
- Robust Training Features:
  - Early stopping with configurable patience
  - Comprehensive metric tracking (loss, GIoU, scores)
  - Advanced data augmentation with shape preservation
  - Real-time progress monitoring

### Advanced Architecture Features
- Memory-Efficient Vision Transformer:
  - 12 transformer layers with 12 attention heads
  - 768-dimensional hidden states
  - 3072-dimensional MLP layers
  - Gradient checkpointing for memory efficiency
- Sophisticated training infrastructure:
  - Mixed precision training for efficiency
  - Dynamic gradient scaling
  - Advanced optimizer configuration (AdamW)
  - Modern learning rate scheduling
- Enhanced data processing:
  - Advanced augmentation pipeline
  - Efficient data loading
  - Real-time metric tracking

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

## 📊 Advanced Model Architecture

The model implements state-of-the-art Vision Transformer architecture with numerous enhancements:
- Enhanced transformer blocks:
  - Pre-norm architecture with epsilon=1e-6
  - SwiGLU activation function
  - Stochastic Depth for regularization
  - Advanced dropout strategies
- Sophisticated attention mechanisms:
  - 12-head attention with relative position bias
  - Memory-efficient attention patterns
  - Optimized attention regularization
- Advanced detection heads:
  - Deep prediction networks
  - Multi-layer feature aggregation
  - Improved localization accuracy

## 🏃‍♂️ Running the Model

### Current TensorFlow Model (Legacy)
The original TensorFlow implementation is temporarily unavailable due to compatibility issues with TensorFlow 2.x.

### Advanced JAX Implementation
Training pipeline features cutting-edge techniques:
- Mixed precision training with dynamic scaling
- Gradient accumulation for stability
- EMA parameter averaging
- Advanced loss functions:
  - GIoU loss for accurate box regression
  - Focal loss with dynamic weighting
- Sophisticated checkpointing:
  - Best model tracking
  - Training state preservation
  - Comprehensive metadata logging

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
