import pytest
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from src.waldo_finder.data.transforms import WaldoTransforms
from src.waldo_finder.data.scene_dataset import SceneDataset

def test_rgba_to_rgb_conversion():
    """Test RGBA to RGB conversion in transform pipeline"""
    # Create a sample RGBA image
    img_size = 100
    rgba_data = np.random.randint(0, 255, (img_size, img_size, 4), dtype=np.uint8)
    rgba_image = Image.fromarray(rgba_data, 'RGBA')
    
    # Basic config for transforms
    config = {
        'augmentation': {
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'random_affine': {
                'degrees': 10,
                'translate': (0.1, 0.1),
                'scale': (0.9, 1.1)
            },
            'random_horizontal_flip': 0.5,
            'random_vertical_flip': 0.5
        }
    }
    
    # Initialize transforms
    transforms = WaldoTransforms(config, is_train=True, img_size=384)
    
    # Apply transforms
    transformed = transforms(rgba_image)
    
    # Verify output
    assert isinstance(transformed, torch.Tensor), "Output should be a tensor"
    assert transformed.shape[0] == 3, f"Expected 3 channels (RGB), got {transformed.shape[0]}"
    assert torch.is_floating_point(transformed), "Output should be float tensor"
    assert transformed.min() >= -1 and transformed.max() <= 1, "Values should be normalized"

def test_dataset_image_loading():
    """Test dataset image loading and transformation"""
    # Sample config
    config = {
        'model': {'img_size': 384},
        'augmentation': {
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            },
            'random_affine': {
                'degrees': 10,
                'translate': (0.1, 0.1),
                'scale': (0.9, 1.1)
            },
            'random_horizontal_flip': 0.5,
            'random_vertical_flip': 0.5
        }
    }
    
    # Initialize dataset
    data_dir = Path('data/scenes')
    if not data_dir.exists():
        pytest.skip("Test data directory not found")
        
    dataset = SceneDataset(
        data_dir=data_dir,
        config=config,
        split='train'
    )
    
    # Test first item
    try:
        item = dataset[0]
        assert isinstance(item['image'], torch.Tensor), "Image should be a tensor"
        assert item['image'].shape[0] == 3, f"Expected 3 channels (RGB), got {item['image'].shape[0]}"
        assert torch.is_floating_point(item['image']), "Image should be float tensor"
        assert item['image'].min() >= -1 and item['image'].max() <= 1, "Values should be normalized"
    except Exception as e:
        pytest.fail(f"Dataset loading failed: {str(e)}")

if __name__ == '__main__':
    # Run tests
    print("Testing RGBA to RGB conversion...")
    test_rgba_to_rgb_conversion()
    print("RGBA to RGB conversion test passed!")
    
    print("\nTesting dataset image loading...")
    test_dataset_image_loading()
    print("Dataset image loading test passed!")
