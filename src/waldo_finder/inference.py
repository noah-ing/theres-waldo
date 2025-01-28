"""Modern inference script for Waldo detection."""

import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Optional, Tuple, Union
from flax.serialization import msgpack_restore

from waldo_finder.model import WaldoDetector, create_train_state

class WaldoFinder:
    """Modern interface for Waldo detection."""
    
    def __init__(self, model_path: Union[str, Path]):
        """Initialize WaldoFinder.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.model_path = Path(model_path)
        
        # Initialize model with same config as training
        self.model = WaldoDetector(
            num_heads=12,
            num_layers=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout_rate=0.1
        )

        # Create initial state to get expected structure
        rng = jax.random.PRNGKey(0)
        state = create_train_state(
            rng,
            learning_rate=1e-4,  # Doesn't matter for inference
            model_kwargs={
                'num_heads': 12,
                'num_layers': 12,
                'hidden_dim': 768,
                'mlp_dim': 3072,
                'dropout_rate': 0.1,
            }
        )
        
        # Load and deserialize parameters
        with open(self.model_path, 'rb') as f:
            serialized_params = f.read()
            params = msgpack_restore(serialized_params)
            self.variables = {'params': params}
        
        # JIT compile inference function
        self.predict_fn = jax.jit(self._predict)
    
    def _preprocess_image(self, 
                         image: Union[str, Path, np.ndarray],
                         target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess image for inference.
        
        Args:
            image: Image path or numpy array
            target_size: Target image size
            
        Returns:
            Preprocessed image as numpy array
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original size for later
        self.original_size = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        height, width = image.shape[:2]
        scale = min(target_size[0]/width, target_size[1]/height)
        new_w, new_h = int(width * scale), int(height * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_w = (target_size[0] - new_w) // 2
        pad_h = (target_size[1] - new_h) // 2
        
        padded = np.pad(
            image,
            ((pad_h, target_size[1] - new_h - pad_h),
             (pad_w, target_size[0] - new_w - pad_w),
             (0, 0)),
            mode='constant'
        )
        
        # Store padding info for later
        self.pad_info = {
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h
        }
        
        # Normalize
        return padded.astype(np.float32) / 255.0
    
    def _predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run model prediction."""
        return self.model.apply(
            self.variables,
            image[None],  # Add batch dimension
            training=False,
            mutable=False,
            rngs={'dropout': jax.random.PRNGKey(0)}  # Fixed seed for inference
        )
    
    def _postprocess_boxes(self, 
                          boxes: np.ndarray,
                          scores: np.ndarray,
                          conf_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Convert normalized boxes back to original image coordinates."""
        # Filter by confidence
        mask = scores.squeeze() > conf_threshold
        if not mask.any():
            return np.array([]), np.array([])
        
        boxes = boxes[mask]
        scores = scores[mask]
        
        # Convert to numpy and ensure 2D array
        boxes = np.array(boxes).reshape(-1, 4)
        scores = np.array(scores).reshape(-1)
        
        # Model outputs normalized coordinates in [0,1]
        boxes = boxes.copy()
        
        # Model now guarantees x1<x2, y1<y2 through its architecture
        # No need to reorder coordinates
        
        # Denormalize to padded image size
        boxes = boxes * 640
        
        # Remove padding
        boxes[:, [0, 2]] -= self.pad_info['pad_w']
        boxes[:, [1, 3]] -= self.pad_info['pad_h']
        
        # Scale back to original size
        boxes = boxes / self.pad_info['scale']
        
        return boxes, scores
    
    def find_waldo(self, 
                   image: Union[str, Path, np.ndarray],
                   conf_threshold: float = 0.5,
                   visualize: bool = True) -> Dict[str, np.ndarray]:
        """Find Waldo in an image.
        
        Args:
            image: Input image (path or numpy array)
            conf_threshold: Confidence threshold for detection
            visualize: Whether to show visualization
            
        Returns:
            Dictionary containing detection results
        """
        # Preprocess
        processed = self._preprocess_image(image)
        
        # Run inference
        outputs = self.predict_fn(processed)
        
        # Print detailed debug info
        boxes = outputs['boxes'][0]
        scores = outputs['scores'][0]
        print("\nRaw Detection Results:")
        print(f"Coordinates (normalized):")
        print(f"  x1, y1: ({boxes[0,0]:.3f}, {boxes[0,1]:.3f})")
        print(f"  x2, y2: ({boxes[0,2]:.3f}, {boxes[0,3]:.3f})")
        print(f"Box size: {boxes[0,2]-boxes[0,0]:.3f} x {boxes[0,3]-boxes[0,1]:.3f}")
        print(f"Confidence: {scores[0,0]:.3%}")
        
        # Postprocess
        boxes, scores = self._postprocess_boxes(
            outputs['boxes'][0],  # Remove batch dimension
            outputs['scores'][0],
            conf_threshold
        )
        
        if len(boxes) == 0:
            print(f"Could not find Waldo :( (threshold: {conf_threshold})")
            return {'boxes': boxes, 'scores': scores}
        
        print(f"Found Waldo! (Confidence: {scores[0]:.2%})")
        
        if visualize:
            self.visualize(image, boxes[0], scores[0])
        
        return {
            'boxes': boxes,
            'scores': scores
        }
    
    def visualize(self,
                 image: Union[str, Path, np.ndarray],
                 box: np.ndarray,
                 score: float,
                 output_path: Optional[str] = None):
        """Visualize detection results.
        
        Args:
            image: Input image
            box: Bounding box coordinates
            score: Confidence score
            output_path: Optional path to save visualization
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure and axes
        fig = plt.figure(figsize=(12, 8))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        
        # Draw blurred boxes around detection
        height, width = image.shape[:2]
        
        # Normalize box coordinates
        box_norm = box / np.array([width, height, width, height])
        
        # Expand box by 50%
        box_norm += np.array([
            -(box_norm[2] - box_norm[0])/2,
            -(box_norm[3] - box_norm[1])/2,
            (box_norm[2] - box_norm[0])/2,
            (box_norm[3] - box_norm[1])/2
        ])
        
        # Draw blurred regions
        ax.add_patch(patches.Rectangle(
            (0, 0), box_norm[1]*width, height,
            linewidth=0, edgecolor='none', facecolor='w', alpha=0.8
        ))
        ax.add_patch(patches.Rectangle(
            (box_norm[3]*width, 0), width, height,
            linewidth=0, edgecolor='none', facecolor='w', alpha=0.8
        ))
        ax.add_patch(patches.Rectangle(
            (box_norm[1]*width, 0), (box_norm[3]-box_norm[1])*width, box_norm[0]*height,
            linewidth=0, edgecolor='none', facecolor='w', alpha=0.8
        ))
        ax.add_patch(patches.Rectangle(
            (box_norm[1]*width, box_norm[2]*height), (box_norm[3]-box_norm[1])*width, height,
            linewidth=0, edgecolor='none', facecolor='w', alpha=0.8
        ))
        
        # Draw bounding box
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add confidence score
        plt.text(
            box[0], box[1] - 10,
            f'Waldo: {score:.2%}',
            color='red',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Show image
        ax.imshow(image)
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        
        plt.close()

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Find Waldo in an image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best_model.pkl',
                      help='Path to model checkpoint')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                      help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    parser.add_argument('--no-viz', action='store_true',
                      help='Disable visualization')
    
    args = parser.parse_args()
    
    finder = WaldoFinder(args.model)
    finder.find_waldo(
        args.image_path,
        conf_threshold=args.conf_threshold,
        visualize=not args.no_viz
    )

if __name__ == '__main__':
    main()
