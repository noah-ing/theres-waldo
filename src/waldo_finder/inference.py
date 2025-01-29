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
from typing import Dict, Optional, Tuple, Union, List
import pickle
from flax.serialization import msgpack_restore
import csv

from waldo_finder.model import WaldoDetector, create_train_state

class WaldoFinder:
    """Modern interface for Waldo detection."""
    
    # Class variable to store ground truth annotations
    ground_truth = {}
    
    @classmethod
    def load_annotations(cls):
        """Load ground truth annotations from CSV."""
        cls.ground_truth = {}
        with open('annotations/annotations.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['filename']
                if img_name not in cls.ground_truth:
                    cls.ground_truth[img_name] = []
                cls.ground_truth[img_name].append({
                    'xmin': int(row['xmin']),
                    'ymin': int(row['ymin']),
                    'xmax': int(row['xmax']),
                    'ymax': int(row['ymax'])
                })
    
    def __init__(self, model_path: Union[str, Path]):
        """Initialize WaldoFinder.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        # Load annotations if not already loaded
        if not self.ground_truth:
            self.load_annotations()
            
        self.model_path = Path(model_path)
        
        # Initialize model with simpler config
        self.model = WaldoDetector(
            num_heads=8,
            num_layers=8,
            hidden_dim=512,
            mlp_dim=2048,
            dropout_rate=0.1
        )

        # Create initial state
        rng = jax.random.PRNGKey(0)
        state = create_train_state(
            rng,
            learning_rate=1e-4,  # Doesn't matter for inference
            model_kwargs={
                'num_heads': 8,
                'num_layers': 8,
                'hidden_dim': 512,
                'mlp_dim': 2048,
                'dropout_rate': 0.1
            }
        )
        
        # Load and deserialize parameters with proper error handling
        try:
            with open(self.model_path, 'rb') as f:
                save_dict = pickle.load(f)
                
                # Deserialize parameters
                if isinstance(save_dict['params'], bytes):
                    params = msgpack_restore(save_dict['params'])
                else:
                    params = save_dict['params']
                
                self.variables = {'params': params}
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Create inference PRNG key
        self.inference_rng = jax.random.PRNGKey(42)
        
        # JIT compile inference function with deterministic mode
        self.predict_fn = jax.jit(self._predict)
    
    def _preprocess_image(self,
                         image: Union[str, Path, np.ndarray],
                         target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Preprocess image for inference."""
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
        # Generate new PRNG key for each prediction
        self.inference_rng, dropout_rng = jax.random.split(self.inference_rng)
        
        return self.model.apply(
            self.variables,
            image[None],  # Add batch dimension
            training=False,
            mutable=False,
            rngs={'dropout': dropout_rng}
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
                   visualize: bool = True,
                   no_blur: bool = False) -> Dict[str, np.ndarray]:
        """Find Waldo in an image."""
        # Preprocess
        processed = self._preprocess_image(image)
        
        # Run inference
        outputs = self.predict_fn(processed)
        
        # Process outputs
        boxes = np.array(outputs['boxes'][0])  # Shape: [4] for single box
        scores = np.array(outputs['scores'][0])  # Shape: [1] for single confidence
        
        # Postprocess boxes
        boxes, scores = self._postprocess_boxes(
            boxes[None],  # Add dimension to match expected shape
            scores[None],
            conf_threshold
        )
        
        if len(boxes) == 0:
            print(f"\nNo detection above confidence threshold ({conf_threshold:.0%})")
            return {'boxes': boxes, 'scores': scores}
        
        print(f"\nFound Waldo with {scores[0]:.1%} confidence!")
        
        if visualize:
            self.visualize(image, boxes[0], scores[0], no_blur=no_blur)
        
        return {
            'boxes': boxes,
            'scores': scores
        }
    
    def visualize(self,
                 image: Union[str, Path, np.ndarray],
                 box: np.ndarray,
                 score: float,
                 output_path: Optional[str] = None,
                 no_blur: bool = False):
        """Visualize detection results."""
        # Store original image path if provided
        self.image_path = None
        if isinstance(image, (str, Path)):
            self.image_path = str(Path(image))
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure and axes
        fig = plt.figure(figsize=(12, 8))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        
        height, width = image.shape[:2]
        
        if not no_blur:
            # Normalize box coordinates for blurring
            box_norm = box / np.array([width, height, width, height])
            
            # Expand box by 20% for better visibility
            box_norm += np.array([
                -(box_norm[2] - box_norm[0])/5,
                -(box_norm[3] - box_norm[1])/5,
                (box_norm[2] - box_norm[0])/5,
                (box_norm[3] - box_norm[1])/5
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
        
        # Draw model prediction box (red)
        rect_pred = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            label='Prediction'
        )
        ax.add_patch(rect_pred)
        
        # Get ground truth coordinates from loaded annotations
        if self.image_path:
            image_name = Path(self.image_path).name
            truth_boxes = self.ground_truth.get(image_name, [])
            
            # Draw all ground truth boxes (green)
            for truth in truth_boxes:
                rect_truth = patches.Rectangle(
                    (truth['xmin'], truth['ymin']),
                    truth['xmax'] - truth['xmin'],
                    truth['ymax'] - truth['ymin'],
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none',
                    label='Ground Truth' if truth == truth_boxes[0] else None  # Only label first box
                )
                ax.add_patch(rect_truth)
        
        # Add confidence score with better visibility
        plt.text(
            box[0], box[1] - 5,
            f'Prediction: {score:.2%}',
            color='red',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, 
                     edgecolor='white', linewidth=2)
        )
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
                 facecolor='white', edgecolor='white')
        
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
    parser.add_argument('--no-blur', action='store_true',
                      help='Disable blurring effect')
    
    args = parser.parse_args()
    
    finder = WaldoFinder(args.model)
    finder.find_waldo(
        args.image_path,
        conf_threshold=args.conf_threshold,
        visualize=not args.no_viz,
        no_blur=args.no_blur
    )

if __name__ == '__main__':
    main()
