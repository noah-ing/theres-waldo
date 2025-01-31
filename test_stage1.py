"""Test Stage 1 classifier with sliding window approach."""

import jax
import jax.numpy as jnp
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import argparse

from waldo_finder.binary_classifier import WaldoClassifier
from waldo_finder.train_utils import load_checkpoint

def load_stage1_model(checkpoint_path: str) -> WaldoClassifier:
    """Load pre-trained stage 1 model."""
    model = WaldoClassifier()
    variables = load_checkpoint(checkpoint_path, check_version=False)  # Don't check version for stage1
    return model.bind({
        'params': variables['params'],
        'batch_stats': variables['batch_stats'],
    })

from tqdm import tqdm

def process_scale(
    image: np.ndarray,
    model: WaldoClassifier,
    scale: float,
    window_size: int,
    stride: int,
    conf_threshold: float,
    pbar: tqdm
) -> list:
    """Process image at a specific scale."""
    # Resize image
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    scaled_image = cv2.resize(image, (new_width, new_height))
    
    # Create JIT-compiled prediction function
    @jax.jit
    def predict_window(window):
        logits = model(window, training=False)
        return jax.nn.sigmoid(logits)[0, 0]
    
    detections = []
    for y in range(0, new_height - window_size + 1, stride):
        for x in range(0, new_width - window_size + 1, stride):
            # Extract window
            window = scaled_image[y:y + window_size, x:x + window_size]
            
            # Add batch dimension and normalize
            window = window.astype(np.float32) / 255.0
            window = window[None, ...]
            
            # Get prediction
            confidence = predict_window(window)
            
            if confidence > conf_threshold:
                # Convert back to original image coordinates
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + window_size) / scale)
                y2 = int((y + window_size) / scale)
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(confidence)
                })
            
            pbar.update(1)
    
    return detections

def non_max_suppression(boxes: list, iou_threshold: float = 0.5) -> list:
    """Apply non-max suppression to remove overlapping boxes."""
    if not boxes:
        return []
    
    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    
    # Initialize list of picked indexes
    keep = []
    
    # Calculate areas
    areas = [(b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]) for b in boxes]
    
    # Process boxes
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        
        if not boxes:
            break
        
        # Get coordinates of current box
        x1 = np.array([b['box'][0] for b in boxes])
        y1 = np.array([b['box'][1] for b in boxes])
        x2 = np.array([b['box'][2] for b in boxes])
        y2 = np.array([b['box'][3] for b in boxes])
        
        # Calculate overlap
        xx1 = np.maximum(current['box'][0], x1)
        yy1 = np.maximum(current['box'][1], y1)
        xx2 = np.minimum(current['box'][2], x2)
        yy2 = np.minimum(current['box'][3], y2)
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / (areas[0] + np.array([
            (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])
            for b in boxes
        ]) - w * h)
        
        # Remove boxes with overlap > threshold
        boxes = [box for i, box in enumerate(boxes) if overlap[i] <= iou_threshold]
    
    return keep

def sliding_window_detection(
    image: np.ndarray,
    model: WaldoClassifier,
    window_size: int = 128,
    stride: int = 64,
    conf_threshold: float = 0.5,
    scales: list = [0.5, 1.0, 2.0]  # Process at multiple scales
) -> list:
    """Run multi-scale sliding window detection."""
    all_detections = []
    
    # Calculate total windows across all scales
    total_windows = 0
    for scale in scales:
        height = int(image.shape[0] * scale)
        width = int(image.shape[1] * scale)
        n_windows_y = (height - window_size) // stride + 1
        n_windows_x = (width - window_size) // stride + 1
        total_windows += n_windows_y * n_windows_x
    
    # Create progress bar
    with tqdm(total=total_windows, desc="Processing windows") as pbar:
        # Process each scale
        for scale in scales:
            detections = process_scale(
                image, model, scale,
                window_size, stride,
                conf_threshold, pbar
            )
            all_detections.extend(detections)
    
    # Apply non-max suppression
    final_detections = non_max_suppression(all_detections)
    
    return final_detections

def visualize_detections(image: np.ndarray, detections: list, output_path: str = None):
    """Visualize detection results."""
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.text(
            x1, y1 - 5,
            f'{conf:.2%}',
            color='red',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--model', default='trained_model/stage1/best_model.pkl',
                      help='Path to Stage 1 model checkpoint')
    parser.add_argument('--window-size', type=int, default=128,
                      help='Sliding window size')
    parser.add_argument('--stride', type=int, default=64,
                      help='Sliding window stride')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                      help='Confidence threshold')
    parser.add_argument('--output', help='Path to save visualization')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_stage1_model(args.model)
    
    # Load and resize image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not load image: {args.image_path}")
    
    # Run detection
    print("Running sliding window detection...")
    detections = sliding_window_detection(
        image,
        model,
        window_size=args.window_size,
        stride=args.stride,
        conf_threshold=args.conf_threshold
    )
    
    print(f"\nFound {len(detections)} potential matches")
    
    # Visualize results
    visualize_detections(image, detections, args.output)
    print("Done!")

if __name__ == '__main__':
    main()
