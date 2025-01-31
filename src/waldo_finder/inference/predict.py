"""
Inference script for scene-level Waldo detection.
Provides visualization and confidence scoring for predictions.
"""

import torch
import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
import json

from ..training.trainer import WaldoTrainer

class WaldoPredictor:
    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config
        self.device = device
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        self.model.to(device)
        
    def _load_model(self, checkpoint_path: str) -> WaldoTrainer:
        """Load model from checkpoint"""
        model = WaldoTrainer.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            pretrain=False
        )
        return model
        
    def predict(
        self,
        image_path: str,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        visualize: bool = True,
        save_path: Optional[str] = None,
        debug: bool = False
    ) -> Dict:
        """
        Run Waldo detection on an image
        
        Args:
            image_path: Path to input image
            conf_threshold: Optional confidence threshold override
            nms_threshold: Optional NMS threshold override
            visualize: Whether to show visualization
            save_path: Optional path to save visualization
            debug: Whether to include debug information
            
        Returns:
            Dictionary containing detection results
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Prepare input tensor
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions, feature_maps = self.model(input_tensor)
            
        # Post-process predictions
        boxes, scores = self.model.detection_head.post_process(
            predictions,
            conf_threshold=conf_threshold or self.config.inference.confidence_threshold,
            nms_threshold=nms_threshold or self.config.inference.nms_threshold
        )
        
        # Convert normalized coordinates to pixels
        boxes = boxes.cpu().numpy()
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        scores = scores.cpu().numpy()
        
        # Prepare results
        results = {
            'boxes': boxes.tolist(),
            'scores': scores.tolist()
        }
        
        # Add debug information if requested
        if debug:
            results['debug'] = {
                'feature_maps': [f.cpu().numpy().tolist() for f in feature_maps],
                'raw_predictions': {
                    k: v.cpu().numpy().tolist()
                    for k, v in predictions.items()
                }
            }
        
        # Visualize if requested
        if visualize or save_path:
            vis_image = self._visualize_predictions(
                image.copy(),
                boxes,
                scores
            )
            
            if visualize:
                plt.figure(figsize=(12, 8))
                plt.imshow(vis_image)
                plt.axis('off')
                plt.show()
                
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(save_path),
                    cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                )
        
        return results
        
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize
        image = cv2.resize(
            image,
            (self.config.model.img_size, self.config.model.img_size)
        )
        
        # Normalize
        image = image / 255.0
        image = (image - self.config.augmentation.val.normalize.mean) / \
                self.config.augmentation.val.normalize.std
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float()
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
        
    def _visualize_predictions(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Visualize detection results on image"""
        # Draw each detection
        for box, score in zip(boxes, scores):
            # Convert to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            
            # Draw score
            label = f"{score:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
        return image

@hydra.main(config_path="../../../config", config_name="scene_model")
def main(config: DictConfig):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Waldo detection inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--conf-threshold", type=float, help="Confidence threshold")
    parser.add_argument("--nms-threshold", type=float, help="NMS threshold")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--save", help="Path to save visualization")
    parser.add_argument("--debug", action="store_true", help="Include debug information")
    parser.add_argument("--output", help="Path to save detection results JSON")
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WaldoPredictor(
        config=config,
        checkpoint_path=args.checkpoint
    )
    
    # Run prediction
    results = predictor.predict(
        image_path=args.image,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        visualize=not args.no_viz,
        save_path=args.save,
        debug=args.debug
    )
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nFound {len(results['boxes'])} potential Waldo locations:")
    for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        print(f"Detection {i+1}: confidence = {score:.2f}, box = {box}")

if __name__ == "__main__":
    main()
