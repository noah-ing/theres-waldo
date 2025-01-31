"""
Inference script for Waldo detection.
This is a wrapper around the new scene-level detection system.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Union
import hydra
from omegaconf import DictConfig

from .inference.predict import WaldoPredictor

def find_waldo(
    image_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    conf_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None,
    visualize: bool = True,
    save_path: Optional[str] = None,
    debug: bool = False,
    output_path: Optional[str] = None
) -> Dict:
    """
    Find Waldo in an image using the scene-level detection system.
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        conf_threshold: Optional confidence threshold override
        nms_threshold: Optional NMS threshold override
        visualize: Whether to show visualization
        save_path: Optional path to save visualization
        debug: Whether to include debug information
        output_path: Optional path to save detection results JSON
        
    Returns:
        Dictionary containing detection results
    """
    # Load config
    config = hydra.compose(config_name="scene_model")
    
    # Initialize predictor
    predictor = WaldoPredictor(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    # Run prediction
    results = predictor.predict(
        image_path=image_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        visualize=visualize,
        save_path=save_path,
        debug=debug
    )
    
    # Save results if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Find Waldo in an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--conf-threshold", type=float, help="Confidence threshold")
    parser.add_argument("--nms-threshold", type=float, help="NMS threshold")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--save", help="Path to save visualization")
    parser.add_argument("--debug", action="store_true", help="Include debug information")
    parser.add_argument("--output", help="Path to save detection results JSON")
    args = parser.parse_args()
    
    # Run detection
    results = find_waldo(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        visualize=not args.no_viz,
        save_path=args.save,
        debug=args.debug,
        output_path=args.output
    )
    
    # Print summary
    print(f"\nFound {len(results['boxes'])} potential Waldo locations:")
    for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        print(f"Detection {i+1}: confidence = {score:.2f}, box = {box}")

if __name__ == "__main__":
    main()
