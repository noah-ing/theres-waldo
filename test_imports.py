try:
    import jax
    import jaxlib
    import flax
    import optax
    import cv2
    import PIL
    import matplotlib
    import plotly
    import wandb
    import hydra
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
