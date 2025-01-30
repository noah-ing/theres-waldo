"""Loss functions for Waldo detection training."""

import jax
import jax.numpy as jnp
import optax
from typing import Tuple

def box_area(boxes: jnp.ndarray) -> jnp.ndarray:
    """Compute area of boxes.
    
    Args:
        boxes: Array of shape (..., 4) containing [x1, y1, x2, y2] coordinates
        
    Returns:
        areas: Array of shape (...) containing box areas
    """
    x1, y1, x2, y2 = jnp.split(boxes, 4, axis=-1)
    return jnp.squeeze((x2 - x1) * (y2 - y1), axis=-1)

def box_iou(boxes1: jnp.ndarray, boxes2: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute IoU and union between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) containing [x1, y1, x2, y2] coordinates
        boxes2: Array of shape (N, 4) containing [x1, y1, x2, y2] coordinates
        
    Returns:
        iou: Array of shape (N,) containing IoU values
        union: Array of shape (N,) containing union areas
    """
    # Get box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = jnp.split(boxes1, 4, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = jnp.split(boxes2, 4, axis=-1)
    
    # Calculate intersection coordinates
    x1 = jnp.maximum(b1_x1, b2_x1)
    y1 = jnp.maximum(b1_y1, b2_y1)
    x2 = jnp.minimum(b1_x2, b2_x2)
    y2 = jnp.minimum(b1_y2, b2_y2)
    
    # Calculate areas
    intersection = jnp.maximum(0, x2 - x1) * jnp.maximum(0, y2 - y1)
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = jnp.where(union > 0, intersection / union, 0.0)
    
    return iou, union

def compute_giou_loss(pred_boxes: jnp.ndarray, true_boxes: jnp.ndarray) -> jnp.ndarray:
    """Compute GIoU loss between predicted and ground truth boxes.
    
    Args:
        pred_boxes: Array of shape (N, 4) containing predicted [x1, y1, x2, y2] coordinates
        true_boxes: Array of shape (N, 4) containing ground truth [x1, y1, x2, y2] coordinates
        
    Returns:
        giou_loss: Array of shape (N,) containing GIoU loss values
    """
    # Calculate IoU and union
    iou, union = box_iou(pred_boxes, true_boxes)
    
    # Calculate enclosing box coordinates
    pred_x1, pred_y1, pred_x2, pred_y2 = jnp.split(pred_boxes, 4, axis=-1)
    true_x1, true_y1, true_x2, true_y2 = jnp.split(true_boxes, 4, axis=-1)
    
    enc_x1 = jnp.minimum(pred_x1, true_x1)
    enc_y1 = jnp.minimum(pred_y1, true_y1)
    enc_x2 = jnp.maximum(pred_x2, true_x2)
    enc_y2 = jnp.maximum(pred_y2, true_y2)
    
    # Calculate enclosing box area
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
    
    # Calculate GIoU
    giou = iou - jnp.where(
        enc_area > 0,
        (enc_area - union) / enc_area,
        0.0
    )
    
    # Return loss (1 - GIoU to convert from similarity to loss)
    return 1 - giou

def compute_size_loss(boxes: jnp.ndarray, min_size: float = 0.05, max_size: float = 0.3) -> jnp.ndarray:
    """Compute size constraint loss to maintain reasonable box dimensions.
    
    Args:
        boxes: Array of shape (N, 4) containing [x1, y1, x2, y2] coordinates
        min_size: Minimum allowed box size as fraction of image dimension
        max_size: Maximum allowed box size as fraction of image dimension
        
    Returns:
        size_loss: Array of shape (N,) containing size penalty values
    """
    # Calculate box dimensions
    widths = boxes[..., 2] - boxes[..., 0]  # x2 - x1
    heights = boxes[..., 3] - boxes[..., 1]  # y2 - y1
    
    # Calculate size penalties
    min_penalty = jnp.maximum(0, min_size - jnp.minimum(widths, heights))
    max_penalty = jnp.maximum(0, jnp.maximum(widths, heights) - max_size)
    
    return min_penalty + max_penalty

def compute_total_loss(
    pred_boxes: jnp.ndarray,
    true_boxes: jnp.ndarray,
    pred_conf: jnp.ndarray,
    true_conf: jnp.ndarray,
    config: dict
) -> Tuple[jnp.ndarray, dict]:
    """Compute total detection loss combining L1, GIoU, confidence, and size constraints.
    
    Args:
        pred_boxes: Predicted boxes of shape (batch_size, 4)
        true_boxes: Ground truth boxes of shape (batch_size, 4)
        pred_conf: Predicted confidence scores of shape (batch_size, 1)
        true_conf: Ground truth confidence of shape (batch_size, 1)
        config: Dictionary containing loss weights
        
    Returns:
        total_loss: Combined loss value
        metrics: Dictionary of individual loss components
    """
    # L1 loss for box coordinates
    l1_loss = jnp.mean(jnp.abs(pred_boxes - true_boxes))
    
    # GIoU loss for better box learning
    giou_loss = jnp.mean(compute_giou_loss(pred_boxes, true_boxes))
    
    # Binary cross entropy for confidence
    conf_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(pred_conf, true_conf)
    )
    
    # Combine losses (size loss removed since coordinate transform handles constraints)
    total_loss = (
        config.loss_weights.l1 * l1_loss +
        config.loss_weights.giou * giou_loss +
        config.loss_weights.confidence * conf_loss
    )
    
    metrics = {
        'l1_loss': l1_loss,
        'giou_loss': giou_loss,
        'conf_loss': conf_loss,
    }
    
    return total_loss, metrics
