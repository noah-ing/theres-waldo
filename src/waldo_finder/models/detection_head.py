"""Detection head for Waldo localization with scale-aware predictions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List, Tuple
from torchvision.ops import nms, box_iou

class DetectionOutput(NamedTuple):
    """Structured output for detection head to ensure tracing compatibility."""
    scores: torch.Tensor  # Confidence scores
    boxes: torch.Tensor   # Bounding box coordinates
    scales: torch.Tensor  # Scale predictions
    context_scores: torch.Tensor  # Context importance scores

class WaldoDetectionHead(nn.Module):
    """Multi-scale detection head for Waldo localization."""
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_scales: int,
        min_size: float = 0.02,
        max_size: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.min_size = min_size
        self.max_size = max_size
        
        # Detection layers
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        # Prediction heads
        self.score_head = nn.Conv2d(hidden_dim, 1, 1)  # Confidence
        self.box_head = nn.Conv2d(hidden_dim, 4, 1)    # Box regression
        self.scale_head = nn.Conv2d(hidden_dim, num_scales, 1)  # Scale prediction
        self.context_head = nn.Conv2d(hidden_dim, 1, 1)  # Context importance
        
    def forward(
        self,
        x: torch.Tensor,
        feature_maps: List[torch.Tensor]
    ) -> DetectionOutput:
        """Forward pass with structured output."""
        # Initial detection features
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        
        # Get predictions with gradient tracking
        scores = self.score_head(feat)  # [B, 1, H, W]
        boxes = self.box_head(feat)     # [B, 4, H, W]
        scales = self.scale_head(feat)   # [B, num_scales, H, W]
        context = self.context_head(feat) # [B, 1, H, W]
        
        # Normalize predictions and reshape
        B = feat.size(0)
        H, W = feat.size(2), feat.size(3)
        
        # Reshape boxes to [B, H*W, 4]
        boxes = torch.sigmoid(boxes)  # Normalize to [0,1]
        boxes = boxes.permute(0, 2, 3, 1).reshape(B, H*W, 4)
        
        # Reshape scales to [B, H*W, num_scales]
        scales = F.softmax(scales, dim=1)  # Scale probabilities
        scales = scales.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # Reshape scores and context to [B, H*W]
        scores = scores.squeeze(1).reshape(B, H*W)
        context = torch.sigmoid(context).squeeze(1).reshape(B, H*W)
        
        return DetectionOutput(
            scores=scores,
            boxes=boxes,
            scales=scales,
            context_scores=context
        )
        
    def post_process(
        self,
        detections: DetectionOutput,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Post-process detections with NMS."""
        batch_boxes = []
        batch_scores = []
        
        for i in range(detections.scores.size(0)):
            # Get batch item
            scores = detections.scores[i].flatten()
            boxes = detections.boxes[i].view(-1, 4)
            
            # Filter by confidence
            mask = scores > conf_threshold
            scores = scores[mask]
            boxes = boxes[mask]
            
            if len(scores) > 0:
                # Apply NMS
                keep = nms(boxes, scores, nms_threshold)
                scores = scores[keep]
                boxes = boxes[keep]
            
            batch_boxes.append(boxes)
            batch_scores.append(scores)
            
        return batch_boxes, batch_scores

class DetectionLoss(nn.Module):
    """Multi-component detection loss."""
    
    def __init__(
        self,
        box_weight: float = 1.0,
        scale_weight: float = 1.0,
        context_weight: float = 1.0,
        conf_weight: float = 1.0
    ):
        super().__init__()
        self.box_weight = box_weight
        self.scale_weight = scale_weight
        self.context_weight = context_weight
        self.conf_weight = conf_weight
        
        # Loss functions
        self.box_loss = nn.SmoothL1Loss(reduction='none')
        self.scale_loss = nn.CrossEntropyLoss(reduction='none')
        self.context_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(
        self,
        predictions: DetectionOutput,
        targets: dict
    ) -> dict:
        """Compute detection losses with shape alignment."""
        # Unpack predictions
        pred_scores = predictions.scores  # [B, H*W]
        pred_boxes = predictions.boxes    # [B, H*W, 4]
        pred_scales = predictions.scales  # [B, H, W, num_scales]
        pred_context = predictions.context_scores  # [B, H*W]
        
        # Unpack targets
        target_boxes = targets['boxes']         # [B, N, 4]
        target_scales = targets['scales']       # [B, N, 2]
        target_context = targets['context_scores']  # [B, N]
        target_conf = targets['confidence']     # [B, N]
        
        # Get batch size and number of predictions per image
        B = pred_boxes.size(0)
        P = pred_boxes.size(1)  # H*W
        N = target_boxes.size(1)  # Number of target boxes
        
        # Reshape predictions to match targets
        pred_boxes = pred_boxes.view(B, -1, 4)  # [B, P, 4]
        pred_scales = pred_scales.view(B, -1, pred_scales.size(-1))  # [B, P, num_scales]
        pred_context = pred_context.view(B, -1)  # [B, P]
        pred_scores = predictions.scores.view(B, -1)  # [B, P]

        # Collect per-batch losses
        batch_box_losses = []
        batch_scale_losses = []
        batch_context_losses = []
        batch_conf_losses = []
        
        for b in range(B):
            # For each target box, find the best matching prediction
            target_boxes_b = target_boxes[b]  # [N, 4]
            pred_boxes_b = pred_boxes[b]  # [P, 4]
            
            # Compute IoU between all predictions and targets
            ious = box_iou(pred_boxes_b, target_boxes_b)  # [P, N]
            
            # For each target, find best matching prediction
            best_ious, best_idx = ious.max(dim=0)  # [N]
            valid_mask = best_ious > 0.5
            
            if valid_mask.any():
                # Compute box loss for valid matches
                batch_box_losses.append(self.box_loss(
                    pred_boxes_b[best_idx[valid_mask]],
                    target_boxes_b[valid_mask]
                ).mean())
                
                # Compute scale loss
                batch_scale_losses.append(self.scale_loss(
                    pred_scales[b, best_idx[valid_mask]],
                    target_scales[b, valid_mask].long()
                ).mean())
                
                # Compute context loss
                batch_context_losses.append(self.context_loss(
                    pred_context[b, best_idx[valid_mask]],
                    target_context[b, valid_mask]
                ).mean())
                
                # Create confidence targets (1 for matched predictions, 0 for others)
                conf_target = torch.zeros_like(pred_scores[b])
                conf_target[best_idx[valid_mask]] = 1.0
                batch_conf_losses.append(self.conf_loss(
                    pred_scores[b],
                    conf_target
                ).mean())
            else:
                # For batches with no valid matches, create small regularization losses
                # This ensures gradients are always computed
                device = pred_boxes.device
                
                # L2 regularization on predictions
                box_reg_loss = (pred_boxes_b ** 2).mean()
                scale_reg_loss = (pred_scales[b] ** 2).mean()
                context_reg_loss = (pred_context[b] ** 2).mean()
                conf_reg_loss = (pred_scores[b] ** 2).mean()
                
                # Small constant factor to ensure non-zero gradients
                eps = 1e-6
                
                batch_box_losses.append(box_reg_loss * eps)
                batch_scale_losses.append(scale_reg_loss * eps)
                batch_context_losses.append(context_reg_loss * eps)
                batch_conf_losses.append(conf_reg_loss * eps)
        
        # Stack and average losses over batch
        box_loss = torch.stack(batch_box_losses).mean()
        scale_loss = torch.stack(batch_scale_losses).mean()
        context_loss = torch.stack(batch_context_losses).mean()
        conf_loss = torch.stack(batch_conf_losses).mean()
        
        # Combine losses
        total_loss = (
            self.box_weight * box_loss +
            self.scale_weight * scale_loss +
            self.context_weight * context_loss +
            self.conf_weight * conf_loss
        )
        
        return {
            'loss': total_loss,
            'box_loss': box_loss,
            'scale_loss': scale_loss,
            'context_loss': context_loss,
            'conf_loss': conf_loss
        }
