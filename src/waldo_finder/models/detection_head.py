"""
Detection head for Waldo localization, implementing multi-scale detection
with context-aware confidence estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange, repeat

class WaldoDetectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_scales: int = 3,
        min_size: float = 0.02,
        max_size: float = 0.1,
        num_points: int = 1000
    ):
        super().__init__()
        self.num_scales = num_scales
        self.min_size = min_size
        self.max_size = max_size
        self.num_points = num_points
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Location regression
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 4)  # (x1, y1, x2, y2)
        )
        
        # Scale prediction
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2)  # (width, height)
        )
        
        # Context scoring
        self.context_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim // 2),  # 7 = 4 (box) + 2 (scale) + 1 (context)
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        features: torch.Tensor,
        feature_maps: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = features.shape[0]
        
        # Process features
        x = self.feature_processor(features)
        
        # Location regression
        boxes = self.location_head(x)
        boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]
        
        # Scale prediction
        scales = self.scale_head(x)
        scales = torch.sigmoid(scales) * (self.max_size - self.min_size) + self.min_size
        
        # Context scoring
        context_scores = self.context_head(x)
        context_scores = torch.sigmoid(context_scores)
        
        # Combine predictions for confidence estimation
        combined = torch.cat([
            boxes,
            scales,
            context_scores
        ], dim=-1)
        confidence = self.confidence_head(torch.cat([x, combined], dim=-1))
        
        return {
            'boxes': boxes,
            'scales': scales,
            'context_scores': context_scores,
            'confidence': confidence
        }
    
    def post_process(
        self,
        predictions: Dict[str, torch.Tensor],
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Post-process raw predictions to get final detections
        
        Args:
            predictions: Raw model predictions
            conf_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Tuple of (boxes, scores) after filtering and NMS
        """
        boxes = predictions['boxes']
        confidence = predictions['confidence']
        
        # Filter by confidence
        mask = confidence.squeeze(-1) > conf_threshold
        boxes = boxes[mask]
        scores = confidence[mask]
        
        if boxes.shape[0] == 0:
            return boxes, scores
        
        # Convert boxes to (x1, y1, x2, y2) format if needed
        if boxes.shape[-1] == 4:
            boxes_xyxy = boxes
        else:
            # If boxes are in center format, convert to corners
            boxes_xyxy = self._center_to_corners(boxes)
        
        # Apply NMS
        keep = self._nms(boxes_xyxy, scores.squeeze(-1), nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        
        return boxes, scores
    
    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert center format boxes to corner format"""
        x_c, y_c, w, h = boxes.unbind(-1)
        x1 = x_c - w/2
        y1 = y_c - h/2
        x2 = x_c + w/2
        y2 = y_c + h/2
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """Non-maximum suppression"""
        if boxes.shape[0] == 0:
            return boxes
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort boxes by scores
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            i = order[0]
            keep.append(i)
            
            # Compute IoU
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            ids = (ovr <= threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

class DetectionLoss(nn.Module):
    """Combined loss for Waldo detection training"""
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
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection losses
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            masks: Optional mask for valid predictions
            
        Returns:
            Dictionary of computed losses
        """
        # Extract predictions and targets
        pred_boxes = predictions['boxes']
        pred_scales = predictions['scales']
        pred_context = predictions['context_scores']
        pred_conf = predictions['confidence']
        
        target_boxes = targets['boxes']
        target_scales = targets['scales']
        target_context = targets['context_scores']
        target_conf = targets['confidence']
        
        # Apply masks if provided
        if masks is not None:
            pred_boxes = pred_boxes[masks]
            pred_scales = pred_scales[masks]
            pred_context = pred_context[masks]
            pred_conf = pred_conf[masks]
            
            target_boxes = target_boxes[masks]
            target_scales = target_scales[masks]
            target_context = target_context[masks]
            target_conf = target_conf[masks]
        
        # Compute individual losses
        box_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
        scale_loss = F.smooth_l1_loss(pred_scales, target_scales)
        context_loss = F.binary_cross_entropy_with_logits(
            pred_context, target_context
        )
        conf_loss = F.binary_cross_entropy(
            pred_conf, target_conf
        )
        
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
