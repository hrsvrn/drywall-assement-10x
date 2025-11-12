import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-7):
    """
    Dice loss for binary segmentation.
    
    Args:
        pred: Predicted mask (B, H, W) with values in [0, 1]
        target: Ground truth mask (B, H, W) with values in [0, 1]
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    # Ensure both are float and in correct range
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary segmentation.
    Helps with class imbalance by focusing on hard examples.
    
    Args:
        pred: Predicted mask (B, H, W) with values in [0, 1]
        target: Ground truth mask (B, H, W) with values in [0, 1]
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss value
    """
    # Ensure both are float and in correct range
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    
    return focal.mean()


def edge_loss(pred, target):
    """
    Edge-aware loss using Sobel operator.
    Helps preserve boundaries in segmentation.
    """
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32, device=pred.device).unsqueeze(0)
    pred_edge = F.conv2d(pred.unsqueeze(1), sobel_x, padding=1)
    target_edge = F.conv2d(target.unsqueeze(1), sobel_x, padding=1)
    return F.l1_loss(pred_edge, target_edge)


def hybrid_loss(pred, target):
    """
    Hybrid loss combining BCE, Dice, and Edge losses.
    Used for CLIPSeg training (backward compatibility).
    
    Args:
        pred: Model logits [batch, H, W]
        target: Ground truth mask [batch, 1, H, W] or [batch, H, W]
    """
    # Remove channel dimension from target and resize to match pred
    if target.dim() == 4:
        target = target.squeeze(1)  # [batch, H, W]
    
    # Resize target to match pred size if needed
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(
            target.unsqueeze(1), 
            size=pred.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
    
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = 1 - (2 * (pred.sigmoid() * target).sum() + 1) / (pred.sigmoid().sum() + target.sum() + 1)
    edge = edge_loss(pred.sigmoid(), target)
    
    return 0.6*bce + 0.3*dice + 0.1*edge
