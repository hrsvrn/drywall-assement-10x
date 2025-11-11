import torch
import torch.nn.functional as F

def edge_loss(pred, target):
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32, device=pred.device).unsqueeze(0)
    pred_edge = F.conv2d(pred.unsqueeze(1), sobel_x, padding=1)
    target_edge = F.conv2d(target.unsqueeze(1), sobel_x, padding=1)
    return F.l1_loss(pred_edge, target_edge)

def hybrid_loss(pred, target):
    # pred shape: [batch, H, W] - model output (e.g., 352x352)
    # target shape: [batch, 1, H, W] - ground truth mask
    
    # Remove channel dimension from target and resize to match pred
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
