import torch
import torch.nn.functional as F

def edge_loss(pred, target):
    sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32, device=pred.device).unsqueeze(0)
    pred_edge = F.conv2d(pred, sobel_x, padding=1)
    target_edge = F.conv2d(target, sobel_x, padding=1)
    return F.l1_loss(pred_edge, target_edge)

def hybrid_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = 1 - (2 * (pred.sigmoid() * target).sum() + 1) / (pred.sigmoid().sum() + target.sum() + 1)
    edge = edge_loss(pred.sigmoid(), target)
    return 0.6*bce + 0.3*dice + 0.1*edge
