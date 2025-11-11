import torch
import torch.nn.functional as F
import numpy as np

def evaluate(model, val_dl, device):
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for batch in val_dl:
            masks = batch.pop("mask").to(device)
            inputs = {k:v.to(device) for k,v in batch.items()}
            preds = torch.sigmoid(model(**inputs).logits)
            
            # Remove channel dimension from masks and resize to match preds
            masks = masks.squeeze(1)
            if masks.shape[-2:] != preds.shape[-2:]:
                masks = F.interpolate(
                    masks.unsqueeze(1), 
                    size=preds.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            preds_bin = (preds > 0.5).float()
            inter = (preds_bin * masks).sum()
            dice = (2 * inter + 1e-7) / (preds_bin.sum() + masks.sum() + 1e-7)
            iou = inter / ((preds_bin + masks - preds_bin*masks).sum() + 1e-7)
            dice_scores.append(dice.item()); iou_scores.append(iou.item())
    return np.mean(dice_scores), np.mean(iou_scores)
