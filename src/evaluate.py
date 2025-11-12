import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def evaluate(model, val_dl, device):
    """
    Evaluate CLIPSeg model (backward compatibility).
    """
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
            
            # Compute metrics per image in the batch
            batch_size = preds_bin.shape[0]
            for i in range(batch_size):
                pred_i = preds_bin[i]
                mask_i = masks[i]
                
                inter = (pred_i * mask_i).sum()
                union = pred_i.sum() + mask_i.sum() - inter
                
                # Dice score
                dice = (2 * inter + 1e-7) / (pred_i.sum() + mask_i.sum() + 1e-7)
                dice_scores.append(dice.item())
                
                # IoU score
                iou = (inter + 1e-7) / (union + 1e-7)
                iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)


def evaluate_grounded_sam(model, val_dl, device):
    """
    Evaluate GroundingDINO + SAM model.
    
    Args:
        model: GroundedSAM model instance
        val_dl: Validation dataloader
        device: Device to run on
        
    Returns:
        mean_dice: Average Dice score
        mean_iou: Average IoU score
    """
    model.eval()
    dice_scores, iou_scores = [], []
    
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            images = batch["images"]  # List of numpy arrays
            prompts = batch["prompts"]  # List of strings
            gt_masks = batch["masks"].to(device)  # (B, H, W)
            
            for i, (image, prompt) in enumerate(zip(images, prompts)):
                # Get prediction
                pred_mask = model.predict(image, prompt)  # (H, W) numpy array
                
                # Convert to tensor
                pred_mask = torch.tensor(pred_mask, device=device)
                gt_mask = gt_masks[i]
                
                # Resize prediction to match GT if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0).unsqueeze(0),
                        size=gt_mask.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                # Binarize prediction
                pred_bin = (pred_mask > 0.5).float()
                
                # Compute metrics
                inter = (pred_bin * gt_mask).sum()
                union = pred_bin.sum() + gt_mask.sum() - inter
                
                # Dice score
                dice = (2 * inter + 1e-7) / (pred_bin.sum() + gt_mask.sum() + 1e-7)
                dice_scores.append(dice.item())
                
                # IoU score
                iou = (inter + 1e-7) / (union + 1e-7)
                iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)


def evaluate_per_prompt(model, val_dl, device):
    """
    Evaluate GroundedSAM model with per-prompt metrics breakdown.
    
    Returns:
        Dictionary with metrics for each prompt type
    """
    model.eval()
    prompt_metrics = {}
    
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating per prompt"):
            images = batch["images"]
            prompts = batch["prompts"]
            gt_masks = batch["masks"].to(device)
            
            for i, (image, prompt) in enumerate(zip(images, prompts)):
                # Initialize metrics for this prompt if not exists
                if prompt not in prompt_metrics:
                    prompt_metrics[prompt] = {"dice": [], "iou": []}
                
                # Get prediction
                pred_mask = model.predict(image, prompt)
                pred_mask = torch.tensor(pred_mask, device=device)
                gt_mask = gt_masks[i]
                
                # Resize if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = F.interpolate(
                        pred_mask.unsqueeze(0).unsqueeze(0),
                        size=gt_mask.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                pred_bin = (pred_mask > 0.5).float()
                
                # Compute metrics
                inter = (pred_bin * gt_mask).sum()
                union = pred_bin.sum() + gt_mask.sum() - inter
                
                dice = (2 * inter + 1e-7) / (pred_bin.sum() + gt_mask.sum() + 1e-7)
                iou = (inter + 1e-7) / (union + 1e-7)
                
                prompt_metrics[prompt]["dice"].append(dice.item())
                prompt_metrics[prompt]["iou"].append(iou.item())
    
    # Compute averages
    results = {}
    for prompt, metrics in prompt_metrics.items():
        results[prompt] = {
            "dice": np.mean(metrics["dice"]),
            "iou": np.mean(metrics["iou"]),
            "num_samples": len(metrics["dice"])
        }
    
    return results
