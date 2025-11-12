"""
Visualization utilities for GroundingDINO + SAM predictions.
"""

import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import wandb
import numpy as np


def visualize(image_path, prompt, model, device, save_path=None):
    """
    Visualize GroundedSAM prediction.
    
    Args:
        image_path: Path to input image
        prompt: Text prompt for segmentation
        model: GroundedSAM model instance
        device: Device to run on
        save_path: Optional path to save visualization
    """
    # Get root directory if path is relative
    if not os.path.isabs(image_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(root_dir, image_path)
    
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # Get prediction
    pred_mask = model.predict(image, prompt)  # (H, W) numpy array [0, 1]
    
    # Create visualizations
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Prediction (Continuous)")
    axes[1].axis("off")
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title("Binary Mask")
    axes[2].axis("off")
    
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay: {prompt}")
    axes[3].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Log to W&B if available
    try:
        wandb.log({
            "Prediction": [
                wandb.Image(image, caption=f"Original: {prompt}"),
                wandb.Image(overlay, caption=f"Prediction: {prompt}")
            ]
        })
    except:
        pass  # W&B not initialized


def visualize_with_gt(image_path, mask_path, prompt, model, device, save_path=None):
    """
    Visualize prediction alongside ground truth.
    
    Args:
        image_path: Path to input image
        mask_path: Path to ground truth mask
        prompt: Text prompt for segmentation
        model: GroundedSAM model instance
        device: Device to run on
        save_path: Optional path to save visualization
    """
    # Get root directory if path is relative
    if not os.path.isabs(image_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_path = os.path.join(root_dir, image_path)
        mask_path = os.path.join(root_dir, mask_path)
    
    # Load image and GT mask
    image = np.array(Image.open(image_path).convert("RGB"))
    gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0
    
    # Get prediction
    pred_mask = model.predict(image, prompt)
    
    # Calculate metrics
    pred_bin = (pred_mask > 0.5).astype(float)
    intersection = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum() - intersection
    dice = (2 * intersection + 1e-7) / (pred_bin.sum() + gt_mask.sum() + 1e-7)
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}")
    axes[2].axis("off")
    
    # Overlay comparison (GT in green, Pred in red, overlap in yellow)
    comparison = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    comparison[..., 1] = (gt_mask * 255).astype(np.uint8)  # GT in green
    comparison[..., 0] = (pred_bin * 255).astype(np.uint8)  # Pred in red
    # Overlap will appear yellow
    
    axes[3].imshow(comparison)
    axes[3].set_title("Comparison\n(GT=Green, Pred=Red, Both=Yellow)")
    axes[3].axis("off")
    
    plt.suptitle(f"Prompt: {prompt}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return dice, iou


def create_comparison_grid(model, dataset, device, num_samples=4, save_path=None):
    """
    Create a grid visualization comparing multiple samples.
    
    Args:
        model: GroundedSAM model instance
        dataset: Dataset instance
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        image = sample["image"]
        prompt = sample["prompt"]
        gt_mask = sample["mask"].numpy()
        
        # Get prediction
        pred_mask = model.predict(image, prompt)
        
        # Resize prediction to match GT if needed
        if pred_mask.shape != gt_mask.shape:
            from scipy.ndimage import zoom
            scale = (gt_mask.shape[0] / pred_mask.shape[0], 
                    gt_mask.shape[1] / pred_mask.shape[1])
            pred_mask = zoom(pred_mask, scale, order=1)
        
        pred_bin = (pred_mask > 0.5).astype(float)
        
        # Calculate metrics
        intersection = (pred_bin * gt_mask).sum()
        dice = (2 * intersection + 1e-7) / (pred_bin.sum() + gt_mask.sum() + 1e-7)
        
        # Plot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f"Prediction (Dice: {dice:.3f})")
        axes[i, 2].axis("off")
        
        # Overlay
        heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f"Overlay")
        axes[i, 3].axis("off")
        
        # Add prompt as y-label
        axes[i, 0].set_ylabel(prompt, fontsize=10, rotation=0, ha='right', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
