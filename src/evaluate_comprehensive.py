#!/usr/bin/env python3
"""
Comprehensive evaluation script for GroundingDINO + SAM model.
Generates detailed metrics, per-prompt analysis, and visualizations.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from model import GroundedSAM
from dataset import GroundedSAMDataset, collate_fn
from evaluate import evaluate_grounded_sam, evaluate_per_prompt


def evaluate_and_save_results(
    model,
    val_dl,
    test_dl,
    device,
    output_dir="../results/evaluation"
):
    """
    Run comprehensive evaluation and save results.
    
    Args:
        model: GroundedSAM model instance
        val_dl: Validation dataloader
        test_dl: Test dataloader
        device: Device to run on
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "GroundingDINO + SAM",
        "device": str(device)
    }
    
    # ==================== VALIDATION SET EVALUATION ====================
    print("\n[1/4] Evaluating on validation set...")
    val_dice, val_miou = evaluate_grounded_sam(model, val_dl, device)
    
    results["validation"] = {
        "overall": {
            "dice": float(val_dice),
            "miou": float(val_miou)
        }
    }
    
    print(f"\nValidation Results:")
    print(f"  Overall Dice: {val_dice:.4f}")
    print(f"  Overall mIoU: {val_miou:.4f}")
    
    # Per-prompt validation metrics
    print("\n[2/4] Computing per-prompt metrics (validation)...")
    val_per_prompt = evaluate_per_prompt(model, val_dl, device)
    
    results["validation"]["per_prompt"] = {}
    print("\nPer-Prompt Validation Metrics:")
    for prompt, metrics in val_per_prompt.items():
        results["validation"]["per_prompt"][prompt] = {
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "num_samples": int(metrics["num_samples"])
        }
        print(f"\n  {prompt}:")
        print(f"    Dice: {metrics['dice']:.4f}")
        print(f"    IoU:  {metrics['iou']:.4f}")
        print(f"    Samples: {metrics['num_samples']}")
    
    # ==================== TEST SET EVALUATION ====================
    if test_dl is not None and len(test_dl) > 0:
        print("\n[3/4] Evaluating on test set...")
        test_dice, test_miou = evaluate_grounded_sam(model, test_dl, device)
        
        results["test"] = {
            "overall": {
                "dice": float(test_dice),
                "miou": float(test_miou)
            }
        }
        
        print(f"\nTest Results:")
        print(f"  Overall Dice: {test_dice:.4f}")
        print(f"  Overall mIoU: {test_miou:.4f}")
        
        # Per-prompt test metrics
        test_per_prompt = evaluate_per_prompt(model, test_dl, device)
        
        results["test"]["per_prompt"] = {}
        print("\nPer-Prompt Test Metrics:")
        for prompt, metrics in test_per_prompt.items():
            results["test"]["per_prompt"][prompt] = {
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "num_samples": int(metrics["num_samples"])
            }
            print(f"\n  {prompt}:")
            print(f"    Dice: {metrics['dice']:.4f}")
            print(f"    IoU:  {metrics['iou']:.4f}")
            print(f"    Samples: {metrics['num_samples']}")
    else:
        print("\n[3/4] Skipping test set (no data available)")
        results["test"] = None
    
    # ==================== SAVE RESULTS ====================
    print("\n[4/4] Saving results...")
    
    # Save JSON results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")
    
    # Create visualizations
    create_evaluation_plots(results, output_dir)
    
    # Create summary table
    create_summary_table(results, output_dir)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - evaluation_results.json")
    print(f"  - evaluation_summary.csv")
    print(f"  - metrics_comparison.png")
    print(f"  - per_prompt_metrics.png")
    
    return results


def create_evaluation_plots(results, output_dir):
    """Create visualization plots from evaluation results."""
    
    # Extract metrics for plotting
    metrics_data = []
    
    # Validation metrics
    if "validation" in results:
        val = results["validation"]["overall"]
        metrics_data.append({
            "Split": "Validation",
            "Prompt": "Overall",
            "Dice": val["dice"],
            "IoU": val["miou"]
        })
        
        for prompt, metrics in results["validation"]["per_prompt"].items():
            metrics_data.append({
                "Split": "Validation",
                "Prompt": prompt,
                "Dice": metrics["dice"],
                "IoU": metrics["iou"]
            })
    
    # Test metrics
    if results.get("test") is not None:
        test = results["test"]["overall"]
        metrics_data.append({
            "Split": "Test",
            "Prompt": "Overall",
            "Dice": test["dice"],
            "IoU": test["miou"]
        })
        
        for prompt, metrics in results["test"]["per_prompt"].items():
            metrics_data.append({
                "Split": "Test",
                "Prompt": prompt,
                "Dice": metrics["dice"],
                "IoU": metrics["iou"]
            })
    
    df = pd.DataFrame(metrics_data)
    
    # Plot 1: Overall metrics comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    overall_df = df[df["Prompt"] == "Overall"]
    if len(overall_df) > 0:
        x = np.arange(len(overall_df))
        width = 0.35
        
        axes[0].bar(x - width/2, overall_df["Dice"], width, label='Dice', color='#3498db')
        axes[0].bar(x + width/2, overall_df["IoU"], width, label='IoU', color='#2ecc71')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Overall Metrics Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(overall_df["Split"])
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Per-prompt metrics
    prompt_df = df[df["Prompt"] != "Overall"]
    if len(prompt_df) > 0:
        prompts = prompt_df["Prompt"].unique()
        x = np.arange(len(prompts))
        width = 0.35
        
        val_data = prompt_df[prompt_df["Split"] == "Validation"]
        test_data = prompt_df[prompt_df["Split"] == "Test"]
        
        if len(val_data) > 0:
            axes[1].bar(x - width/2, val_data["Dice"], width, label='Validation', color='#3498db')
        if len(test_data) > 0:
            axes[1].bar(x + width/2, test_data["Dice"], width, label='Test', color='#e74c3c')
        
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Per-Prompt Dice Scores')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([p.replace("segment ", "") for p in prompts], rotation=15, ha='right')
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Detailed per-prompt comparison (both Dice and IoU)
    if len(prompt_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for grouped bar chart
        splits = prompt_df["Split"].unique()
        prompts = prompt_df["Prompt"].unique()
        
        x = np.arange(len(prompts))
        width = 0.2
        
        for i, split in enumerate(splits):
            split_data = prompt_df[prompt_df["Split"] == split]
            offset = (i - len(splits)/2) * width + width/2
            
            dice_values = [split_data[split_data["Prompt"] == p]["Dice"].values[0] 
                          if len(split_data[split_data["Prompt"] == p]) > 0 else 0 
                          for p in prompts]
            
            ax.bar(x + offset, dice_values, width, label=f'{split} (Dice)', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Detailed Per-Prompt Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("segment ", "") for p in prompts], rotation=15, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_prompt_detailed.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    print("✓ Plots saved")


def create_summary_table(results, output_dir):
    """Create CSV summary table."""
    
    rows = []
    
    # Validation metrics
    if "validation" in results:
        val = results["validation"]
        rows.append({
            "Split": "Validation",
            "Prompt": "Overall",
            "Dice": val["overall"]["dice"],
            "mIoU": val["overall"]["miou"],
            "Samples": sum(m["num_samples"] for m in val["per_prompt"].values())
        })
        
        for prompt, metrics in val["per_prompt"].items():
            rows.append({
                "Split": "Validation",
                "Prompt": prompt,
                "Dice": metrics["dice"],
                "mIoU": metrics["iou"],
                "Samples": metrics["num_samples"]
            })
    
    # Test metrics
    if results.get("test") is not None:
        test = results["test"]
        rows.append({
            "Split": "Test",
            "Prompt": "Overall",
            "Dice": test["overall"]["dice"],
            "mIoU": test["overall"]["miou"],
            "Samples": sum(m["num_samples"] for m in test["per_prompt"].values())
        })
        
        for prompt, metrics in test["per_prompt"].items():
            rows.append({
                "Split": "Test",
                "Prompt": prompt,
                "Dice": metrics["dice"],
                "mIoU": metrics["iou"],
                "Samples": metrics["num_samples"]
            })
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Summary table saved to: {csv_path}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--checkpoint", default="../checkpoints/grounded_sam_final.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--csv", default="../processed_datasets/dataset.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--output", default="../results/evaluation",
                       help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = GroundedSAM(
        grounding_dino_config_path="../checkpoints/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="../checkpoints/groundingdino_swint_ogc.pth",
        sam_checkpoint_path=args.checkpoint,
        device=device
    )
    model.eval()
    print("✓ Model loaded")
    
    # Load datasets
    print("\nLoading datasets...")
    val_ds = GroundedSAMDataset(args.csv, split="valid")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)
    print(f"✓ Validation set: {len(val_ds)} samples")
    
    # Load test set if available
    try:
        test_ds = GroundedSAMDataset(args.csv, split="test")
        test_dl = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)
        print(f"✓ Test set: {len(test_ds)} samples")
    except:
        test_dl = None
        print("⚠ Test set not available")
    
    # Run evaluation
    results = evaluate_and_save_results(model, val_dl, test_dl, device, args.output)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

