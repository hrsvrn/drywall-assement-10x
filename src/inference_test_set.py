#!/usr/bin/env python3
"""
Generate predictions for test set with required naming format.
Outputs: {image_id}__{prompt}.png

This script generates all predictions needed for final submission.
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from model import GroundedSAM


def generate_test_predictions(
    model,
    csv_path,
    output_dir="../results/test_predictions",
    device="cuda"
):
    """
    Generate predictions for all test images with proper naming.
    
    Args:
        model: GroundedSAM model instance
        csv_path: Path to dataset CSV
        output_dir: Directory to save predictions
        device: Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data from CSV
    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    if len(test_df) == 0:
        print("❌ No test data found in dataset CSV!")
        print("Please run: python main.py --mode preprocess")
        return []
    
    print("="*70)
    print("TEST SET INFERENCE")
    print("="*70)
    print(f"Test samples: {len(test_df)}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Get root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    predictions = []
    
    # Process each test image
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
        image_path = os.path.join(root_dir, row["image_path"])
        prompt = row["prompt"]
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # Get prediction
        try:
            pred_mask = model.predict(image, prompt)  # (H, W) numpy array [0, 1]
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
        # Binary mask (0 or 255)
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        
        # Generate filename: {image_id}__{prompt}.png
        image_id = Path(image_path).stem  # Get filename without extension
        prompt_clean = prompt.replace("segment ", "").replace(" ", "_")
        mask_filename = f"{image_id}__{prompt_clean}.png"
        
        # Save mask
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, binary_mask)
        
        # Also save visualization (optional)
        heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        overlay_filename = f"{image_id}__{prompt_clean}_overlay.jpg"
        cv2.imwrite(os.path.join(output_dir, overlay_filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        predictions.append({
            "image_id": image_id,
            "prompt": prompt,
            "mask_file": mask_filename,
            "overlay_file": overlay_filename,
            "original_image": image_path
        })
    
    # Save prediction manifest
    manifest_path = os.path.join(output_dir, "predictions_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"Generated {len(predictions)} predictions")
    print(f"Saved to: {output_dir}")
    print(f"  - Binary masks: {image_id}__{prompt}.png")
    print(f"  - Overlays: {image_id}__{prompt}_overlay.jpg")
    print(f"  - Manifest: predictions_manifest.json")
    
    return predictions


def verify_naming_format(output_dir):
    """Verify that all predictions follow required naming format."""
    mask_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and '__' in f]
    
    print("\n" + "="*70)
    print("NAMING FORMAT VERIFICATION")
    print("="*70)
    
    required_format = "{image_id}__{prompt}.png"
    print(f"Required format: {required_format}")
    print(f"Found {len(mask_files)} mask files")
    
    valid = 0
    invalid = []
    
    for filename in mask_files:
        if '__' in filename:
            parts = filename.split('__')
            if len(parts) == 2 and parts[1].endswith('.png'):
                valid += 1
            else:
                invalid.append(filename)
        else:
            invalid.append(filename)
    
    print(f"\n✓ Valid: {valid}")
    if invalid:
        print(f"✗ Invalid: {len(invalid)}")
        for f in invalid[:5]:  # Show first 5
            print(f"  - {f}")
    else:
        print("✓ All files follow required format!")
    
    return valid, invalid


def create_submission_summary(predictions, output_dir):
    """Create a summary CSV for submission."""
    
    df = pd.DataFrame(predictions)
    summary_path = os.path.join(output_dir, "submission_summary.csv")
    df.to_csv(summary_path, index=False)
    
    print(f"\n✓ Submission summary saved: {summary_path}")
    
    # Group by prompt
    prompts = df["prompt"].value_counts()
    print("\nPredictions by prompt:")
    for prompt, count in prompts.items():
        print(f"  {prompt}: {count} images")


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate test set predictions with required naming format"
    )
    parser.add_argument("--checkpoint", default="../checkpoints/grounded_sam_final.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--csv", default="../processed_datasets/dataset.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--output", default="../results/test_predictions",
                       help="Output directory for predictions")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing predictions (don't generate)")
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Just verify naming format
        verify_naming_format(args.output)
        return
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nPlease train the model first:")
        print("  python main.py --mode train")
        return
    
    # Load model
    print("\nLoading model...")
    try:
        model = GroundedSAM(
            grounding_dino_config_path="../checkpoints/GroundingDINO_SwinT_OGC.py",
            grounding_dino_checkpoint_path="../checkpoints/groundingdino_swint_ogc.pth",
            sam_checkpoint_path=args.checkpoint,
            device=device
        )
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate predictions
    predictions = generate_test_predictions(model, args.csv, args.output, device)
    
    if predictions:
        # Verify naming format
        verify_naming_format(args.output)
        
        # Create submission summary
        create_submission_summary(predictions, args.output)
        
        print("\n✅ Test set inference complete!")
        print(f"\nSubmission files ready in: {args.output}")
    else:
        print("\n❌ No predictions generated")


if __name__ == "__main__":
    main()

