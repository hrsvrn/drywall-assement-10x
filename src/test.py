"""
Test/Inference script for GroundingDINO + SAM model.
Generates prediction masks following the required naming format: {image_id}__{prompt}.png
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import GroundedSAM
import pandas as pd
from tqdm import tqdm

# Paths
GROUNDING_DINO_CONFIG = "../checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "../checkpoints/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT = "../checkpoints/grounded_sam_final.pth"  # Fine-tuned SAM weights
RESULTS_DIR = "../results/"
CSV_PATH = "../processed_datasets/dataset.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
print("Loading GroundingDINO + SAM model...")
model = GroundedSAM(
    grounding_dino_config_path=GROUNDING_DINO_CONFIG,
    grounding_dino_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    sam_checkpoint_path=SAM_CHECKPOINT,
    sam_encoder_version="vit_h",
    box_threshold=0.25,
    text_threshold=0.25,
    device=device
)
model.eval()
print("✅ Model loaded successfully!")

# Prompts for each task
PROMPTS = ["segment crack", "segment taping area"]


def predict_and_save(image_path, prompt, save_name=None):
    """
    Run segmentation and save overlay + mask with required naming format.
    
    Args:
        image_path: Path to input image
        prompt: Text prompt for segmentation
        save_name: Custom name for saving (defaults to image basename)
    """
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    
    # Run prediction
    pred_mask = model.predict(image, prompt)  # (H, W) float array [0, 1]
    
    # Binary mask (0 or 255)
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Create visualization overlay
    heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    # Generate save name following required format: {image_id}__{prompt}.png
    if save_name is None:
        image_id = os.path.basename(image_path).split('.')[0]
    else:
        image_id = save_name
    
    # Format prompt for filename (remove "segment " prefix and replace spaces with underscores)
    prompt_clean = prompt.replace("segment ", "").replace(" ", "_")
    
    # Save with required naming format
    mask_filename = f"{image_id}__{prompt_clean}.png"
    overlay_filename = f"{image_id}__{prompt_clean}_overlay.jpg"
    
    cv2.imwrite(os.path.join(RESULTS_DIR, mask_filename), binary_mask)
    cv2.imwrite(os.path.join(RESULTS_DIR, overlay_filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return binary_mask, overlay


def test_on_validation_set(num_samples=10):
    """Test on validation set samples"""
    df = pd.read_csv(CSV_PATH)
    val_df = df[df["split"] == "valid"].reset_index(drop=True)
    
    print(f"\nTesting on {min(num_samples, len(val_df))} validation samples...")
    
    for i in tqdm(range(min(num_samples, len(val_df)))):
        row = val_df.iloc[i]
        image_path = os.path.join("..", row["image_path"])
        prompt = row["prompt"]
        
        if os.path.exists(image_path):
            predict_and_save(image_path, prompt)


def test_on_specific_images():
    """Test on specific test images"""
    print("\nTesting on specific images...")
    
    # Example test images from both datasets
    TEST_IMAGES = [
        {
            "path": "../dataset/cracks-1/test/2056_jpg.rf.68cb444f62494b5756c4708d119a66da.jpg",
            "id": "crack_test_1",
            "prompts": ["segment crack"]
        },
        {
            "path": "../dataset/cracks-1/test/photo_6165758190491385217_y_jpg.rf.43cd195b51a519e64362afd0a255c18d.jpg",
            "id": "crack_test_2",
            "prompts": ["segment crack"]
        }
    ]
    
    for test_img in TEST_IMAGES:
        if os.path.exists(test_img["path"]):
            for prompt in test_img["prompts"]:
                print(f"Processing {test_img['id']} with prompt: {prompt}")
                predict_and_save(test_img["path"], prompt, save_name=test_img["id"])
        else:
            print(f"Warning: Image not found: {test_img['path']}")


def visualize_results(image_path, prompt):
    """Visualize results with matplotlib"""
    image = np.array(Image.open(image_path).convert("RGB"))
    pred_mask = model.predict(image, prompt)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Prediction mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title(f"Prediction: {prompt}")
    axes[1].axis("off")
    
    # Overlay
    heatmap = cv2.applyColorMap((pred_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "visualization_example.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run testing
    print("=" * 60)
    print("GroundingDINO + SAM Inference")
    print("=" * 60)
    
    # Test on validation set
    test_on_validation_set(num_samples=10)
    
    # Test on specific test images
    test_on_specific_images()
    
    print("\n✅ Inference complete! Results saved in:", RESULTS_DIR)
    print(f"   Mask format: {{image_id}}__{{prompt}}.png")
    print(f"   Total prompts tested: {PROMPTS}")
